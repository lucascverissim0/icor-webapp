"""Deterministic acquisition of finalized annual EEA CO2 monitoring aggregates."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.parse import urlencode, urlsplit

from icor.evidence.sources.eea import ANNUAL_AGGREGATE_SCHEMA

API_URL = "https://discodata.eea.europa.eu/sql"
_COMBINED_TABLE = "[CO2Emission].[latest].[co2cars]"
_MAX_RESPONSE_BYTES = 250 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AnnualRelease:
    year: int
    version: str
    table: str
    expected_source_rows: int
    expected_accepted_rows: int
    expected_rejected_rows: int
    expected_registrations: int


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    path: Path
    artifact_bytes: int
    sha256: str
    group_count: int
    source_row_count: int
    accepted_row_count: int
    rejected_row_count: int
    registration_count: int


class _Response(Protocol):
    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> object: ...

    def read(self) -> bytes: ...

    def geturl(self) -> str: ...


_COUNTS = {
    2010: (285_764, 282_966, 2_798, 12_939_010),
    2011: (331_580, 324_215, 7_365, 12_424_653),
    2012: (364_385, 345_402, 18_983, 11_404_763),
    2013: (442_475, 421_639, 20_836, 11_823_538),
    2014: (417_939, 396_163, 21_776, 12_527_273),
    2015: (440_645, 432_296, 8_349, 13_758_903),
    2016: (494_123, 491_778, 2_345, 14_715_814),
    2017: (4_955_599, 4_948_947, 6_652, 15_116_551),
    2018: (15_272_915, 15_233_794, 39_121, 15_234_152),
    2019: (15_499_728, 15_493_649, 6_079, 15_493_706),
    2020: (11_742_439, 11_709_622, 32_817, 11_709_622),
    2021: (9_920_521, 9_907_308, 13_213, 9_907_308),
    2022: (9_479_544, 9_396_410, 83_134, 9_396_410),
    2023: (10_734_898, 10_734_228, 670, 10_734_228),
}


def _annual_release(year: int) -> AnnualRelease:
    version = f"v{2 * (year - 2009)}"
    table = (
        _COMBINED_TABLE
        if year <= 2019
        else f"[CO2Emission].[latest].[co2cars_{year}F{version}]"
    )
    raw, accepted, rejected, registrations = _COUNTS[year]
    return AnnualRelease(year, version, table, raw, accepted, rejected, registrations)


ANNUAL_RELEASES = {year: _annual_release(year) for year in range(2010, 2024)}


def acquire_year(
    release: AnnualRelease,
    destination: Path,
    *,
    opener: Callable[..., _Response] = urllib.request.urlopen,
    page_size: int = 100_000,
) -> AcquisitionResult:
    """Export one canonical annual aggregate atomically from the official API."""

    if type(page_size) is not int or page_size <= 0 or page_size > 100_000:
        raise ValueError("EEA page size must be between 1 and 100000")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    group_count = source_rows = accepted_rows = rejected_rows = registrations = 0
    query = _query(release)
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=destination.parent
        ) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(
                output,
                ANNUAL_AGGREGATE_SCHEMA,
                delimiter=";",
                lineterminator="\n",
            )
            writer.writeheader()
            page = 1
            while True:
                rows = _fetch_page(query, page, page_size, opener)
                if not rows:
                    break
                if len(rows) > page_size:
                    raise ValueError("EEA API returned more rows than requested")
                for row in rows:
                    canonical = _canonical_row(row, release)
                    writer.writerow(canonical)
                    group_count += 1
                    row_count = int(canonical["SourceRows"])
                    source_rows += row_count
                    if all(canonical[field].strip() for field in ("MS", "Mk", "Cn")):
                        accepted_rows += row_count
                        registrations += int(canonical["Registrations"])
                    else:
                        rejected_rows += row_count
                page += 1
            output.flush()
            os.fsync(output.fileno())

        actual = (source_rows, accepted_rows, rejected_rows, registrations)
        expected = (
            release.expected_source_rows,
            release.expected_accepted_rows,
            release.expected_rejected_rows,
            release.expected_registrations,
        )
        if actual != expected:
            raise ValueError(f"EEA aggregate totals do not match pinned metadata: {actual!r}")
        assert temporary is not None
        artifact_bytes = temporary.stat().st_size
        sha256 = _sha256(temporary)
        os.replace(temporary, destination)
        temporary = None
        return AcquisitionResult(
            destination,
            artifact_bytes,
            sha256,
            group_count,
            source_rows,
            accepted_rows,
            rejected_rows,
            registrations,
        )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _query(release: AnnualRelease) -> str:
    fields = "[Year], [Status], [Version_file], [MS], [Mk], [Cn], [TAN], [T], [Va], [Ve], [Ft]"
    return (
        f"SELECT {fields}, SUM(CAST([R] AS BIGINT)) AS [Registrations], "
        f"COUNT(*) AS [SourceRows] FROM {release.table} "
        f"WHERE [Year] = {release.year} AND [Status] = 'F' "
        f"AND [Version_file] = '{release.version}' GROUP BY {fields} ORDER BY {fields}"
    )


def _fetch_page(
    query: str,
    page: int,
    page_size: int,
    opener: Callable[..., _Response],
) -> list[dict[str, object]]:
    url = f"{API_URL}?{urlencode({'query': query, 'p': page, 'nrOfHits': page_size})}"
    request = urllib.request.Request(url, headers={"User-Agent": "ICOR-evidence/1"})
    with opener(request, timeout=120) as response:
        actual = urlsplit(response.geturl())
        if actual.scheme != "https" or actual.netloc != "discodata.eea.europa.eu":
            raise ValueError("EEA API redirected outside its allowlisted origin")
        payload_bytes = response.read()
    if len(payload_bytes) > _MAX_RESPONSE_BYTES:
        raise ValueError("EEA API response exceeds the byte limit")
    payload = json.loads(payload_bytes)
    if not isinstance(payload, dict) or "errors" in payload:
        raise ValueError("EEA API returned an error response")
    rows = payload.get("results")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("EEA API response schema is unsupported")
    return rows


def _canonical_row(row: dict[str, object], release: AnnualRelease) -> dict[str, str]:
    if set(row) != set(ANNUAL_AGGREGATE_SCHEMA):
        raise ValueError("EEA API row schema is unsupported")
    canonical = {
        field: "" if row[field] is None else str(row[field])
        for field in ANNUAL_AGGREGATE_SCHEMA
    }
    if canonical["Year"] != str(release.year):
        raise ValueError("EEA API row year is unexpected")
    if canonical["Status"] != "F" or canonical["Version_file"] != release.version:
        raise ValueError("EEA API row release status or version is unexpected")
    for field in ("Registrations", "SourceRows"):
        value = canonical[field]
        if not value.isascii() or not value.isdigit():
            raise ValueError(f"EEA API {field} is not a non-negative integer")
    if int(canonical["SourceRows"]) == 0:
        raise ValueError("EEA API SourceRows must be positive")
    return canonical


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

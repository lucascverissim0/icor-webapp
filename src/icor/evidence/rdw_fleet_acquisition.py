"""Deterministic acquisition of the RDW registered passenger-car fleet by cohort."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import unicodedata
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.parse import urlencode, urlsplit

from icor.evidence.sources.rdw import SCHEMA, classify_group

API_URL = "https://opendata.rdw.nl/resource/m9d7-ebf2.csv"
_ALLOWED_HOST = "opendata.rdw.nl"
_MAX_PAGE_SIZE = 50_000
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
# The pinned stock date's year; cohorts beyond it are rejected like in the loader.
_LATEST_COHORT_YEAR = 2026
_QUERY = {
    "$select": (
        "merk, handelsbenaming, date_extract_y(datum_eerste_toelating_dt) as cohort, "
        "count(*) as vehicles"
    ),
    "$where": "voertuigsoort='Personenauto' AND tenaamstellen_mogelijk='Ja'",
    "$group": "merk, handelsbenaming, cohort",
    "$order": "merk, handelsbenaming, cohort",
}


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    path: Path
    artifact_bytes: int
    sha256: str
    group_count: int
    accepted_group_count: int
    rejected_group_count: int
    vehicle_count: int


class _Response(Protocol):
    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> object: ...

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...


def acquire_fleet(
    destination: Path,
    *,
    opener: Callable[..., _Response] = urllib.request.urlopen,
    page_size: int = _MAX_PAGE_SIZE,
) -> AcquisitionResult:
    """Export the aggregated register atomically as a sorted canonical CSV.

    Rejected groups (blank labels or cohort, unusable counts, impossible cohorts) stay
    in the artifact so that the loader reports them against the manifest.
    """

    if type(page_size) is not int or page_size <= 0 or page_size > _MAX_PAGE_SIZE:
        raise ValueError("RDW page size must be between 1 and 50000")
    merged: dict[tuple[str, str, str], int] = {}
    unusable: list[tuple[str, str, str, str]] = []
    offset = 0
    while True:
        rows = _fetch_page(offset, page_size, opener)
        if len(rows) > page_size:
            raise ValueError("RDW API returned more rows than requested")
        for row in rows:
            make, model, cohort, vehicles = (_collapse(row[field]) for field in SCHEMA)
            if vehicles.isascii() and vehicles.isdigit():
                key = (make, model, cohort)
                merged[key] = merged.get(key, 0) + int(vehicles)
            else:
                unusable.append((make, model, cohort, vehicles))
        offset += len(rows)
        if len(rows) < page_size:
            break
    records = [(*key, str(count)) for key, count in merged.items()] + unusable
    if not records:
        raise ValueError("RDW API returned no rows")
    records.sort(key=_sort_key)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    accepted = rejected = vehicle_count = 0
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=destination.parent
        ) as output:
            temporary = Path(output.name)
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(SCHEMA)
            for record in records:
                writer.writerow(record)
                group = classify_group(
                    dict(zip(SCHEMA, record, strict=True)),
                    latest_cohort_year=_LATEST_COHORT_YEAR,
                )
                if group is None:
                    rejected += 1
                else:
                    accepted += 1
                    vehicle_count += group.vehicles
            output.flush()
            os.fsync(output.fileno())
        artifact_bytes = temporary.stat().st_size
        sha256 = _sha256(temporary)
        os.replace(temporary, destination)
        temporary = None
        return AcquisitionResult(
            destination, artifact_bytes, sha256, len(records), accepted, rejected, vehicle_count
        )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _fetch_page(
    offset: int, page_size: int, opener: Callable[..., _Response]
) -> list[dict[str, str]]:
    query = {**_QUERY, "$limit": page_size, "$offset": offset}
    url = f"{API_URL}?{urlencode(query)}"
    _require_allowlisted(url)
    request = urllib.request.Request(url, headers={"User-Agent": "ICOR-evidence/1"})
    with opener(request, timeout=120) as response:
        _require_allowlisted(response.geturl())
        payload = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise ValueError("RDW API response exceeds the byte limit")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("RDW API response is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != SCHEMA:
        raise ValueError("RDW API response schema is unsupported")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("RDW API response row is malformed")
    return rows


def _require_allowlisted(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != _ALLOWED_HOST:
        raise ValueError("RDW API URL is outside its allowlisted origin")


def _collapse(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _sort_key(record: tuple[str, ...]) -> tuple[object, ...]:
    make, model, cohort, vehicles = record
    numeric = cohort.isascii() and cohort.isdigit()
    return (make, model, 0 if numeric else 1, int(cohort) if numeric else 0, cohort, vehicles)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

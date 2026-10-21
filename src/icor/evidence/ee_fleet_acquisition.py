"""Deterministic acquisition of Estonia's registered passenger cars by make, model and cohort.

Source: Transpordiamet (the Estonian Transport Administration), "Sõidukite staatused
Eestis" ("Status of vehicles in Estonia"), published on the open data portal
andmed.eesti.ee as dataset 9c5ef13d-c838-4998-acdd-e73a5da1721b. It is a monthly
snapshot of the vehicle register, one CSV per snapshot date for categories M1 and M1G
(passenger cars) and one for all other categories.

Licence: CC BY-SA 4.0. The portal states it on the older distributions; the newest two
carry an empty licence field, so the dataset's licence is taken to apply. Share-alike
applies to derived datasets: an aggregate written here is an adaptation of the register
and is shared under the same licence, with attribution to Transpordiamet.

The ~190 MB file (UTF-8 with a byte-order mark, semicolon-separated, 46 columns) is
streamed through a hashing reader straight into the aggregator, so no raw copy lands on
disk. Only make (MARK), model (MUDEL), the year of first registration anywhere
(ESMANE_REG) and a count leave it. Two more columns are read to filter and profile:
the register status (YLDINE_STAATUS, STAATUS) and the date of first registration in
Estonia (EESTIS_ESMAREG). Every other column, including the responsible user's county
and type (VKOM_MAAKOND, VKOM_TYYP), is dropped unread and never written.

Two aggregates come from the one pass. The strict one keeps vehicles whose general status
is exactly REGISTREERITUD (actively registered). The variant adds PEATATUD (the register
entry is suspended, not deleted); the portal gives no count to say which one Eurostat's
national stock follows, so both are written and the profile compares them.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import ssl
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.parse import parse_qs, urlsplit

import certifi

from icor.evidence.normalization import normalize_vehicle_label
from icor.evidence.register_aggregate import (
    HashingReader,
    cohort_of,
    collapse,
    write_aggregate,
)

DATASET_ID = "9c5ef13d-c838-4998-acdd-e73a5da1721b"
METADATA_URL = f"https://andmed.eesti.ee/api/datasets/{DATASET_ID}"
#: The 2026-10-01 M1+M1G snapshot, the newest distribution on 2026-10-21. Transpordiamet
#: adds one file a month to this share, so ``--latest`` resolves a newer one.
SOURCE_URL = (
    "https://pilv.transpordiamet.ee/s/Iiee4OAYFq4lT1v/download"
    "?path=%2F&files=ar_2026-10-01_M1%2BM1G.csv"
)
LICENCE = "CC BY-SA 4.0"
_DATA_HOST = "pilv.transpordiamet.ee"
_METADATA_HOST = "andmed.eesti.ee"
#: The snapshot is ~192 MB; the ceiling leaves room for growth without allowing a runaway.
MAX_SOURCE_BYTES = 400 * 1024 * 1024
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_READ_BUFFER = 1024 * 1024

#: The 46 columns of the 2026-10-01 file, in order. Any other header is refused, so a
#: column added later (which might identify a person) is never silently skipped.
SOURCE_HEADER = [
    "MARK",
    "MUDEL",
    "YLDINE_STAATUS",
    "STAATUS",
    "ESMANE_REG",
    "EESTIS_ESMAREG",
    "KATEGOORIA",
    "ALAMKATEGOORIA",
    "KERE_NIMETUS",
    "KERETYYP",
    "VARV",
    "PIKKUS",
    "LAIUS",
    "KORGUS",
    "TAISMASS",
    "REG_MASS",
    "TYHIMASS",
    "HAAGIS_PIDURITEGA",
    "HAAGIS_PIDURITETA",
    "AUTORONGI_MASS",
    "HAAKESEADME_KOORMUS",
    "SUURIM_KIIRUS",
    "HEITMENORM",
    "SOIDUMYRA",
    "MOOTORI_MAHT",
    "MOOTORI_VOIMSUS",
    "MOOTORI_TYYP",
    "KAIGUKASTI_TYYP",
    "HYBRIIDI_TYYP",
    "KYTUSEKOMBINATSIOON",
    "KYTUSE_TYYP",
    "LISAKYTUS",
    "CO2_NEDC",
    "CO2_WLTP",
    "KYTUSEKULU_KESK_NEDC",
    "KYTUSEKULU_KESK_WLTP",
    "ELEKTRILINE_SOIDUULATUS",
    "UKSI",
    "ISTEKOHTI",
    "TYYP",
    "TELGI_KOKKU",
    "RATTAID",
    "JUHTTELGI",
    "VEOTELGI",
    "VKOM_MAAKOND",
    "VKOM_TYYP",
]
PASSENGER_CATEGORIES = frozenset({"M1", "M1G"})
REGISTERED = "REGISTREERITUD"
SUSPENDED = "PEATATUD"
#: The profile breaks cohorts down from this year on.
PROFILE_FROM_COHORT = 2000
TOP_MAKES = 40
_BLANK = "(blank)"
_ISO_YEAR = re.compile(r"(\d{4})-\d{2}-\d{2}")
_SNAPSHOT_FILE = re.compile(r"ar_(\d{4}-\d{2}-\d{2})_M1\+M1G\.csv\Z")
_DISTRIBUTION_TITLE = re.compile(r"\s*(\d{4}-\d{2}-\d{2})\s+M1\+M1G\s*\Z")
#: The only columns a row is read through.
_USED = {
    name: SOURCE_HEADER.index(name)
    for name in (
        "MARK",
        "MUDEL",
        "YLDINE_STAATUS",
        "STAATUS",
        "ESMANE_REG",
        "EESTIS_ESMAREG",
        "KATEGOORIA",
    )
}


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """The strict aggregate, with the variant that also counts suspended entries."""

    path: Path
    artifact_bytes: int
    sha256: str
    source_bytes: int
    source_sha256: str
    source_rows: int
    category_rows: int
    status_rows: int
    group_count: int
    vehicles: int
    masked_vehicles: int
    blank_model_vehicles: int
    blank_cohort_vehicles: int
    variant_path: Path
    variant_artifact_bytes: int
    variant_sha256: str
    variant_status_rows: int
    variant_group_count: int
    variant_vehicles: int
    profile_path: Path


@dataclass(frozen=True, slots=True)
class Distribution:
    url: str
    snapshot_date: str


class _Response(Protocol):
    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> object: ...

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...


Opener = Callable[..., _Response]


class _SameOriginRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect that leaves the host the request started on."""

    def __init__(self, host: str) -> None:
        self._host = host

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_allowlisted(newurl, self._host)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(request: urllib.request.Request, timeout: float) -> _Response:
    host = urlsplit(request.full_url).netloc
    # Python from uv ships no system CA bundle; the certificate is still verified, against certifi.
    context = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(
        _SameOriginRedirects(host), urllib.request.HTTPSHandler(context=context)
    )
    return opener.open(request, timeout=timeout)


def snapshot_date_of(url: str) -> str:
    """The snapshot date (``YYYY-MM-DD``) in the file name the share URL asks for."""

    names = parse_qs(urlsplit(url).query).get("files", [])
    match = _SNAPSHOT_FILE.match(names[0]) if len(names) == 1 else None
    if match is None:
        raise ValueError("Estonian register URL does not name an M1+M1G snapshot file")
    return match.group(1)


def latest_distribution(opener: Opener = _open) -> Distribution:
    """The newest M1+M1G distribution the portal lists, by the date in its title."""

    _require_allowlisted(METADATA_URL, _METADATA_HOST)
    request = urllib.request.Request(METADATA_URL, headers={"User-Agent": "ICOR-evidence/1"})
    with opener(request, timeout=120) as response:
        _require_allowlisted(response.geturl(), _METADATA_HOST)
        payload = response.read(_MAX_METADATA_BYTES + 1)
    if len(payload) > _MAX_METADATA_BYTES:
        raise ValueError("Estonian portal metadata exceeds the byte limit")
    try:
        document = json.loads(payload.decode("utf-8"))
        entries = document["distributions"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Estonian portal metadata is unsupported") from error
    candidates: list[Distribution] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        match = _DISTRIBUTION_TITLE.match(entry.get("titleEn") or entry.get("titleEt") or "")
        urls = entry.get("accessUrls")
        if match and isinstance(urls, list) and len(urls) == 1 and isinstance(urls[0], str):
            candidates.append(Distribution(urls[0], match.group(1)))
    if not candidates:
        raise ValueError("Estonian portal lists no M1+M1G distribution")
    newest = max(candidates, key=lambda item: item.snapshot_date)
    _require_allowlisted(newest.url, _DATA_HOST)
    if snapshot_date_of(newest.url) != newest.snapshot_date:
        raise ValueError("Estonian portal distribution title and file name disagree")
    return newest


def acquire_fleet(
    destination: Path,
    *,
    variant_destination: Path | None = None,
    url: str = SOURCE_URL,
    opener: Opener = _open,
    max_bytes: int = MAX_SOURCE_BYTES,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AcquisitionResult:
    """Stream the register into the strict aggregate, the variant and a counts-only profile.

    Masked, blank-model and blank-cohort groups stay in the aggregates so that the
    loader reports them. Nothing is written when the source fails validation midway.
    """

    _require_allowlisted(url, _DATA_HOST)
    snapshot_date = snapshot_date_of(url)
    if variant_destination is None:
        variant_destination = destination.with_name(
            f"{destination.stem}-with-suspended{destination.suffix}"
        )
    request = urllib.request.Request(url, headers={"User-Agent": "ICOR-evidence/1"})
    with opener(request, timeout=120) as response:
        _require_allowlisted(response.geturl(), _DATA_HOST)
        headers = getattr(response, "headers", None)
        last_modified = headers.get("Last-Modified") if headers is not None else None
        declared = headers.get("Content-Length") if headers is not None else None
        reader = HashingReader(response, limit=max_bytes)
        tally = _aggregate(reader)
        if declared is not None and declared.isdigit() and int(declared) != reader.bytes_read:
            raise ValueError("the Estonian register download is truncated")
    retrieved_at = clock().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    strict = write_aggregate(tally.strict, destination)
    variant = write_aggregate(tally.variant, variant_destination)
    strict_quality = _quality(tally.strict)
    variant_quality = _quality(tally.variant)
    profile_path = destination.with_name(destination.name + ".profile.json")
    _write_json(
        profile_path,
        _profile(
            tally,
            url=url,
            snapshot_date=snapshot_date,
            last_modified=last_modified,
            retrieved_at=retrieved_at,
            source_bytes=reader.bytes_read,
            source_sha256=reader.sha256,
            strict=(strict.path.name, strict.artifact_bytes, strict.sha256),
            variant=(variant.path.name, variant.artifact_bytes, variant.sha256),
            strict_quality=strict_quality,
            variant_quality=variant_quality,
        ),
    )
    return AcquisitionResult(
        path=strict.path,
        artifact_bytes=strict.artifact_bytes,
        sha256=strict.sha256,
        source_bytes=reader.bytes_read,
        source_sha256=reader.sha256,
        source_rows=tally.source_rows,
        category_rows=tally.category_rows,
        status_rows=tally.registered_rows,
        group_count=strict.group_count,
        vehicles=strict.vehicles,
        masked_vehicles=strict_quality["masked_vehicles"],
        blank_model_vehicles=strict_quality["blank_model_vehicles"],
        blank_cohort_vehicles=strict_quality["blank_cohort_vehicles"],
        variant_path=variant.path,
        variant_artifact_bytes=variant.artifact_bytes,
        variant_sha256=variant.sha256,
        variant_status_rows=tally.variant_rows,
        variant_group_count=variant.group_count,
        variant_vehicles=variant.vehicles,
        profile_path=profile_path,
    )


@dataclass(slots=True)
class _Tally:
    source_rows: int = 0
    category_rows: int = 0
    registered_rows: int = 0
    variant_rows: int = 0
    strict: dict[tuple[str, str, str], int] = field(default_factory=dict)
    variant: dict[tuple[str, str, str], int] = field(default_factory=dict)
    # Within the category filter.
    general_status: Counter[str] = field(default_factory=Counter)
    status: Counter[str] = field(default_factory=Counter)
    general_and_status: Counter[str] = field(default_factory=Counter)
    cohort_by_status: dict[str, Counter[str]] = field(default_factory=dict)
    # Within the strict filter.
    status_in_registered: Counter[str] = field(default_factory=Counter)
    cohort: Counter[str] = field(default_factory=Counter)
    national_year: Counter[str] = field(default_factory=Counter)
    cohort_vs_national: dict[str, Counter[str]] = field(default_factory=dict)
    unparseable_first_registration: int = 0


def _aggregate(source: HashingReader) -> _Tally:
    text = io.TextIOWrapper(
        io.BufferedReader(_RawAdapter(source), buffer_size=_READ_BUFFER),
        encoding="utf-8-sig",
        newline="",
    )
    tally = _Tally()
    iterator = csv.reader(text, delimiter=";")
    try:
        header = next(iterator, None)
        if header != SOURCE_HEADER:
            raise ValueError("Estonian register schema is unsupported")
        width = len(SOURCE_HEADER)
        make_at, model_at = _USED["MARK"], _USED["MUDEL"]
        general_at, status_at = _USED["YLDINE_STAATUS"], _USED["STAATUS"]
        first_at, national_at = _USED["ESMANE_REG"], _USED["EESTIS_ESMAREG"]
        category_at = _USED["KATEGOORIA"]
        for row in iterator:
            if len(row) != width:
                raise ValueError("Estonian register row is malformed")
            tally.source_rows += 1
            if row[category_at].strip() not in PASSENGER_CATEGORIES:
                continue
            tally.category_rows += 1
            general = row[general_at].strip()
            status = row[status_at].strip()
            first = row[first_at].strip()
            match = _ISO_YEAR.match(first)
            cohort = cohort_of(match.group(1) if match else "")
            if first and not cohort:
                tally.unparseable_first_registration += 1
            tally.general_status[general] += 1
            tally.status[status] += 1
            tally.general_and_status[f"{general} / {status}"] += 1
            if cohort and int(cohort) >= PROFILE_FROM_COHORT:
                tally.cohort_by_status.setdefault(cohort, Counter())[f"{general} / {status}"] += 1
            if general not in (REGISTERED, SUSPENDED):
                continue
            key = (collapse(row[make_at]), collapse(row[model_at]), cohort)
            tally.variant_rows += 1
            tally.variant[key] = tally.variant.get(key, 0) + 1
            if general != REGISTERED:
                continue
            tally.registered_rows += 1
            tally.strict[key] = tally.strict.get(key, 0) + 1
            tally.status_in_registered[status] += 1
            tally.cohort[cohort or _BLANK] += 1
            if cohort and int(cohort) >= PROFILE_FROM_COHORT:
                national = _ISO_YEAR.match(row[national_at].strip())
                year = national.group(1) if national else ""
                tally.national_year[year or _BLANK] += 1
                tally.cohort_vs_national.setdefault(cohort, Counter())[
                    _gap(int(cohort), int(year) if year else None)
                ] += 1
    except UnicodeDecodeError as error:
        raise ValueError("Estonian register is not UTF-8") from error
    except csv.Error as error:
        raise ValueError("Estonian register row is malformed") from error
    return tally


def _gap(cohort: int, national: int | None) -> str:
    """Where the Estonian first registration falls against the first one anywhere."""

    if national is None:
        return "national_blank"
    if national == cohort:
        return "same_year"
    return "later_year" if national > cohort else "earlier_year"


class _RawAdapter(io.RawIOBase):
    """Let the standard buffered text reader pull from a ``HashingReader``."""

    def __init__(self, source: HashingReader) -> None:
        self._source = source

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        chunk = self._source.read(len(buffer))
        buffer[: len(chunk)] = chunk
        return len(chunk)


def _quality(counts: Mapping[tuple[str, str, str], int]) -> dict[str, int]:
    masked = blank_make = blank_model = blank_cohort = 0
    for (make, model, cohort), vehicles in counts.items():
        # A placeholder such as "-" or "n/a" is a label the loader treats as missing.
        if (make and normalize_vehicle_label(make) is None) or (
            model and normalize_vehicle_label(model) is None
        ):
            masked += vehicles
        blank_make += vehicles if not make else 0
        blank_model += vehicles if not model else 0
        blank_cohort += vehicles if not cohort else 0
    return {
        "masked_vehicles": masked,
        "blank_make_vehicles": blank_make,
        "blank_model_vehicles": blank_model,
        "blank_cohort_vehicles": blank_cohort,
    }


def _profile(
    tally: _Tally,
    *,
    url: str,
    snapshot_date: str,
    last_modified: str | None,
    retrieved_at: str,
    source_bytes: int,
    source_sha256: str,
    strict: tuple[str, int, str],
    variant: tuple[str, int, str],
    strict_quality: dict[str, int],
    variant_quality: dict[str, int],
) -> dict[str, object]:
    makes: Counter[str] = Counter()
    for (make, _model, _cohort), vehicles in tally.strict.items():
        makes[make] += vehicles
    top_makes = sorted(makes.items(), key=lambda item: (-item[1], item[0]))[:TOP_MAKES]
    return {
        "source": {
            "dataset_id": DATASET_ID,
            "licence": LICENCE,
            "url": url,
            "snapshot_date": snapshot_date,
            "last_modified": last_modified,
            "retrieved_at": retrieved_at,
            "bytes": source_bytes,
            "sha256": source_sha256,
        },
        "filters": {
            "category": sorted(PASSENGER_CATEGORIES),
            "strict_general_status": REGISTERED,
            "variant_general_status": [REGISTERED, SUSPENDED],
            "cohort": "year of ESMANE_REG, the first registration in any country",
        },
        "funnel": {
            "source_rows": tally.source_rows,
            "category_rows": tally.category_rows,
            "registered_rows": tally.registered_rows,
            "registered_or_suspended_rows": tally.variant_rows,
        },
        "aggregate": _file(strict, tally.strict, strict_quality),
        "aggregate_with_suspended": _file(variant, tally.variant, variant_quality),
        "vehicles_by_general_status": _sorted(tally.general_status),
        "vehicles_by_status": _sorted(tally.status),
        "vehicles_by_general_and_status": _sorted(tally.general_and_status),
        "vehicles_by_status_within_registered": _sorted(tally.status_in_registered),
        "registered_vehicles_by_cohort": _by_year(tally.cohort),
        "vehicles_by_cohort_and_status_2000_plus": {
            cohort: _sorted(counts) for cohort, counts in sorted(tally.cohort_by_status.items())
        },
        "registered_vehicles_by_national_first_registration_year_cohorts_2000_plus": _by_year(
            tally.national_year
        ),
        "registered_cohorts_2000_plus_by_national_registration_vs_cohort": {
            cohort: _sorted(counts) for cohort, counts in sorted(tally.cohort_vs_national.items())
        },
        "unparseable_first_registration_vehicles": tally.unparseable_first_registration,
        "top_makes": [[make, vehicles] for make, vehicles in top_makes],
    }


def _file(
    identity: tuple[str, int, str],
    counts: Mapping[tuple[str, str, str], int],
    quality: dict[str, int],
) -> dict[str, object]:
    name, size, digest = identity
    return {
        "file": name,
        "bytes": size,
        "sha256": digest,
        "groups": len(counts),
        "vehicles": sum(counts.values()),
        **quality,
    }


def _sorted(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _by_year(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (item[0] == _BLANK, item[0])))


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=path.parent
        ) as output:
            temporary = Path(output.name)
            json.dump(document, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _require_allowlisted(url: str, host: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != host:
        raise ValueError("Estonian register URL is outside its allowlisted origin")

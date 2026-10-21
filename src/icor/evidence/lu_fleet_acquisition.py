"""Streaming acquisition of Luxembourg's registered passenger cars by make, model and cohort.

Source: "Parc automobile du Luxembourg", the monthly export of the SNCA (Société nationale
de circulation automobile) published on data.public.lu under licence CC0 1.0 (``cc-zero``,
public-domain dedication; no attribution is required, the source is still named in every
profile). The dataset description says each export holds all registered vehicles currently
running in Luxembourg. Field meanings follow the publisher's data dictionary
"Description des données exportées vers Opendata" v1.9.1.

The newest per-vehicle XML resource is found through the dataset's metadata API, then
streamed through an incremental parser and a ``HashingReader`` straight into counts: no raw
copy is stored. Only make (``LIBMRQ``), model (``TYPCOM``), the year of first registration
anywhere (``DATCIRPRM``) and a count reach the aggregate. The profile adds counts by status,
category, national first-registration year and origin country. ``INFOUTI`` (user information),
``INDUTI`` (use indicator) and every other field are never read.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.parse import urlsplit

from icor.evidence.register_aggregate import (
    HashingReader,
    cohort_of,
    collapse,
    write_aggregate,
)

DATASET_URL = "https://data.public.lu/api/1/datasets/parc-automobile-du-luxembourg/"
LICENCE = "cc-zero"
_API_HOST = "data.public.lu"
_DOWNLOAD_HOST = "download.data.public.lu"
_MAX_API_BYTES = 8 * 1024 * 1024
#: The October 2026 export is 890 MB and grows about 0.4 MB a month; this is a ceiling.
MAX_SOURCE_BYTES = 1_500_000_000
_RESOURCE_URL = re.compile(
    r"https://download\.data\.public\.lu/resources/parc-automobile-du-luxembourg/"
    r"(\d{8})-(\d{6})/parc-automobile-(\d{6})\.xml\Z"
)

#: ``CATEU``: the European categories of passenger cars.
PASSENGER_CATEGORIES = frozenset({"M1", "M1G"})
#: ``CATSTC`` 1 is the national category "Voiture"; it is profiled, never filtered on.
NATIONAL_CAR_CATEGORY = "1"
#: The only fields read from a vehicle. Everything else in the record stays unread.
_FIELDS = frozenset(
    {
        "LIBMRQ",
        "TYPCOM",
        "CATEU",
        "CATSTC",
        "DATCIRPRM",
        "DATCIR_GD",
        "DATHORCIR",
        "PAYPVN",
    }
)
_DATE = re.compile(r"\d{8}\Z", re.ASCII)
ON_ROAD = "on_road"
OFF_ROAD = "off_road"
_PROFILE_COHORT_FROM = 2000
_USED_IMPORT_COHORT_FROM = 2006
_TOP_MAKES = 40
#: A stream whose dates are mostly not YYYYMMDD has changed format; a few stray values do not.
_MAX_MALFORMED_DATE_SHARE = 0.01


@dataclass(frozen=True, slots=True)
class Resource:
    url: str
    period: str
    snapshot_date: str
    declared_bytes: int | None = None
    last_modified: str = ""


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    path: Path
    artifact_bytes: int
    sha256: str
    profile_path: Path
    source_url: str
    snapshot_date: str
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


class _Response(Protocol):
    headers: Mapping[str, str]

    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> object: ...

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...


class _AllowlistedRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect that leaves the two allowlisted hosts before following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlsplit(newurl)
        if parts.scheme != "https" or parts.netloc not in {_API_HOST, _DOWNLOAD_HOST}:
            raise ValueError("Luxembourg register redirect leaves its allowlisted origins")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
    opener = urllib.request.build_opener(_AllowlistedRedirects)
    return opener.open(request, timeout=timeout)


def latest_resource(*, opener: Callable[..., _Response] = urlopen) -> Resource:
    """The newest monthly per-vehicle XML resource listed by the dataset's metadata API."""

    _require_allowlisted(DATASET_URL, _API_HOST)
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "ICOR-evidence/1"})
    with opener(request, timeout=60) as response:
        _require_allowlisted(response.geturl(), _API_HOST)
        payload = response.read(_MAX_API_BYTES + 1)
    if len(payload) > _MAX_API_BYTES:
        raise ValueError("Luxembourg dataset metadata exceeds the byte limit")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Luxembourg dataset metadata is not JSON") from error
    if not isinstance(document, dict) or not isinstance(document.get("resources"), list):
        raise ValueError("Luxembourg dataset metadata has no resource list")
    if document.get("license") != LICENCE:
        raise ValueError("Luxembourg dataset is no longer published under licence CC0")
    candidates: list[tuple[tuple[str, str, str], Resource]] = []
    for entry in document["resources"]:
        if not isinstance(entry, dict) or entry.get("format") != "xml":
            continue
        match = _RESOURCE_URL.match(str(entry.get("url")))
        if match is None:
            continue
        size = entry.get("filesize")
        modified = entry.get("last_modified")
        candidates.append(
            (
                (match.group(3), match.group(1), match.group(2)),
                Resource(
                    str(entry["url"]),
                    match.group(3),
                    match.group(1),
                    size if type(size) is int and size > 0 else None,
                    modified if isinstance(modified, str) else "",
                ),
            )
        )
    if not candidates:
        raise ValueError("Luxembourg dataset lists no monthly vehicle XML resource")
    return max(candidates, key=lambda item: item[0])[1]


def pinned_resource(url: str) -> Resource:
    """A resource named by its download URL, which must be an allowlisted XML export."""

    _require_allowlisted(url, _DOWNLOAD_HOST)
    match = _RESOURCE_URL.match(url)
    if match is None:
        raise ValueError("Luxembourg register URL is not a monthly vehicle XML export")
    return Resource(url, match.group(3), match.group(1))


def acquire_fleet(
    destination: Path,
    *,
    url: str | None = None,
    opener: Callable[..., _Response] = urlopen,
    max_bytes: int = MAX_SOURCE_BYTES,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AcquisitionResult:
    """Stream the register into a sorted canonical CSV of counts and a counts-only profile.

    ``url`` pins one export; without it the newest one is used. Rows with a blank or
    placeholder make or model, or without a usable first-registration year, stay in the
    aggregate so that the loader reports them against the manifest.
    """

    resource = pinned_resource(url) if url is not None else latest_resource(opener=opener)
    _require_allowlisted(resource.url, _DOWNLOAD_HOST)
    retrieved = now().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    request = urllib.request.Request(resource.url, headers={"User-Agent": "ICOR-evidence/1"})
    tally = _Tally()
    with opener(request, timeout=120) as response:
        _require_allowlisted(response.geturl(), _DOWNLOAD_HOST)
        reader = HashingReader(response, limit=max_bytes)
        for vehicle in _vehicles(reader):
            tally.add(vehicle)
        while reader.read(1024 * 1024):
            pass
        declared = _content_length(response.headers.get("Content-Length"))
        last_modified = response.headers.get("Last-Modified") or ""
    for expected in (declared, resource.declared_bytes):
        if expected is not None and expected != reader.bytes_read:
            raise ValueError("Luxembourg register download is incomplete or changed size")
    tally.require_known_date_format()

    aggregate = write_aggregate(tally.counts, destination)
    masked, blank_model, blank_cohort = tally.unusable_vehicles()
    profile_path = destination.with_name(destination.name + ".profile.json")
    _write_json(
        profile_path,
        tally.profile(
            {
                "dataset": DATASET_URL,
                "licence": LICENCE,
                "url": resource.url,
                "snapshot_date": resource.snapshot_date,
                "period": resource.period,
                "http_last_modified": last_modified,
                "resource_last_modified": resource.last_modified,
                "retrieved_at_utc": retrieved,
                "bytes": reader.bytes_read,
                "sha256": reader.sha256,
            },
            {
                "path": destination.name,
                "bytes": aggregate.artifact_bytes,
                "sha256": aggregate.sha256,
                "groups": aggregate.group_count,
                "vehicles": aggregate.vehicles,
            },
        ),
    )
    return AcquisitionResult(
        destination,
        aggregate.artifact_bytes,
        aggregate.sha256,
        profile_path,
        resource.url,
        resource.snapshot_date,
        reader.bytes_read,
        reader.sha256,
        tally.source_rows,
        tally.category_rows,
        tally.status_rows,
        aggregate.group_count,
        aggregate.vehicles,
        masked,
        blank_model,
        blank_cohort,
    )


def _vehicles(stream: HashingReader) -> Iterator[dict[str, str]]:
    """Yield the read fields of each ``<VEHICLE>``, clearing it before the next one."""

    root: ET.Element | None = None
    depth = 0
    try:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            if event == "start":
                depth += 1
                if depth == 1:
                    if element.tag != "VEHICLES":
                        raise ValueError("Luxembourg register root element is unsupported")
                    root = element
                elif depth == 2 and element.tag != "VEHICLE":
                    raise ValueError("Luxembourg register holds an element other than VEHICLE")
                continue
            depth -= 1
            if depth != 1:
                continue
            fields: dict[str, str] = {}
            for child in element:
                if child.tag in _FIELDS:
                    if child.tag in fields:
                        raise ValueError("Luxembourg register vehicle repeats a field")
                    fields[child.tag] = child.text or ""
            if fields.keys() != _FIELDS:
                raise ValueError("Luxembourg register schema is unsupported")
            element.clear()
            if root is not None:
                root.clear()
            yield fields
    except ET.ParseError as error:
        raise ValueError("Luxembourg register is not well-formed XML") from error


class _Tally:
    """Counts only: nothing here keeps a row, an identifier or an unread field."""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str, str], int] = {}
        self.source_rows = self.category_rows = self.status_rows = 0
        self.malformed_dates = 0
        self.status: Counter[str] = Counter()
        self.catstc: Counter[str] = Counter()
        self.cars_outside_category: Counter[str] = Counter()
        self.cohorts: Counter[str] = Counter()
        self.cohort_status: dict[str, Counter[str]] = {}
        self.national_by_cohort: dict[str, Counter[str]] = {}
        self.used_imports: Counter[tuple[str, str]] = Counter()
        self.off_road_years: Counter[str] = Counter()
        self.makes: Counter[str] = Counter()

    def add(self, vehicle: dict[str, str]) -> None:
        self.source_rows += 1
        category = vehicle["CATEU"].strip()
        national_category = vehicle["CATSTC"].strip()
        if category not in PASSENGER_CATEGORIES:
            if national_category == NATIONAL_CAR_CATEGORY:
                self.cars_outside_category[category] += 1
            return
        self.category_rows += 1
        self.catstc[national_category] += 1
        first = self._date(vehicle["DATCIRPRM"])
        cohort = cohort_of(first[:4]) if first else ""
        off_road = vehicle["DATHORCIR"].strip()
        status = OFF_ROAD if off_road else ON_ROAD
        self.status[status] += 1
        if _from(cohort, _PROFILE_COHORT_FROM):
            self.cohort_status.setdefault(cohort, Counter())[status] += 1
        if off_road:
            left = self._date(off_road)
            self.off_road_years[left[:4] if left else "unparseable"] += 1
            return
        self.status_rows += 1
        make = collapse(vehicle["LIBMRQ"])
        key = (make, collapse(vehicle["TYPCOM"]), cohort)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.makes[make] += 1
        self.cohorts[cohort] += 1
        national = self._date(vehicle["DATCIR_GD"])
        if _from(cohort, _PROFILE_COHORT_FROM):
            year = national[:4] if national else ""
            self.national_by_cohort.setdefault(cohort, Counter())[year] += 1
        if (
            _from(cohort, _USED_IMPORT_COHORT_FROM)
            and first
            and national
            and first < national  # YYYYMMDD sorts chronologically
        ):
            self.used_imports[(cohort, collapse(vehicle["PAYPVN"]))] += 1

    def _date(self, value: str) -> str:
        """The value as YYYYMMDD, "" when blank, and "" (counted) when malformed."""

        text = value.strip()
        if not text:
            return ""
        if _DATE.match(text) is None:
            self.malformed_dates += 1
            return ""
        return text

    def require_known_date_format(self) -> None:
        if self.malformed_dates > _MAX_MALFORMED_DATE_SHARE * max(self.category_rows, 1):
            raise ValueError("Luxembourg register date format is unsupported")

    def unusable_vehicles(self) -> tuple[int, int, int]:
        masked = blank_model = blank_cohort = 0
        for (make, model, cohort), count in self.counts.items():
            if _masked(make) or _masked(model):
                masked += count
            if not model:
                blank_model += count
            if not cohort:
                blank_cohort += count
        return masked, blank_model, blank_cohort

    def profile(
        self, source: Mapping[str, object], aggregate: Mapping[str, object]
    ) -> dict[str, object]:
        masked, blank_model, blank_cohort = self.unusable_vehicles()
        used_by_origin: Counter[str] = Counter()
        used_by_cohort: Counter[str] = Counter()
        for (cohort, origin), count in self.used_imports.items():
            used_by_origin[origin] += count
            used_by_cohort[cohort] += count
        national_totals: Counter[str] = Counter()
        for years in self.national_by_cohort.values():
            national_totals.update(years)
        return {
            "source": dict(source),
            "aggregate": dict(aggregate),
            "filters": {
                "category": f"CATEU in {sorted(PASSENGER_CATEGORIES)}",
                "status": "DATHORCIR empty (a filled off-road date is off the road)",
                "cohort": "year of DATCIRPRM, first registration anywhere (not DATCIR_GD)",
            },
            "funnel": {
                "source_rows": self.source_rows,
                "after_category": self.category_rows,
                "after_status": self.status_rows,
            },
            "vehicles_by_status_within_category": _ordered(self.status),
            "vehicles_by_cohort_strict": _ordered(self.cohorts),
            "vehicles_by_cohort_and_status_from_2000": {
                cohort: _ordered(self.cohort_status[cohort])
                for cohort in _years(self.cohort_status)
            },
            "vehicles_by_national_year_for_cohorts_from_2000": _ordered(national_totals),
            "national_year_by_cohort_from_2000": {
                cohort: _ordered(self.national_by_cohort[cohort])
                for cohort in _years(self.national_by_cohort)
            },
            "used_imports_from_2006": {
                "definition": "DATCIRPRM before DATCIR_GD, strict filter, cohorts from 2006",
                "vehicles": sum(used_by_origin.values()),
                "by_origin_country": _ordered(used_by_origin),
                "by_cohort": _ordered(used_by_cohort),
            },
            "unusable_vehicles": {
                "blank_make": sum(c for (make, _, _), c in self.counts.items() if not make),
                "masked": masked,
                "blank_model": blank_model,
                "blank_cohort": blank_cohort,
            },
            "malformed_dates": self.malformed_dates,
            "national_category_within_category": _ordered(self.catstc),
            "national_cars_outside_category_by_cateu": _ordered(self.cars_outside_category),
            "off_road_by_year": _ordered(self.off_road_years),
            "top_makes": {
                "basis": "strict filter, by vehicles",
                "makes": [
                    [make, count]
                    for make, count in sorted(
                        self.makes.items(), key=lambda item: (-item[1], item[0])
                    )[:_TOP_MAKES]
                ],
            },
        }


def _from(cohort: str, first_year: int) -> bool:
    return bool(cohort) and int(cohort) >= first_year


def _masked(label: str) -> bool:
    """A non-blank label without a single letter or digit, such as "*" or "?"."""

    return bool(label) and not any(character.isalnum() for character in label)


def _years(groups: Iterable[str]) -> list[str]:
    return sorted(groups, key=lambda year: (not year.isdigit(), year))


def _ordered(counter: Mapping[str, int]) -> dict[str, int]:
    return {key: counter[key] for key in _years(counter)}


def _content_length(value: str | None) -> int | None:
    return int(value) if value is not None and value.isascii() and value.isdigit() else None


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
        ) as output:
            temporary = Path(output.name)
            json.dump(document, output, ensure_ascii=False, sort_keys=True, indent=2)
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
        raise ValueError("Luxembourg register URL is outside its allowlisted origin")


__all__ = [
    "DATASET_URL",
    "LICENCE",
    "MAX_SOURCE_BYTES",
    "AcquisitionResult",
    "Resource",
    "acquire_fleet",
    "latest_resource",
    "pinned_resource",
    "urlopen",
]

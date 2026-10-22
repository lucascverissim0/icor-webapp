"""Streamed acquisition of the Czech passenger-car stock by make, model and cohort.

Source: the Czech Ministry of Transport's Registr silnicnich vozidel (vehicle register),
published as one monthly full extract by https://download.dataovozidlech.cz. The data is
open data released under a CC0-equivalent licence, so no attribution or use restriction
binds the counts derived from it.

The extract is several gigabytes and is never kept: it is streamed straight into the
counters through :class:`HashingReader`, which records the byte count and SHA-256 of the
exact bytes the server sent (the gzip stream when the server compresses it). Only make,
model, the year of the first registration anywhere and a count leave it. The VIN, engine
number, body number, type-approval and registration-certificate numbers and the vehicle
identifier (PCV) are never read into any output.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import unicodedata
import urllib.request
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.parse import urlsplit

from icor.evidence.register_aggregate import (
    AggregateFile,
    HashingReader,
    cohort_of,
    collapse,
    write_aggregate,
)

SOURCE_URL = "https://download.dataovozidlech.cz/vypiszregistru/vypisvozidel"
SCHEMA_URL = "https://download.dataovozidlech.cz/schema/vypisvozidel"
LICENCE = "CC0-equivalent open data (Ministerstvo dopravy, Registr silnicnich vozidel)"
_ALLOWED_HOST = "download.dataovozidlech.cz"
#: The extract is ~3 GB as served (gzip); this ceiling stops a runaway stream.
MAX_SOURCE_BYTES = 12 * 1024**3
_CHUNK = 1024 * 1024

COL_MAKE = "Tovární značka"
COL_MODEL = "Obchodní označení"
COL_CATEGORY = "Kategorie vozidla"
COL_STATUS = "Status"
COL_FIRST = "Datum 1. registrace"
COL_NATIONAL = "Datum 1. registrace v ČR"
COL_KIND = "Druh vozidla"
COL_PRODUCTION_YEAR = "Rok výroby"
#: The only columns this module reads; any other column is never looked at.
REQUIRED_COLUMNS = (
    COL_MAKE,
    COL_MODEL,
    COL_CATEGORY,
    COL_STATUS,
    COL_FIRST,
    COL_NATIONAL,
    COL_KIND,
    COL_PRODUCTION_YEAR,
)
CATEGORY_PREFIX = "M1"
STATUS_STRICT = "PROVOZOVANÉ"
STATUS_WITHDRAWN = "VYŘAZENO Z PROVOZU"
BLANK = "(blank)"
UNUSABLE = "(unusable)"
OTHER = "(other)"
_TOP_MAKES = 40
_MAX_KIND_VALUES = 100
_FIRST_PROFILED_COHORT = 2000
_DATE = re.compile(r"(\d{4})-\d{2}-\d{2}\Z")
_FILENAME = re.compile(r'filename="?RSV_vypis_vozidel_(\d{8})\.csv"?')
_PROGRESS_EVERY = 1_000_000


@dataclass(frozen=True, slots=True)
class VariantResult:
    """The aggregate that also counts temporarily withdrawn cars."""

    path: Path
    artifact_bytes: int
    sha256: str
    status_rows: int
    group_count: int
    vehicles: int
    masked_vehicle_count: int
    blank_model_vehicle_count: int
    blank_cohort_vehicle_count: int


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    path: Path
    artifact_bytes: int
    sha256: str
    profile_path: Path
    snapshot_date: str
    source_bytes: int
    source_sha256: str
    source_gzip: bool
    source_rows: int
    category_rows: int
    status_rows: int
    group_count: int
    vehicles: int
    masked_vehicle_count: int
    blank_model_vehicle_count: int
    blank_cohort_vehicle_count: int
    with_withdrawn: VariantResult


class _Headers(Protocol):
    def get(self, name: str, default: str | None = None) -> str | None: ...


class _Response(Protocol):
    headers: _Headers

    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> object: ...

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...


class _AllowlistedRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse to follow a redirect that leaves the allowlisted origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_allowlisted(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_opener(request: urllib.request.Request, timeout: float) -> _Response:
    return urllib.request.build_opener(_AllowlistedRedirects()).open(request, timeout=timeout)


class _Decompressed(io.RawIOBase):
    """Raw byte stream over the response that gunzips while streaming when asked to."""

    def __init__(self, source: HashingReader, *, gzipped: bool) -> None:
        self._source = source
        self._inflater = zlib.decompressobj(wbits=31) if gzipped else None
        self._pending = b""
        self._offset = 0
        self._finished = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while self._offset >= len(self._pending) and not self._finished:
            chunk = self._source.read(_CHUNK)
            self._offset = 0
            if self._inflater is None:
                self._pending = chunk
                self._finished = not chunk
            elif chunk:
                if self._inflater.eof:
                    raise ValueError("the register download has data after its gzip stream")
                self._pending = self._inflater.decompress(chunk)
                if self._inflater.unused_data:
                    raise ValueError("the register download has data after its gzip stream")
            else:
                if not self._inflater.eof:
                    raise ValueError("the register download ends inside its gzip stream")
                self._pending = b""
                self._finished = True
        count = min(len(buffer), len(self._pending) - self._offset)
        buffer[:count] = self._pending[self._offset : self._offset + count]
        self._offset += count
        return count


def acquire_fleet(
    destination: Path,
    *,
    opener: Callable[..., _Response] = _default_opener,
    url: str = SOURCE_URL,
    byte_limit: int | None = MAX_SOURCE_BYTES,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    progress: Callable[[int, int], None] | None = None,
) -> AcquisitionResult:
    """Stream the register into two counts-only aggregates and a counts-only profile.

    ``destination`` may contain ``{date}``, replaced by the snapshot date (YYYYMMDD) that
    the server names in Content-Disposition. The strict aggregate holds cars with status
    PROVOZOVANE; ``<name>-with-withdrawn.csv`` next to it adds the temporarily withdrawn
    ones. Blank and masked labels and blank cohorts stay in the aggregates, so that the
    loader reports them. Nothing is written unless the whole stream was read and verified.
    """

    _require_allowlisted(url)
    request = urllib.request.Request(
        url, headers={"Accept-Encoding": "gzip", "User-Agent": "ICOR-evidence/1"}
    )
    retrieved_at = now().astimezone(UTC)
    with opener(request, timeout=120) as response:
        _require_allowlisted(response.geturl())
        snapshot = _snapshot_date(response.headers.get("Content-Disposition"))
        last_modified = response.headers.get("Last-Modified")
        encoding = (response.headers.get("Content-Encoding") or "identity").strip().lower()
        if encoding not in ("gzip", "identity"):
            raise ValueError("the register download uses an unsupported content encoding")
        source = HashingReader(response, limit=byte_limit)
        raw = _Decompressed(source, gzipped=encoding == "gzip")
        stream = io.TextIOWrapper(
            io.BufferedReader(raw, buffer_size=_CHUNK), encoding="utf-8-sig", newline=""
        )
        tally = _read_register(csv.reader(stream), source, progress)
    destination = Path(str(destination).replace("{date}", snapshot))

    strict = write_aggregate(tally.strict, destination)
    withdrawn_path = _variant_path(destination)
    merged: Counter[tuple[str, str, str]] = Counter(tally.strict)
    merged.update(tally.withdrawn)
    variant = write_aggregate(merged, withdrawn_path)

    strict_flags = _label_flags(tally.strict)
    variant_flags = _label_flags(merged)
    profile_path = destination.with_name(destination.name + ".profile.json")
    profile = _profile(
        tally=tally,
        url=url,
        snapshot=snapshot,
        last_modified=last_modified,
        retrieved_at=retrieved_at,
        source_bytes=source.bytes_read,
        source_sha256=source.sha256,
        gzipped=encoding == "gzip",
        strict=strict,
        strict_flags=strict_flags,
        variant=variant,
        variant_flags=variant_flags,
    )
    _write_json(profile, profile_path)
    return AcquisitionResult(
        path=destination,
        artifact_bytes=strict.artifact_bytes,
        sha256=strict.sha256,
        profile_path=profile_path,
        snapshot_date=snapshot,
        source_bytes=source.bytes_read,
        source_sha256=source.sha256,
        source_gzip=encoding == "gzip",
        source_rows=tally.source_rows,
        category_rows=tally.category_rows,
        status_rows=tally.status_rows,
        group_count=strict.group_count,
        vehicles=strict.vehicles,
        masked_vehicle_count=strict_flags.masked,
        blank_model_vehicle_count=strict_flags.blank_model,
        blank_cohort_vehicle_count=strict_flags.blank_cohort,
        with_withdrawn=VariantResult(
            path=withdrawn_path,
            artifact_bytes=variant.artifact_bytes,
            sha256=variant.sha256,
            status_rows=tally.status_rows + tally.withdrawn_rows,
            group_count=variant.group_count,
            vehicles=variant.vehicles,
            masked_vehicle_count=variant_flags.masked,
            blank_model_vehicle_count=variant_flags.blank_model,
            blank_cohort_vehicle_count=variant_flags.blank_cohort,
        ),
    )


def _counter() -> Counter:
    return Counter()


@dataclass(slots=True)
class _Tally:
    """Counts only: nothing here can hold a row-level field."""

    source_rows: int = 0
    category_rows: int = 0
    status_rows: int = 0
    withdrawn_rows: int = 0
    strict: Counter[tuple[str, str, str]] = field(default_factory=_counter)
    withdrawn: Counter[tuple[str, str, str]] = field(default_factory=_counter)
    makes: Counter[str] = field(default_factory=_counter)
    category_values: Counter[str] = field(default_factory=_counter)
    status_in_category: Counter[str] = field(default_factory=_counter)
    cohort_by_status: Counter[tuple[str, str]] = field(default_factory=_counter)
    kind_by_status: Counter[tuple[str, str]] = field(default_factory=_counter)
    uncategorised_passenger: Counter[str] = field(default_factory=_counter)
    national_year: Counter[str] = field(default_factory=_counter)
    national_vs_first: Counter[tuple[str, str]] = field(default_factory=_counter)
    blank_cohort_by_status: Counter[str] = field(default_factory=_counter)
    blank_cohort_by_kind: Counter[str] = field(default_factory=_counter)
    blank_cohort_by_production_year: Counter[str] = field(default_factory=_counter)


@dataclass(frozen=True, slots=True)
class _Flags:
    masked: int
    blank_model: int
    blank_cohort: int


def _read_register(
    reader: Iterable[list[str]],
    source: HashingReader,
    progress: Callable[[int, int], None] | None,
) -> _Tally:
    rows = iter(reader)
    header = next(rows, None)
    if header is None:
        raise ValueError("the register download is empty")
    names = [unicodedata.normalize("NFC", name) for name in header]
    if len(set(names)) != len(names) or any(name not in names for name in REQUIRED_COLUMNS):
        raise ValueError("the register schema is unsupported: a required column is missing")
    position = {name: names.index(name) for name in REQUIRED_COLUMNS}
    i_make, i_model = position[COL_MAKE], position[COL_MODEL]
    i_category, i_status = position[COL_CATEGORY], position[COL_STATUS]
    i_first, i_national = position[COL_FIRST], position[COL_NATIONAL]
    i_kind, i_production = position[COL_KIND], position[COL_PRODUCTION_YEAR]
    width = len(names)

    tally = _Tally()
    labels: dict[str, str] = {}
    statuses: dict[str, str] = {}
    years: dict[str, str] = {}
    category_cache: dict[str, tuple[bool, str]] = {}

    def label(value: str) -> str:
        found = labels.get(value)
        if found is None:
            found = labels[value] = collapse(value)
        return found

    def year(value: str) -> str:
        found = years.get(value)
        if found is None:
            matched = _DATE.match(value.strip())
            found = years[value] = cohort_of(matched.group(1)) if matched else ""
        return found

    for fields in rows:
        tally.source_rows += 1
        if len(fields) != width:
            raise ValueError("the register download has a malformed row")
        if progress is not None and tally.source_rows % _PROGRESS_EVERY == 0:
            progress(tally.source_rows, source.bytes_read)
        raw_category = fields[i_category]
        cached = category_cache.get(raw_category)
        if cached is None:
            normal = raw_category.strip().upper()
            cached = category_cache[raw_category] = (normal.startswith(CATEGORY_PREFIX), normal)
        tally.category_values[cached[1] or BLANK] += 1
        if not cached[0]:
            if cached[1] in ("", ".") and fields[i_kind].strip().upper().startswith("OSOBNÍ"):
                tally.uncategorised_passenger[_status(fields[i_status], statuses) or BLANK] += 1
            continue
        tally.category_rows += 1
        status = _status(fields[i_status], statuses)
        kind = label(fields[i_kind]) or BLANK
        cohort = year(fields[i_first])
        tally.status_in_category[status or BLANK] += 1
        tally.kind_by_status[(kind, status or BLANK)] += 1
        tally.cohort_by_status[(cohort or BLANK, status or BLANK)] += 1
        if not cohort:
            tally.blank_cohort_by_status[status or BLANK] += 1
        if status == STATUS_STRICT:
            tally.status_rows += 1
            make = label(fields[i_make])
            tally.strict[(make, label(fields[i_model]), cohort)] += 1
            tally.makes[make or BLANK] += 1
            national = year(fields[i_national])
            tally.national_year[national or BLANK] += 1
            if cohort and int(cohort) >= _FIRST_PROFILED_COHORT:
                tally.national_vs_first[(cohort, _compare(national, cohort))] += 1
            if not cohort:
                tally.blank_cohort_by_kind[kind] += 1
                tally.blank_cohort_by_production_year[_production_year(fields[i_production])] += 1
        elif status == STATUS_WITHDRAWN:
            tally.withdrawn_rows += 1
            tally.withdrawn[(label(fields[i_make]), label(fields[i_model]), cohort)] += 1
    if tally.source_rows == 0:
        raise ValueError("the register download holds no rows")
    return tally


def _status(value: str, cache: dict[str, str]) -> str:
    found = cache.get(value)
    if found is None:
        found = cache[value] = collapse(value).upper()
    return found


def _compare(national: str, first: str) -> str:
    if not national:
        return "national_year_blank"
    if national == first:
        return "national_year_same"
    return "national_year_later" if national > first else "national_year_earlier"


def _production_year(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return BLANK
    return cleaned if cohort_of(cleaned) else UNUSABLE


def _label_flags(counts: Mapping[tuple[str, str, str], int]) -> _Flags:
    masked = blank_model = blank_cohort = 0
    for (make, model, cohort), vehicles in counts.items():
        if _is_masked(make) or _is_masked(model):
            masked += vehicles
        if not model:
            blank_model += vehicles
        if not cohort:
            blank_cohort += vehicles
    return _Flags(masked, blank_model, blank_cohort)


def _is_masked(label: str) -> bool:
    """A placeholder such as "." or "¡": not blank, yet without any letter or digit."""

    return bool(label) and (not any(char.isalnum() for char in label) or "¡" in label)


def _profile(
    *,
    tally: _Tally,
    url: str,
    snapshot: str,
    last_modified: str | None,
    retrieved_at: datetime,
    source_bytes: int,
    source_sha256: str,
    gzipped: bool,
    strict: AggregateFile,
    strict_flags: _Flags,
    variant: AggregateFile,
    variant_flags: _Flags,
) -> dict[str, object]:
    status_totals = _ordered(tally.status_in_category, by_count=True)
    cohort_strict: Counter[str] = Counter()
    for key, vehicles in tally.strict.items():
        cohort_strict[key[2] or BLANK] += vehicles
    recent_by_status = {
        cohort: dict(
            sorted(
                (status, count)
                for (year, status), count in tally.cohort_by_status.items()
                if year == cohort
            )
        )
        for cohort in _cohorts(year for year, _ in tally.cohort_by_status)
        if cohort.isdigit() and int(cohort) >= _FIRST_PROFILED_COHORT
    }
    national: dict[str, int] = {}
    for key, count in sorted(tally.national_year.items()):
        if key.isdigit() and int(key) < _FIRST_PROFILED_COHORT:
            national["before_2000"] = national.get("before_2000", 0) + count
        else:
            national[key] = count
    imports: dict[str, dict[str, int]] = {}
    for (cohort, outcome), count in tally.national_vs_first.items():
        imports.setdefault(cohort, {})[outcome] = count
    kinds = _kind_table(tally.kind_by_status)
    return {
        "source": {
            "url": url,
            "schema_url": SCHEMA_URL,
            "licence": LICENCE,
            "snapshot_date": f"{snapshot[:4]}-{snapshot[4:6]}-{snapshot[6:]}",
            "last_modified": last_modified,
            "retrieved_at_utc": retrieved_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bytes": source_bytes,
            "sha256": source_sha256,
            "gzip": gzipped,
        },
        "filters": {
            "category": f"Kategorie vozidla starts with {CATEGORY_PREFIX} (strip, upper-case)",
            "status_strict": STATUS_STRICT,
            "status_with_withdrawn": [STATUS_STRICT, STATUS_WITHDRAWN],
            "cohort": "year of Datum 1. registrace (first registration anywhere)",
        },
        "funnel": {
            "source_rows": tally.source_rows,
            "category_rows": tally.category_rows,
            "status_rows_strict": tally.status_rows,
            "status_rows_with_withdrawn": tally.status_rows + tally.withdrawn_rows,
        },
        "aggregates": {
            "strict": _aggregate_entry(strict, strict_flags),
            "with_withdrawn": _aggregate_entry(variant, variant_flags),
        },
        "vehicles_by_status_within_category": status_totals,
        "vehicles_by_cohort_strict": _ordered_cohorts(cohort_strict),
        "vehicles_by_cohort_and_status_2000_plus": recent_by_status,
        "vehicles_by_national_year_strict": national,
        "national_year_against_first_registration_strict_2000_plus": {
            cohort: dict(sorted(outcomes.items())) for cohort, outcomes in sorted(imports.items())
        },
        "blank_cohort": {
            "by_status_within_category": _ordered(tally.blank_cohort_by_status, by_count=True),
            "strict_by_vehicle_kind": _ordered(tally.blank_cohort_by_kind, by_count=True),
            "strict_by_production_year": _ordered(
                tally.blank_cohort_by_production_year, by_count=True
            ),
        },
        "vehicle_kind_within_category_by_status": kinds,
        "category_values_top": dict(tally.category_values.most_common(40)),
        "uncategorised_passenger_kind_by_status": _ordered(
            tally.uncategorised_passenger, by_count=True
        ),
        "top_makes_strict": [
            {"make": make, "vehicles": count}
            for make, count in sorted(tally.makes.items(), key=lambda item: (-item[1], item[0]))[
                :_TOP_MAKES
            ]
        ],
    }


def _aggregate_entry(aggregate: AggregateFile, flags: _Flags) -> dict[str, object]:
    return {
        "file": aggregate.path.name,
        "bytes": aggregate.artifact_bytes,
        "sha256": aggregate.sha256,
        "group_count": aggregate.group_count,
        "vehicles": aggregate.vehicles,
        "masked_vehicles": flags.masked,
        "blank_model_vehicles": flags.blank_model,
        "blank_cohort_vehicles": flags.blank_cohort,
    }


def _kind_table(counts: Mapping[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    totals: Counter[str] = Counter()
    for (kind, _), count in counts.items():
        totals[kind] += count
    keep = {kind for kind, _ in totals.most_common(_MAX_KIND_VALUES)}
    table: dict[str, dict[str, int]] = {}
    for (kind, status), count in counts.items():
        row = table.setdefault(kind if kind in keep else OTHER, {})
        row[status] = row.get(status, 0) + count
    return {
        kind: dict(sorted(row.items()))
        for kind, row in sorted(table.items(), key=lambda item: (-sum(item[1].values()), item[0]))
    }


def _ordered(counter: Mapping[str, int], *, by_count: bool) -> dict[str, int]:
    if by_count:
        return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
    return dict(sorted(counter.items()))


def _cohorts(values: Iterable[str]) -> list[str]:
    unique = set(values)
    numeric = sorted((value for value in unique if value.isdigit()), key=int)
    return [*numeric, *sorted(unique.difference(numeric))]


def _ordered_cohorts(counter: Mapping[str, int]) -> dict[str, int]:
    return {cohort: counter[cohort] for cohort in _cohorts(counter)}


def _snapshot_date(disposition: str | None) -> str:
    matched = _FILENAME.search(disposition or "")
    if matched is None:
        raise ValueError("the register download does not name its snapshot date")
    try:
        date.fromisoformat(f"{matched.group(1)[:4]}-{matched.group(1)[4:6]}-{matched.group(1)[6:]}")
    except ValueError:
        raise ValueError("the register download names an impossible snapshot date") from None
    return matched.group(1)


def _variant_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.stem}-with-withdrawn{destination.suffix}")


def _write_json(payload: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=destination.parent
        ) as output:
            temporary = Path(output.name)
            output.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _require_allowlisted(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != _ALLOWED_HOST:
        raise ValueError("the register URL is outside its allowlisted origin")

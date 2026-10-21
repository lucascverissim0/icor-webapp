"""Deterministic acquisition of Lithuania's registered passenger cars from Regitra.

Source: Regitra, "Atviri TP parko duomenys" (open vehicle-fleet data), published on
the Lithuanian open-data portal as dataset 1211 (https://data.gov.lt/datasets/1211)
under the Creative Commons Attribution 4.0 licence (CC BY 4.0). The ZIP (about
168 MB) holds one UTF-8 CSV (BOM, comma, every field quoted, 69 columns, about
2.5 million rows) and needs random access, so it is downloaded, aggregated and then
deleted by the caller. The server resets connections that carry no browser-like
User-Agent.

Only make, model, the year of first registration anywhere and a count leave a row.
The register's owner-type, owner-birth-date, municipality and county columns, like
every other column, are never read into the output. Private-owner vehicles that are
unique by make, model and municipality arrive with their labels replaced by
``Nuasmeninta`` ("anonymised"); they stay as a masked bucket for the loader to report.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Protocol, cast
from urllib.parse import urlsplit

from icor.evidence.register_aggregate import (
    AggregateFile,
    HashingReader,
    cohort_of,
    collapse,
    sha256_file,
    write_aggregate,
)

ARCHIVE_URL = "https://www.regitra.lt/wp-content/uploads/failai/Atviri_TP_parko_duomenys.zip"
DATASET_URL = "https://data.gov.lt/datasets/1211"
LICENCE = "CC BY 4.0"
_ALLOWED_HOST = "www.regitra.lt"
USER_AGENT = "Mozilla/5.0 (compatible; ICOR-evidence/1)"
#: The 2026-10-21 download was 167,667,174 bytes; the ceiling leaves room to grow.
MAX_ARCHIVE_BYTES = 400 * 1024 * 1024
#: The member was 960,765,704 bytes; a ZIP claiming more than this is refused unread.
MAX_MEMBER_BYTES = 4 * 1024**3
ARCHIVE_MEMBER = "Atviri_TP_parko_duomenys.csv"
#: The 69 columns of the member, in order (header checked on 2026-10-21).
SOURCE_HEADER = (
    "MARKE",
    "KOMERCINIS_PAV",
    "GAMINTOJO_PAV",
    "GAMINTOJO_PAV_BAZ",
    "TIPAS",
    "VARIANTAS",
    "VERSIJA",
    "ES_TIPO_PATVIRTINIMO_NR",
    "NAC_TIPO_PATVIRTINIMO_NR",
    "INDIVIDUAL_PATVIRTINIMO_NR",
    "INTERPOLIACIJA",
    "UZBAIGTUMO_PAKOPA",
    "VAIRAS_DESINEJE",
    "KATEGORIJA_PILNAI",
    "KATEGORIJA_KLASE",
    "KEB_KODAS",
    "KEB_PAVADINIMAS",
    "SPEC_KODAS",
    "SPEC_PAVADINIMAS",
    "KEB_KODAS_ES",
    "NUOSAVA_MASE",
    "NUOSAVA_MASE_BAZ",
    "MAKS_MASE",
    "MAKS_MASE_F2",
    "MAKS_MASE_F5",
    "BANDOMOJI_MASE",
    "DARBINIS_TURIS",
    "GALIA",
    "SUKIU_SK",
    "GALIA_ELEKTR",
    "DEGALAI",
    "DEGALU_REZIMAS",
    "ELEKTRINE_TP",
    "HIBRIDINES_TP_KATEGORIJA",
    "PAVARU_DEZES_TIPAS",
    "CO2_KIEKIS",
    "CO2_KIEKIS_WLTP",
    "EKO_NAUJOVES_KODAS",
    "CO2_SUMAZEJIMAS_NEDC",
    "CO2_SUMAZEJIMAS_WLTP",
    "ELEKTR_ENERG_SANAUD_NEDC",
    "ELEKTR_ENERG_SANAUD_WLTP_E",
    "ELEKTR_ENERG_SANAUD_WLTP_H",
    "ELEKTRINE_RIDA_NEDC",
    "ELEKTRINE_RIDA_WLTP_E",
    "ELEKTRINE_RIDA_WLTP_H",
    "TERSALU_LYGIS",
    "TERSALU_NORM_AKTO_NR",
    "RATU_BAZE",
    "TV_PLOTIS1",
    "TV_PLOTIS2",
    "SPALVA",
    "GALIOS_MASES_SANT",
    "MAKS_GREITIS",
    "SEDIMU_VIETU_SK",
    "STOVIMU_VIETU_SK",
    "GAMYBOS_METAI",
    "MODELIO_METAI",
    "RIDA",
    "PIRM_REG_DATA",
    "PIRM_REG_DATA_LT",
    "GALIOJIMO_LAIK",
    "PASKUTINES_REG_DATA",
    "DAE_STATUSAS",
    "KILMES_SALIS",
    "VALD_TIPAS",
    "VALD_GIM_DAT_INT",
    "SAVIVALDYBE",
    "APSKRITIS",
)
#: The only columns the aggregation reads; the rest of a row is discarded unread.
_USED_COLUMNS = (
    "MARKE",
    "KOMERCINIS_PAV",
    "KATEGORIJA_KLASE",
    "KATEGORIJA_PILNAI",
    "DAE_STATUSAS",
    "PIRM_REG_DATA",
    "PIRM_REG_DATA_LT",
)
#: Passenger cars by the register's own class (the full category must agree).
PASSENGER_CLASS = "M1"
#: "T" registered and in traffic. "N" is registered but not in traffic.
STRICT_STATUSES = frozenset({"T"})
VARIANT_STATUSES = frozenset({"T", "N"})
MASKED_LABEL = "Nuasmeninta"
#: Eurostat's Lithuanian passenger-car stock for 2025, the reconciliation target.
EUROSTAT_STOCK_2025 = 1_816_856
VARIANT_SUFFIX = "-with-not-in-traffic"
_PROFILE_FROM_YEAR = 2000
_TOP_MAKES = 40
_DATE = re.compile(r"(\d{4})-\d{2}-\d{2}\Z")


@dataclass(frozen=True, slots=True)
class Download:
    """What was served for the archive: where from, when, and the exact bytes."""

    url: str
    last_modified: str
    retrieved_at: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VariantResult:
    """The aggregate that also counts registered cars that are not in traffic."""

    path: Path
    artifact_bytes: int
    sha256: str
    status_rows: int
    group_count: int
    vehicles: int
    masked_vehicles: int
    blank_model_vehicles: int
    blank_cohort_vehicles: int


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    path: Path
    artifact_bytes: int
    sha256: str
    profile_path: Path
    snapshot_date: str
    source_bytes: int
    source_sha256: str
    source_rows: int
    category_rows: int
    status_rows: int
    category_disagreements: int
    group_count: int
    vehicles: int
    masked_vehicles: int
    blank_model_vehicles: int
    blank_cohort_vehicles: int
    variant: VariantResult


class _Response(Protocol):
    headers: Mapping[str, str]

    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> object: ...

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...


class _AllowlistedRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect that leaves the allowlisted origin before following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _require_allowlisted(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_allowlisted(request: urllib.request.Request, timeout: float) -> _Response:
    opener = urllib.request.build_opener(_AllowlistedRedirects)
    return opener.open(request, timeout=timeout)  # type: ignore[return-value]


def download_archive(
    destination: Path,
    *,
    opener: Callable[..., _Response] = open_allowlisted,
    url: str = ARCHIVE_URL,
    max_bytes: int = MAX_ARCHIVE_BYTES,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Download:
    """Download the allowlisted archive atomically, recording its bytes and sha256."""

    _require_allowlisted(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with opener(request, timeout=120) as response:
            _require_allowlisted(response.geturl())
            last_modified = response.headers.get("Last-Modified", "")
            retrieved_at = clock().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            reader = HashingReader(cast(BinaryIO, response), limit=max_bytes)
            with NamedTemporaryFile("wb", delete=False, dir=destination.parent) as output:
                temporary = Path(output.name)
                while chunk := reader.read(1024 * 1024):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary = None
        return Download(url, last_modified, retrieved_at, reader.bytes_read, reader.sha256)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def snapshot_date(archive: Path, *, member: str = ARCHIVE_MEMBER) -> date:
    """The member's date stamp: the day Regitra produced the snapshot."""

    with zipfile.ZipFile(archive) as bundle:
        year, month, day = _only_member(bundle, member).date_time[:3]
    return date(year, month, day)


def destination_for(template: Path, snapshot: date) -> Path:
    """``{date}`` in the template becomes the snapshot as YYYYMMDD."""

    return Path(str(template).replace("{date}", f"{snapshot:%Y%m%d}"))


def variant_path(destination: Path) -> Path:
    return destination.with_name(destination.stem + VARIANT_SUFFIX + destination.suffix)


def profile_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".profile.json")


def acquire_fleet(
    destination: Path,
    archive: Path,
    *,
    download: Download | None = None,
    member: str = ARCHIVE_MEMBER,
    max_member_bytes: int = MAX_MEMBER_BYTES,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AcquisitionResult:
    """Aggregate the archive into sorted counts-only CSVs and a counts-only profile.

    The strict aggregate counts passenger cars (``KATEGORIJA_KLASE`` is ``M1``) whose
    status is ``T``; the variant beside it also counts status ``N``. Cohort is the
    year of the first registration anywhere (``PIRM_REG_DATA``), left blank when the
    register gives none; the national date only feeds the profile. Masked and blank
    labels stay in the artifacts so that the loader reports them.
    """

    size = archive.stat().st_size
    digest = sha256_file(archive)
    if download is not None and (download.bytes, download.sha256) != (size, digest):
        raise ValueError("Regitra archive does not match its recorded download")
    if download is None:
        modified = datetime.fromtimestamp(archive.stat().st_mtime, UTC)
        download = Download(
            ARCHIVE_URL,
            modified.strftime("%Y-%m-%dT%H:%M:%SZ"),
            clock().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            size,
            digest,
        )

    tally = _scan(archive, member=member, max_member_bytes=max_member_bytes)
    snapshot = snapshot_date(archive, member=member)

    strict = write_aggregate(tally.strict, destination)
    variant = write_aggregate(tally.variant, variant_path(destination))
    strict_summary = _summarise(tally.strict)
    variant_summary = _summarise(tally.variant)
    profile = _profile(tally, download, snapshot, member, strict, variant)
    profile_file = profile_path(destination)
    _write_json(profile_file, profile)
    return AcquisitionResult(
        strict.path,
        strict.artifact_bytes,
        strict.sha256,
        profile_file,
        snapshot.isoformat(),
        download.bytes,
        download.sha256,
        tally.source_rows,
        tally.category_rows,
        sum(tally.status_vehicles[status] for status in STRICT_STATUSES),
        sum(tally.category_disagreements.values()),
        strict.group_count,
        strict.vehicles,
        *strict_summary,
        VariantResult(
            variant.path,
            variant.artifact_bytes,
            variant.sha256,
            sum(tally.status_vehicles[status] for status in VARIANT_STATUSES),
            variant.group_count,
            variant.vehicles,
            *variant_summary,
        ),
    )


@dataclass(slots=True)
class _Tally:
    source_rows: int = 0
    category_rows: int = 0
    unparseable_dates: int = 0
    strict: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    variant: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    status_vehicles: Counter[str] = field(default_factory=Counter)
    cohort_status: Counter[tuple[str, str]] = field(default_factory=Counter)
    cohort_national: Counter[tuple[str, str]] = field(default_factory=Counter)
    undated_national: Counter[tuple[str, str]] = field(default_factory=Counter)
    category_disagreements: Counter[str] = field(default_factory=Counter)


def _only_member(bundle: zipfile.ZipFile, member: str) -> zipfile.ZipInfo:
    infos = bundle.infolist()
    if [info.filename for info in infos] != [member]:
        raise ValueError("Regitra archive members are unsupported")
    return infos[0]


def _scan(archive: Path, *, member: str, max_member_bytes: int) -> _Tally:
    tally = _Tally()
    labels: dict[str, str] = {}

    def label(value: str) -> str:
        cached = labels.get(value)
        if cached is None:
            cached = labels[value] = collapse(value)
        return cached

    def year(value: str) -> str:
        text = value.strip()
        if not text:
            return ""
        found = _DATE.match(text)
        if found is None:
            tally.unparseable_dates += 1
            return ""
        return cohort_of(found.group(1))

    with zipfile.ZipFile(archive) as bundle:
        info = _only_member(bundle, member)
        if info.file_size > max_member_bytes:
            raise ValueError("Regitra archive member exceeds its size limit")
        with bundle.open(info) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.reader(stream, strict=True)
            try:
                header = next(reader, None)
                if header is None or tuple(header) != SOURCE_HEADER:
                    raise ValueError("Regitra archive schema is unsupported")
                at = {name: SOURCE_HEADER.index(name) for name in _USED_COLUMNS}
                width = len(SOURCE_HEADER)
                i_make, i_model = at["MARKE"], at["KOMERCINIS_PAV"]
                i_class, i_full = at["KATEGORIJA_KLASE"], at["KATEGORIJA_PILNAI"]
                i_status = at["DAE_STATUSAS"]
                i_world, i_national = at["PIRM_REG_DATA"], at["PIRM_REG_DATA_LT"]
                for row in reader:
                    if len(row) != width:
                        raise ValueError("Regitra archive row is malformed")
                    tally.source_rows += 1
                    in_class = row[i_class].strip() == PASSENGER_CLASS
                    in_full = row[i_full].strip().startswith(PASSENGER_CLASS)
                    if in_class != in_full:
                        tally.category_disagreements["class" if in_class else "full"] += 1
                    if not in_class:
                        continue
                    tally.category_rows += 1
                    status = row[i_status].strip()
                    cohort = year(row[i_world])
                    national = year(row[i_national])
                    tally.status_vehicles[status] += 1
                    tally.cohort_status[(cohort, status)] += 1
                    if cohort == "":
                        tally.undated_national[(status, national)] += 1
                    if status in VARIANT_STATUSES:
                        key = (label(row[i_make]), label(row[i_model]), cohort)
                        tally.variant[key] += 1
                        if status in STRICT_STATUSES:
                            tally.strict[key] += 1
                            tally.cohort_national[(cohort, national)] += 1
            except UnicodeDecodeError as error:
                raise ValueError("Regitra archive is not UTF-8") from error
            except csv.Error as error:
                raise ValueError("Regitra archive row is malformed") from error
    return tally


def _summarise(counts: Mapping[tuple[str, str, str], int]) -> tuple[int, int, int]:
    """Masked, blank-model and blank-cohort vehicles of an aggregate."""

    masked = blank_model = blank_cohort = 0
    for (make, model, cohort), vehicles in counts.items():
        if (
            make.casefold() == MASKED_LABEL.casefold()
            or model.casefold() == MASKED_LABEL.casefold()
        ):
            masked += vehicles
        if model == "":
            blank_model += vehicles
        if cohort == "":
            blank_cohort += vehicles
    return masked, blank_model, blank_cohort


def _numbered(counter: Mapping[str, int]) -> dict[str, int]:
    return {key or "blank": counter[key] for key in sorted(counter)}


def _profile(
    tally: _Tally,
    download: Download,
    snapshot: date,
    member: str,
    strict_agg: AggregateFile,
    variant_agg: AggregateFile,
) -> dict[str, object]:
    strict_masked, strict_blank_model, strict_blank_cohort = _summarise(tally.strict)
    variant_masked, variant_blank_model, variant_blank_cohort = _summarise(tally.variant)

    strict_cohorts: Counter[str] = Counter()
    for (cohort, status), vehicles in tally.cohort_status.items():
        if status in STRICT_STATUSES:
            strict_cohorts[cohort] += vehicles

    status_by_cohort: dict[str, dict[str, int]] = {}
    for (cohort, status), vehicles in sorted(tally.cohort_status.items()):
        if cohort and int(cohort) >= _PROFILE_FROM_YEAR:
            status_by_cohort.setdefault(cohort, {})[status or "blank"] = vehicles

    by_cohort_national: dict[str, dict[str, int]] = {}
    comparison: dict[str, dict[str, int]] = {}
    by_national: Counter[str] = Counter()
    national_before_cohort = 0
    for (cohort, national), vehicles in tally.cohort_national.items():
        if national:
            by_national[national] += vehicles
        if cohort and national and int(national) < int(cohort):
            national_before_cohort += vehicles
    years = sorted(
        {
            year
            for year in (*strict_cohorts, *by_national)
            if year and int(year) >= _PROFILE_FROM_YEAR
        }
    )
    for year in years:
        row = {
            "by_cohort": strict_cohorts.get(year, 0),
            "by_national_year": by_national.get(year, 0),
            "cohort_registered_in_lt_same_year": tally.cohort_national.get((year, year), 0),
            "cohort_registered_in_lt_later": sum(
                vehicles
                for (cohort, national), vehicles in tally.cohort_national.items()
                if cohort == year and national and int(national) > int(year)
            ),
            "cohort_without_lt_date": tally.cohort_national.get((year, ""), 0),
        }
        comparison[year] = row
        by_cohort_national[year] = {
            national or "blank": vehicles
            for (cohort, national), vehicles in sorted(tally.cohort_national.items())
            if cohort == year
        }

    undated: dict[str, dict[str, int]] = {}
    for (status, national), vehicles in sorted(tally.undated_national.items()):
        undated.setdefault(status or "blank", {})[national or "blank"] = vehicles

    makes: Counter[str] = Counter()
    for (make, _model, _cohort), vehicles in tally.strict.items():
        makes[make] += vehicles
    top_makes = [
        {"make": make or "blank", "vehicles": vehicles}
        for make, vehicles in sorted(makes.items(), key=lambda item: (-item[1], item[0]))[
            :_TOP_MAKES
        ]
    ]

    strict_status_rows = sum(tally.status_vehicles[status] for status in STRICT_STATUSES)
    variant_status_rows = sum(tally.status_vehicles[status] for status in VARIANT_STATUSES)
    return {
        "source": {
            "name": "Regitra, Atviri TP parko duomenys",
            "dataset": DATASET_URL,
            "licence": LICENCE,
            "url": download.url,
            "http_last_modified": download.last_modified,
            "retrieved_at_utc": download.retrieved_at,
            "bytes": download.bytes,
            "sha256": download.sha256,
            "member": member,
            "snapshot_date": snapshot.isoformat(),
        },
        "filters": {
            "category": f"KATEGORIJA_KLASE == {PASSENGER_CLASS!r}",
            "strict_status": sorted(STRICT_STATUSES),
            "variant_status": sorted(VARIANT_STATUSES),
            "cohort": "year of PIRM_REG_DATA (first registration anywhere); blank when empty",
        },
        "funnel": {
            "source_rows": tally.source_rows,
            "category_rows": tally.category_rows,
            "strict_status_rows": strict_status_rows,
            "variant_status_rows": variant_status_rows,
            "category_cross_check": {
                "class_m1_but_full_category_not_m1": tally.category_disagreements["class"],
                "full_category_m1_but_class_not_m1": tally.category_disagreements["full"],
            },
        },
        "aggregates": {
            "strict": {
                "file": strict_agg.path.name,
                "bytes": strict_agg.artifact_bytes,
                "sha256": strict_agg.sha256,
                "group_count": strict_agg.group_count,
                "vehicles": strict_agg.vehicles,
            },
            "with_not_in_traffic": {
                "file": variant_agg.path.name,
                "bytes": variant_agg.artifact_bytes,
                "sha256": variant_agg.sha256,
                "group_count": variant_agg.group_count,
                "vehicles": variant_agg.vehicles,
            },
        },
        "reconciliation": {
            "eurostat_lt_2025_stock": EUROSTAT_STOCK_2025,
            "strict_vehicles": strict_agg.vehicles,
            "strict_minus_eurostat": strict_agg.vehicles - EUROSTAT_STOCK_2025,
            "strict_over_eurostat": round(strict_agg.vehicles / EUROSTAT_STOCK_2025, 4),
            "with_not_in_traffic_vehicles": variant_agg.vehicles,
            "with_not_in_traffic_minus_eurostat": variant_agg.vehicles - EUROSTAT_STOCK_2025,
            "with_not_in_traffic_over_eurostat": round(
                variant_agg.vehicles / EUROSTAT_STOCK_2025, 4
            ),
        },
        "vehicles_by_status_within_category": _numbered(tally.status_vehicles),
        "vehicles_by_cohort_strict": _numbered(strict_cohorts),
        f"vehicles_by_cohort_and_status_{_PROFILE_FROM_YEAR}_plus": status_by_cohort,
        f"national_year_comparison_strict_{_PROFILE_FROM_YEAR}_plus": comparison,
        f"cohort_by_national_year_strict_{_PROFILE_FROM_YEAR}_plus": by_cohort_national,
        "strict_vehicles_with_national_year_before_cohort": national_before_cohort,
        "national_year_of_vehicles_without_first_registration_date_by_status": undated,
        "masked": {
            "label": MASKED_LABEL,
            "strict_vehicles": strict_masked,
            "with_not_in_traffic_vehicles": variant_masked,
        },
        "blank_model_vehicles": {
            "strict": strict_blank_model,
            "with_not_in_traffic": variant_blank_model,
        },
        "blank_cohort_vehicles": {
            "strict": strict_blank_cohort,
            "with_not_in_traffic": variant_blank_cohort,
        },
        "unparseable_date_values": tally.unparseable_dates,
        "top_makes_strict": top_makes,
    }


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=path.parent
        ) as output:
            temporary = Path(output.name)
            output.write(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _require_allowlisted(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != _ALLOWED_HOST:
        raise ValueError("Regitra archive URL is outside its allowlisted origin")

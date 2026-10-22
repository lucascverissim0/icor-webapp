"""Deterministic acquisition of the Slovenian registered passenger-car fleet.

Source: "Evidenca registriranih vozil - presek stanja" (register of registered
vehicles, snapshot), published as open data on the Slovenian OPSI portal
(podatki.gov.si) under CC BY 4.0. Attribution: OPSI, podatki.gov.si.

The publisher stopped issuing new snapshots "do nadaljnjega" (until further
notice) for GDPR reasons, so this is a frozen snapshot taken 2023-06-30 (HTTP
Last-Modified 2023-07-20). The file is static, so its byte count and sha256 are
pinned below and re-acquisition reproduces the aggregate.

The ZIP holds VINs and owner and user data (age, sex, type, location). It is a
ZIP, so it needs random access: it is downloaded, checked against the pins,
aggregated and deleted by the caller. Only make, model, the year of first
registration anywhere and a count leave it; the counts-only profile written next to
the aggregate holds no row-level value either.
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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.parse import urlsplit

from icor.evidence.register_aggregate import (
    AggregateFile,
    HashingReader,
    cohort_of,
    collapse,
    sha256_file,
    write_aggregate,
)

ARCHIVE_URL = (
    "https://podatki.gov.si/dataset/75b38301-c0bc-4371-a225-8ba026a66142"
    "/resource/81c9ca70-bb57-41ff-b96f-bb1b506ee2b9/download/"
    "opsipodatkireg.vozilih30.06.2023.zip"
)
# The server answered 200 directly (no redirect) when checked on 2026-10-21, so the
# portal host is the only one allowed.
_ALLOWED_HOST = "podatki.gov.si"
SNAPSHOT_DATE = "2023-06-30"
ARCHIVE_LAST_MODIFIED = "Thu, 20 Jul 2023 09:47:54 GMT"
# The frozen snapshot, measured on 2026-10-21; the publisher no longer replaces it.
ARCHIVE_BYTES = 384_940_669
ARCHIVE_SHA256 = "346ec80a6a1c6d37ac46719f25073eb8d090ded122999f751de4d6ba17eb2598"
_MEMBER_DIRECTORY = "OPSI_podatki_reg._vozilih_30.06.2023/"
#: Together the three files are the whole register.
MEMBERS = tuple(f"{_MEMBER_DIRECTORY}Nio_vozila_{number}.csv" for number in (1, 2, 3))
EXPLANATION_MEMBER = f"{_MEMBER_DIRECTORY}Pojasnilo za uporabo.txt"

FIRST_REGISTRATION = "B-Datum prve registracije vozila"
NATIONAL_REGISTRATION = "2A-Datum prve registracije vozila v SLO"
STATUS_ID = "Status vozila (id)"
STATUS_NAME = "Status vozila (opis)"
MAKE = "D1-Znamka"
MODEL = "D3-Komerc oznaka"
CATEGORY = "J-Kategorija in vrsta vozila (oznaka)"
PASSENGER_CATEGORIES = frozenset({"M1", "M1G"})
#: Status 1 is "registrirano" (registered); 3 is "v postopku" (in procedure).
REGISTERED_STATUS = "1"
#: The 102 columns, as the header of every member spells them (ASCII, except one
#: UTF-8 column name; the rows are cp1250). Owner, user, VIN, plate-area and
#: type-approval columns are listed only so that a changed layout is caught.
SOURCE_HEADER = (
    "B-Datum prve registracije vozila",
    "2A-Datum prve registracije vozila v SLO",
    "1K-Oznaka potrdila o skladnosti",
    "Datum statusa vozila",
    "Status vozila (id)",
    "Status vozila (opis)",
    "Izvajalna enota prve registracije",
    "4A-Registrsko obmocje tablice prve registracije",
    "4A-Podvrsta tablice prve registracije",
    "C-Starost uporabnika vozila",
    "C-Ali je uporabnik pravna ali fizicna oseba",
    "C-Spol uporabnika (ce gre za fizicno osebo)",
    "C-Ali je uporabnik tudi lastnik vozila",
    "C13-Upravna enota uporabnika vozila (oznaka)",
    "C13-Upravna enota uporabnika vozila (opis)",
    "C13-Obcina uporabnika vozila (oznaka)",
    "C13-Obcina uporabnika vozila (opis)",
    "C2-Starost lastnika vozila",
    "C2-Ali je lastnik pravna ali fizicna oseba",
    "C2-Spol lastnika (ce gre za fizicno osebo)",
    "D1-Znamka",
    "D2-Tovar oznaka",
    "D3-Komerc oznaka",
    "D42-Drzava (opis)",
    "D42-Drzava (koda)",
    "D5-Datum izdaje COC dokumenta oz prve registracije",
    "E-VIN",
    "E1-Identifikacijska koda",
    "F1-Najvecja tehnicno dovoljena masa vozila",
    "F2-Najvecja dovoljena masa vozila pri registraciji",
    "F3-Najvecja dovoljena masa skupine vozil pri registraciji",
    "G-Masa vozila",
    "J-Kategorija in vrsta vozila (oznaka)",
    "J-Kategorija in vrsta vozila (opis)",
    "K-Homologacijska oznaka vozila",
    "L-Stevilo osi",
    "M-Medosje",
    "M1-Zadnji previs",
    "N-Razporeditev najv tehnicno dovoljene mase na osi",
    "N1-Dovoljene osne obremenitve",
    "O11-Priklopnik",
    "O12-Polpriklopnik",
    "O13-Priklopnik s centralno osjo",
    "O2-Nezavirano priklopno vozilo",
    "O3-Navpicna obremenitev vlecne naprave / sedla",
    "O4-Tlak v napajalnem vodu zavornega sistema",
    "P11-Delovna prostornina",
    "P12-Nazivna moc",
    "P13-Vrsta goriva (opis)",
    "P13-Vrsta goriva (oznaka)",
    "P14-Nazivna vrtilna frekvenca",
    "P15-Oznaka motorja",
    "P21-Tip",
    "P22-Nazivna trajna moc",
    "P23-Delovna napetost",
    "P24-Pogonske baterije",
    "P25-Oznaka motorja",
    "Q-Razmerje moc/masa (samo za motorna kolesa)",
    "R-Barva vozila (oznaka)",
    "R-Barva vozila (opis)",
    "S1-Stevilo sedezev (vkljucno z vozniskim)",
    "S2-Stevilo stojisc",
    "T-Najvisja hitrost",
    "U1-V mirovanju",
    "U2-Pri vrtilni frekvenci motorja",
    "U3-V voznji",
    "V1-CO",
    "V10-Vrtilna frekvenca prostega teka, nizka",
    "V101-Vsebina CO",
    "V102-Temperatura olja",
    "V11-Vrtilna frekvenca prostega teka, visoka",
    "V111-Vsebina CO",
    "V112-Vrednost Lambda",
    "V113-Temperatura olja",
    "V2-HC",
    "V3-Nox",
    "V4-HC + Nox",
    "V5-Delci pri dizel motorjih",
    "V6-Korigiran absorpcijski koeficient pri dizel motorjih",
    "V61-Vrtilna frekvenca prostega teka",
    "V62-Najvisja vrtilna frekvenca motorja",
    "V63-Temperatura olja",
    "V7-CO2",
    "CO2_WLTP",
    "V8-Kombinirana poraba goriva",
    "V9-Podatek o okoljevarstveni kategoriji vozila",
    "X-Oblika nadgradnje (oznaka)",
    "X-Oblika nadgradnje (opis)",
    "X1-Dodatni opis nadgradnje",
    "X2-Homologacijska oznaka zascitne konstrukcije",
    "Y1-Dolzina",
    "Y2-Sirina",
    "Y3-Visina",
    "Z1-Dovoljene pnevmatike in platisca",
    "Z2-Homologacijska oznaka vlecne naprave",
    "Z21-D Đ vrednost",
    "Datum izdaje potrdila o skladnosti",
    "Okoljevarstvena oznaka",
    "Pogonska os z zracnim vzmetenjem ali priznanim ekvivalentom",
    "8B-Namen vozila",
    "Komerc oznaka  do prvega /",
    "Prevozeni kilometri milje",
)

_COLUMNS = {name: index for index, name in enumerate(SOURCE_HEADER)}
_COHORT_FLOOR = 2000
_TOP_MAKES = 40
_DATE = re.compile(r"\d{1,2}\.\d{1,2}\.(\d{4})\Z")


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    path: Path
    artifact_bytes: int
    artifact_sha256: str
    profile_path: Path
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
    model_head: AggregateFile | None


@dataclass(frozen=True, slots=True)
class Download:
    path: Path
    bytes: int
    sha256: str
    last_modified: str
    retrieved_utc: str


class _Response(Protocol):
    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> object: ...

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...


class _AllowlistedRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect before it is followed when it leaves the allowlisted origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_allowlisted(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(request: urllib.request.Request, timeout: float) -> _Response:
    return urllib.request.build_opener(_AllowlistedRedirects()).open(request, timeout=timeout)


def download_archive(
    destination: Path,
    *,
    opener: Callable[..., _Response] = _open,
    url: str = ARCHIVE_URL,
    expected_bytes: int = ARCHIVE_BYTES,
    expected_sha256: str = ARCHIVE_SHA256,
) -> Download:
    """Download the allowlisted archive atomically, checking it against the pins.

    The server ignores range upper bounds, so the whole file is fetched. Nothing is
    left at ``destination`` unless the bytes and sha256 match the pinned release.
    """

    _require_allowlisted(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ICOR-evidence/1"})
        with opener(request, timeout=120) as response:
            _require_allowlisted(response.geturl())
            reader = HashingReader(response, limit=expected_bytes)
            with NamedTemporaryFile("wb", delete=False, dir=destination.parent) as output:
                temporary = Path(output.name)
                while chunk := reader.read(1024 * 1024):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            headers = getattr(response, "headers", None)
            last_modified = str(headers.get("Last-Modified") or "") if headers is not None else ""
        if reader.bytes_read != expected_bytes or reader.sha256 != expected_sha256:
            raise ValueError(
                "SI archive does not match the pinned release "
                f"({reader.bytes_read} of {expected_bytes} bytes received)"
            )
        os.replace(temporary, destination)
        temporary = None
        return Download(
            destination,
            reader.bytes_read,
            reader.sha256,
            last_modified,
            datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def acquire_fleet(
    destination: Path,
    archive: Path,
    *,
    expected_bytes: int = ARCHIVE_BYTES,
    expected_sha256: str = ARCHIVE_SHA256,
    source_url: str = ARCHIVE_URL,
    last_modified: str = ARCHIVE_LAST_MODIFIED,
    retrieved_utc: str | None = None,
    model_head_destination: Path | None = None,
) -> AcquisitionResult:
    """Aggregate the pinned archive into a counts-only CSV and a counts-only profile.

    Cars are category M1 or M1G with status 1 (registered). The cohort is the year of
    the first registration anywhere (``B-Datum prve registracije vozila``), not the
    Slovenian date, which a used import receives later. Blank or masked labels stay in
    the artifact so that the loader reports them against the manifest.

    The model label is ``D3-Komerc oznaka`` as the register writes it, which is
    "MODEL / BODY AND ENGINE / FUEL" (for example ``GOLF / 1.6 / TDI``). When
    ``model_head_destination`` is given, a second aggregate with the model alone (the
    text before the first slash) is written from the same pass.
    """

    source_bytes = archive.stat().st_size
    source_sha256 = sha256_file(archive)
    if source_bytes != expected_bytes or source_sha256 != expected_sha256:
        raise ValueError("SI archive does not match the pinned release")
    if retrieved_utc is None:
        retrieved_utc = datetime.fromtimestamp(archive.stat().st_mtime, UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    tally = _Tally()
    with zipfile.ZipFile(archive) as bundle:
        if sorted(bundle.namelist()) != sorted((*MEMBERS, EXPLANATION_MEMBER)):
            raise ValueError("SI archive members are unsupported")
        for member in MEMBERS:
            with bundle.open(member) as raw:
                for fields in _rows(raw, member):
                    tally.add(fields)

    aggregate = write_aggregate(tally.counts, destination)
    head = (
        write_aggregate(tally.head_counts, model_head_destination)
        if model_head_destination is not None
        else None
    )
    profile_path = destination.with_name(destination.name + ".profile.json")
    _write_json(
        profile_path,
        tally.profile(
            source={
                "dataset": "Evidenca registriranih vozil - presek stanja",
                "publisher_portal": "OPSI, podatki.gov.si",
                "licence": "CC BY 4.0",
                "snapshot_date": SNAPSHOT_DATE,
                "url": source_url,
                "last_modified": last_modified,
                "retrieved_utc": retrieved_utc,
                "bytes": source_bytes,
                "sha256": source_sha256,
            },
            aggregate=aggregate,
            head=head,
        ),
    )
    return AcquisitionResult(
        destination,
        aggregate.artifact_bytes,
        aggregate.sha256,
        profile_path,
        source_bytes,
        source_sha256,
        tally.source_rows,
        tally.category_rows,
        tally.status_rows,
        aggregate.group_count,
        aggregate.vehicles,
        tally.masked_vehicles,
        tally.blank_model_vehicles,
        tally.blank_cohort_vehicles,
        head,
    )


class _Tally:
    """Counts only: no row-level value is kept beyond the current row."""

    def __init__(self) -> None:
        self.source_rows = self.category_rows = self.status_rows = 0
        self.masked_vehicles = self.blank_model_vehicles = 0
        self.blank_cohort_vehicles = self.blank_make_vehicles = 0
        self.counts: Counter[tuple[str, str, str]] = Counter()
        self.head_counts: Counter[tuple[str, str, str]] = Counter()
        self.status_all: Counter[tuple[str, str]] = Counter()
        self.status_in_category: Counter[tuple[str, str]] = Counter()
        self.status_by_cohort: dict[str, Counter[str]] = {}
        self.cohorts: Counter[str] = Counter()
        self.national_years: Counter[str] = Counter()
        self.national_before_floor = self.national_blank = 0
        self.imports: dict[str, Counter[str]] = {}
        self.makes: Counter[str] = Counter()

    def add(self, fields: list[str]) -> None:
        self.source_rows += 1
        status = (
            collapse(fields[_COLUMNS[STATUS_ID]]),
            collapse(fields[_COLUMNS[STATUS_NAME]]),
        )
        self.status_all[status] += 1
        if collapse(fields[_COLUMNS[CATEGORY]]) not in PASSENGER_CATEGORIES:
            return
        self.category_rows += 1
        self.status_in_category[status] += 1
        cohort = cohort_of(_year_of(fields[_COLUMNS[FIRST_REGISTRATION]]))
        if cohort and int(cohort) >= _COHORT_FLOOR:
            self.status_by_cohort.setdefault(cohort, Counter())[status[0]] += 1
        if status[0] != REGISTERED_STATUS:
            return
        self.status_rows += 1
        make = collapse(fields[_COLUMNS[MAKE]])
        model = collapse(fields[_COLUMNS[MODEL]])
        self.counts[(make, model, cohort)] += 1
        self.head_counts[(make, collapse(model.split("/", 1)[0]), cohort)] += 1
        self.makes[make] += 1
        self.cohorts[cohort] += 1
        self.blank_cohort_vehicles += not cohort
        self.blank_model_vehicles += not model
        self.blank_make_vehicles += not make
        self.masked_vehicles += _is_masked(make) or _is_masked(model)
        national = cohort_of(_year_of(fields[_COLUMNS[NATIONAL_REGISTRATION]]))
        if not national:
            self.national_blank += 1
        elif int(national) < _COHORT_FLOOR:
            self.national_before_floor += 1
        else:
            self.national_years[national] += 1
        if cohort and int(cohort) >= _COHORT_FLOOR:
            relation = (
                "national_blank"
                if not national
                else "same_year"
                if national == cohort
                else "national_later"
                if national > cohort
                else "national_earlier"
            )
            self.imports.setdefault(cohort, Counter())["vehicles"] += 1
            self.imports[cohort][relation] += 1

    def profile(
        self, *, source: dict[str, object], aggregate: AggregateFile, head: AggregateFile | None
    ) -> dict[str, object]:
        floor = str(_COHORT_FLOOR)
        return {
            "source": source,
            "filters": {
                "category": f"{CATEGORY} in {sorted(PASSENGER_CATEGORIES)}",
                "status": f"{STATUS_ID} = {REGISTERED_STATUS}",
                "make": MAKE,
                "model": MODEL,
                "cohort": f"year of {FIRST_REGISTRATION} (first registration anywhere)",
                "national_year": f"year of {NATIONAL_REGISTRATION}",
            },
            "funnel": {
                "source_rows": self.source_rows,
                "category_rows": self.category_rows,
                "status_rows": self.status_rows,
            },
            "status_values_all_rows": _statuses(self.status_all),
            "vehicles_by_status_within_category": _statuses(self.status_in_category),
            "vehicles_by_cohort_strict": _by_year(self.cohorts),
            f"vehicles_by_cohort_and_status_{floor}_plus": {
                cohort: dict(sorted(self.status_by_cohort[cohort].items()))
                for cohort in sorted(self.status_by_cohort)
            },
            f"vehicles_by_national_year_{floor}_plus": _by_year(self.national_years),
            "national_year_before_floor_vehicles": self.national_before_floor,
            "national_year_blank_vehicles": self.national_blank,
            f"used_imports_by_cohort_{floor}_plus": {
                cohort: {
                    key: self.imports[cohort][key]
                    for key in (
                        "vehicles",
                        "same_year",
                        "national_later",
                        "national_earlier",
                        "national_blank",
                    )
                }
                for cohort in sorted(self.imports)
            },
            "masked_vehicles": self.masked_vehicles,
            "blank_model_vehicles": self.blank_model_vehicles,
            "blank_cohort_vehicles": self.blank_cohort_vehicles,
            "blank_make_vehicles": self.blank_make_vehicles,
            "top_makes": [
                {"make": make, "vehicles": vehicles}
                for make, vehicles in sorted(
                    self.makes.items(), key=lambda item: (-item[1], item[0])
                )[:_TOP_MAKES]
            ],
            "aggregate": _aggregate(aggregate),
            "model_head_aggregate": _aggregate(head) if head is not None else None,
        }


def _rows(raw: io.BufferedIOBase, member: str) -> Iterator[list[str]]:
    # The header spells one name in UTF-8; every row is cp1250 (one label holds a
    # byte cp1250 leaves undefined, which becomes U+FFFD).
    try:
        header = next(
            csv.reader([raw.readline().decode("utf-8").rstrip("\r\n")], delimiter=";"), []
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"SI register header is unsupported in {member}") from error
    if tuple(header) != SOURCE_HEADER:
        raise ValueError(f"SI register header is unsupported in {member}")
    stream = io.TextIOWrapper(raw, encoding="cp1250", errors="replace", newline="")
    for fields in csv.reader(stream, delimiter=";"):
        if len(fields) != len(SOURCE_HEADER):
            raise ValueError(f"SI register row is malformed in {member}")
        yield fields


def _year_of(value: str) -> str:
    match = _DATE.match(value.strip())
    return match.group(1) if match else ""


def _is_masked(label: str) -> bool:
    """A placeholder such as "/" that holds no letter or digit."""

    return bool(label) and not any(character.isalnum() for character in label)


def _by_year(counts: Counter[str]) -> dict[str, int]:
    ordered = sorted(counts.items(), key=lambda item: (item[0] == "", item[0]))
    return {(year or "blank"): vehicles for year, vehicles in ordered}


def _statuses(counts: Counter[tuple[str, str]]) -> list[dict[str, object]]:
    return [
        {"id": status_id, "description": description, "vehicles": vehicles}
        for (status_id, description), vehicles in sorted(counts.items())
    ]


def _aggregate(aggregate: AggregateFile) -> dict[str, object]:
    return {
        "file": aggregate.path.name,
        "bytes": aggregate.artifact_bytes,
        "sha256": aggregate.sha256,
        "group_count": aggregate.group_count,
        "vehicles": aggregate.vehicles,
    }


def _write_json(destination: Path, payload: dict[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=destination.parent
        ) as output:
            temporary = Path(output.name)
            output.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
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
        raise ValueError("SI archive URL is outside its allowlisted origin")

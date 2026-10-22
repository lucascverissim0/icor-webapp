"""KBA FZ10 December workbooks 2004-2012: registrations by make and model series.

The EEA's model-level monitoring starts in 2010, so Germany's cars first
registered before it were counted only as a national total. KBA published the
same register's annual counts by model series in one legacy `.xls` workbook per
year. These rules read them, and each fails closed:

- The sheet is the year's December issue. Its header names `Marke/Modellreihe`,
  and the annual cumulative count (`Jan./Dezember` or `Jan.-Dezember`) is the
  third column, under `Anzahl`.
- A label without a count opens a make; `ZUSAMMEN` closes it and must equal the
  sum of its rows.
- A model row's label starts with its make (`ALFA` under `ALFA ROMEO`). Under a
  group header (`BMW, MINI`, `GENERAL MOTORS`) the label's own brand is the make:
  GM labels name it after `GM`, and GM's Chevrolets without one (`GM CORVETTE`)
  are Chevrolets. `BMW MINI` is the MINI.
- Rows without a model series stay observations, so the totals reconcile, but
  are never attributed: `SONSTIGE` inside a make, a make published as one row
  (`ASTON`, `BENTLEY`), `SONSTIGE HERSTELLER`, and `TOYOTA LEXUS` (every Lexus
  before KBA named the series).
- All rows sum to `Neuzulassungen insgesamt`, the published total.

Every observation is flagged as archived registrations whose labels do not teach
the identity vocabulary: the years add their own cohorts and change nothing the
monitored years planned or folded.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Protocol

import xlrd

from icor.domain.evidence import (
    ARCHIVE_REGISTRATIONS_FLAG,
    VOCABULARY_EXCLUDED_FLAG,
    EvidenceConfidence,
    MappingStatus,
    Measure,
    Observation,
    PeriodPrecision,
    PublicationStatus,
)
from icor.evidence.normalization import normalize_vehicle_label, stable_evidence_id
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

PARSER_NAME = "kba_fz10_xls_v1"
#: The December workbooks whose layout these rules were verified against.
YEARS = range(2004, 2013)
_ANNUAL_COLUMN = 2
_LABEL_HEADER = "Marke/\nModellreihe"
#: Single makes whose model labels abbreviate the make.
_PREFIXES = {"ALFA ROMEO": "ALFA"}
#: Group headers and the brands their labels may start with.
_GROUP_BRANDS = {
    "BMW, MINI": ("BMW", "ALPINA"),
    "CHRYSLER, JEEP": ("CHRYSLER", "JEEP", "DODGE"),
    "CHRYSLER, JEEP, DODGE": ("CHRYSLER", "JEEP", "DODGE"),
    "GENERAL MOTORS": ("GM",),
    "OPEL, GM ...": ("OPEL", "CADILLAC", "CHEVROLET", "GM"),
    "MG ROVER, AUSTIN": ("MG ROVER",),
    "NISSAN, INFINITI": ("NISSAN", "INFINITI"),
    "RENAULT, DACIA": ("RENAULT", "DACIA"),
    "TOYOTA, LEXUS": ("TOYOTA", "LEXUS"),
}
#: Labels that name their make differently: (make, model series or None).
_SPECIAL_LABELS = {"BMW MINI": ("MINI", "MINI"), "TOYOTA LEXUS": ("LEXUS", None)}
_GM_BRANDS = ("CADILLAC", "CHEVROLET", "HUMMER")
_GM_CHEVROLETS = ("CORVETTE", "TRAILBLAZER", "CAMARO")
_UNNAMED_MAKE = "SONSTIGE HERSTELLER"


class _Sheet(Protocol):
    name: str
    nrows: int

    def cell_value(self, rowx: int, colx: int) -> object: ...


@dataclass(frozen=True, slots=True)
class ArchiveRow:
    """One published count: a model series, or a row naming none (`model` None)."""

    row_number: int
    label: str
    make: str
    model: str | None
    vehicles: int
    flag: str | None = None


@dataclass(frozen=True, slots=True)
class ArchiveTable:
    year: int
    sheet_name: str
    rows: tuple[ArchiveRow, ...]
    raw_count: int
    rejected_count: int
    published_total: int


def read_workbook(path: object, year: int) -> ArchiveTable:
    """Read one December workbook, failing closed on any unverified layout."""

    if year not in YEARS:
        raise ValueError("KBA archive year is outside the verified layouts")
    workbook = xlrd.open_workbook(str(path), on_demand=True)
    try:
        if workbook.nsheets != 1:
            raise ValueError("KBA archive workbook must hold one sheet")
        return read_sheet(workbook.sheet_by_index(0), year)
    finally:
        workbook.release_resources()


def read_sheet(sheet: _Sheet, year: int) -> ArchiveTable:
    if not re.fullmatch(rf"Dezember[- ]{year}", sheet.name.strip()):
        raise ValueError("KBA archive sheet is not the year's December issue")
    header = next(
        (row for row in range(min(sheet.nrows, 8)) if sheet.cell_value(row, 0) == _LABEL_HEADER),
        None,
    )
    if header is None:
        raise ValueError("KBA archive header is missing")
    if (
        not re.fullmatch(
            rf"Jan\.[/-]Dezember {year}", str(sheet.cell_value(header + 1, _ANNUAL_COLUMN))
        )
        or str(sheet.cell_value(header + 2, _ANNUAL_COLUMN)) != "Anzahl"
    ):
        raise ValueError("KBA archive annual column is unsupported")
    first = header + 3
    cells = []
    for row in range(first, sheet.nrows):
        label = _label(sheet.cell_value(row, 0))
        value = sheet.cell_value(row, _ANNUAL_COLUMN)
        if label is None:
            # 2008 on, a row numbers the columns under the header.
            if value in ("", None) or (row == first and value == 2.0):
                continue
            raise ValueError("KBA archive count has no label")
        cells.append((row + 1, label, value))
    rows, raw_count, rejected_count, total = parse_rows(cells)
    return ArchiveTable(year, sheet.name.strip(), rows, raw_count, rejected_count, total)


def parse_rows(
    cells: Iterable[tuple[int, str, object]],
) -> tuple[tuple[ArchiveRow, ...], int, int, int]:
    """Apply the make, model and subtotal rules to labelled annual counts."""

    rows: list[ArchiveRow] = []
    header: str | None = None
    since_header = 0
    raw_count = rejected_count = 0
    for row_number, label, value in cells:
        if value in ("", None):
            if label.upper() == "ZUSAMMEN":
                raise ValueError("KBA archive make subtotal is empty")
            header, since_header = label, 0
            continue
        vehicles = _vehicle_count(value)
        raw_count += 1
        if label.upper() == "NEUZULASSUNGEN INSGESAMT":
            rejected_count += 1
            if sum(row.vehicles for row in rows) != vehicles:
                raise ValueError("KBA archive rows do not match the published total")
            return tuple(rows), raw_count, rejected_count, vehicles
        if label.upper() == "ZUSAMMEN":
            if header is None:
                raise ValueError("KBA archive subtotal closes no make")
            if sum(row.vehicles for row in rows[len(rows) - since_header :]) != vehicles:
                raise ValueError(f"KBA archive subtotal of {header} does not match its rows")
            rejected_count += 1
            header = None
            continue
        if header is None:
            # A make published as one row, or every other make together.
            make = "SONSTIGE" if label == _UNNAMED_MAKE else label
            rows.append(ArchiveRow(row_number, label, make, None, vehicles, "source_model_missing"))
            continue
        since_header += 1
        if label == "SONSTIGE":
            make = header if header not in _GROUP_BRANDS else header.split(",")[0]
            rows.append(
                ArchiveRow(row_number, label, make, None, vehicles, "source_model_aggregate")
            )
            continue
        make, model = split_label(header, label)
        flag = None if model is not None else "source_model_missing"
        rows.append(ArchiveRow(row_number, label, make, model, vehicles, flag))
    raise ValueError("KBA archive published total is missing")


def split_label(header: str, label: str) -> tuple[str, str | None]:
    """The make and model series a label names under its make header."""

    if label in _SPECIAL_LABELS:
        return _SPECIAL_LABELS[label]
    if header not in _GROUP_BRANDS:
        prefix = _PREFIXES.get(header, header)
        if not label.startswith(prefix + " "):
            raise ValueError(f"KBA archive label {label!r} does not name its make {header!r}")
        return header, label[len(prefix) + 1 :]
    for brand in _GROUP_BRANDS[header]:
        if not label.startswith(brand + " "):
            continue
        rest = label[len(brand) + 1 :]
        if brand != "GM":
            return brand, rest
        first, _, remainder = rest.partition(" ")
        if first in _GM_BRANDS and remainder:
            return first, remainder
        if first in _GM_CHEVROLETS:
            return "CHEVROLET", rest
        break
    raise ValueError(f"KBA archive label {label!r} names no brand of {header!r}")


class KBAFZ10ArchiveLoader:
    """Load one KBA December workbook per year as archived registrations."""

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        for release in releases:
            self._load_release(release, repository)

    def _load_release(self, release: StoredRelease, repository: SQLiteEvidenceRepository) -> None:
        manifest = release.manifest
        if manifest.parser_name != PARSER_NAME:
            raise ValueError("KBA archive release parser name is unsupported")
        if manifest.publication_status is not PublicationStatus.FINAL:
            raise ValueError("KBA archive release must be final")
        year = manifest.coverage_start.year
        if (manifest.coverage_start, manifest.coverage_end) != (
            date(year, 1, 1),
            date(year, 12, 31),
        ):
            raise ValueError("KBA archive release must cover one calendar year")
        table = read_workbook(release.artifact_path, year)
        observations = observations_for(release, table)
        if (table.raw_count, len(observations), table.rejected_count, 0) != (
            manifest.raw_record_count,
            manifest.accepted_record_count,
            manifest.rejected_record_count,
            manifest.quarantined_record_count,
        ):
            raise ValueError("KBA archive parser counts do not match manifest")
        repository.add_observations(observations)


def observations_for(release: StoredRelease, table: ArchiveTable) -> list[Observation]:
    year = table.year
    observations = []
    for row in table.rows:
        model = row.model if row.model is not None else "(not reported)"
        mapping_status = (
            MappingStatus.REJECTED
            if row.model is None or row.make == "SONSTIGE"
            else MappingStatus.UNRESOLVED
        )
        notes = [
            f"Read from the Jan.-Dezember {year} annual cumulative count, label {row.label!r}."
        ]
        if row.flag == "source_model_missing":
            notes.append("KBA published no model-series label for this row.")
        elif row.flag == "source_model_aggregate":
            notes.append("KBA grouped low-volume or unlisted model series as SONSTIGE.")
        observations.append(
            Observation(
                observation_id=stable_evidence_id(
                    "obs-kba-archive", release.release_id, str(row.row_number), row.make, model
                ),
                release_id=release.release_id,
                original_row_locator=f"{table.sheet_name}!C{row.row_number}",
                geography="DE",
                geography_version=release.manifest.geography_version,
                period_start=date(year, 1, 1),
                period_end=date(year, 12, 31),
                period_precision=PeriodPrecision.YEAR,
                measure=Measure.NEW_REGISTRATIONS,
                value=Decimal(row.vehicles),
                unit="vehicles",
                publication_status=PublicationStatus.FINAL,
                original_make=row.make,
                original_model=model,
                original_model_year=None,
                original_type="KBA FZ10 model series",
                source_make_identifier=row.make,
                source_model_identifier=stable_evidence_id("kba-model", row.make, model),
                normalized_make=normalize_vehicle_label(row.make),
                normalized_model=normalize_vehicle_label(model),
                normalized_model_year=None,
                canonical_vehicle_id=None,
                mapping_status=mapping_status,
                transformation_notes=tuple(notes),
                validation_flags=(
                    *((row.flag,) if row.flag else ()),
                    ARCHIVE_REGISTRATIONS_FLAG,
                    VOCABULARY_EXCLUDED_FLAG,
                ),
                evidence_confidence=_confidence(mapping_status),
            )
        )
    return observations


def _label(value: object) -> str | None:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text or None


def _vehicle_count(value: object) -> int:
    if value == "-":
        return 0
    if type(value) is not float or value < 0 or not value.is_integer():
        raise ValueError("KBA archive registration count is invalid")
    return int(value)


def _confidence(mapping_status: MappingStatus) -> EvidenceConfidence:
    identity = 0 if mapping_status is MappingStatus.REJECTED else 5
    return EvidenceConfidence(
        authority=25,
        publication_status=10,
        coverage=25,
        identity=identity,
        independent_agreement=10,
        reasons=(
            "Official finalized KBA Central Vehicle Register evidence.",
            "KBA model series is retained without an unreviewed cross-source alias.",
            "Archived years before the EEA's model-level monitoring; no overlap to compare.",
        ),
    )


__all__: Sequence[str] = (
    "PARSER_NAME",
    "YEARS",
    "ArchiveRow",
    "ArchiveTable",
    "KBAFZ10ArchiveLoader",
    "observations_for",
    "parse_rows",
    "read_sheet",
    "read_workbook",
    "split_label",
)

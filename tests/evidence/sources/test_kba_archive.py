from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from icor.domain.evidence import (
    ARCHIVE_REGISTRATIONS_FLAG,
    VOCABULARY_EXCLUDED_FLAG,
    MappingStatus,
    Measure,
    PublicationStatus,
    ReleaseManifest,
)
from icor.evidence.sources.kba_archive import (
    ArchiveRow,
    ArchiveTable,
    loaded_rows,
    observations_for,
    parse_rows,
    read_sheet,
    split_label,
)
from icor.infrastructure.release_store import StoredRelease


class _Sheet:
    """A legacy workbook sheet as xlrd presents it: rows of cell values."""

    def __init__(self, name: str, rows: list[list[object]]) -> None:
        self.name = name
        self.nrows = len(rows)
        self._rows = rows

    def cell_value(self, rowx: int, colx: int) -> object:
        row = self._rows[rowx]
        return row[colx] if colx < len(row) else ""


_DETAIL = [
    ["VW", "", ""],
    ["VW GOLF, JETTA", 30.0, 300.0],
    ["VW POLO", 10.0, 100.0],
    ["SONSTIGE", "-", 5.0],
    ["", "", ""],
    ["ZUSAMMEN", 40.0, 405.0],
    ["BENTLEY", 1.0, 7.0],
    ["TOYOTA, LEXUS", "", ""],
    ["TOYOTA YARIS", 5.0, 50.0],
    ["LEXUS IS", 1.0, 10.0],
    ["Zusammen", 6.0, 60.0],
    ["SONSTIGE HERSTELLER", 2.0, 20.0],
]


def _sheet_2004(total: float = 492.0) -> _Sheet:
    return _Sheet(
        "Dezember-2004",
        [
            ["Kraftfahrt-Bundesamt"],
            [""],
            ["N", "Vorab-Information über die Neuzulassungen"],
            ["Marke/\nModellreihe", "Insgesamt", ""],
            ["", "Dezember\n2004", "Jan./Dezember 2004"],
            ["", "", "Anzahl"],
            *_DETAIL,
            ["Neuzulassungen insgesamt", 50.0, total],
            ["1) Durch die Harmonisierung der Fz.-Papiere ..."],
        ],
    )


def _sheet_2010() -> _Sheet:
    return _Sheet(
        "Dezember 2010",
        [
            ["", " ", ""],
            ["Neuzulassungen von Personenkraftwagen im Dezember 2010 nach Marken"],
            ["Marke/\nModellreihe", "Insgesamt", ""],
            ["", "Dezember\n2010", "Jan.-Dezember 2010"],
            ["", "", "Anzahl"],
            ["", 1.0, 2.0],
            *_DETAIL,
            ["NEUZULASSUNGEN INSGESAMT", 50.0, 492.0],
        ],
    )


def test_the_annual_count_of_each_model_series_is_read_under_its_make() -> None:
    table = read_sheet(_sheet_2004(), 2004)

    assert [(row.make, row.model, row.vehicles) for row in table.rows] == [
        ("VW", "GOLF, JETTA", 300),
        ("VW", "POLO", 100),
        ("VW", None, 5),
        ("BENTLEY", None, 7),
        ("TOYOTA", "YARIS", 50),
        ("LEXUS", "IS", 10),
        ("SONSTIGE", None, 20),
    ]
    assert table.published_total == 492
    assert (table.raw_count, table.rejected_count) == (10, 3)


def test_the_later_layout_numbers_its_columns_and_reads_the_same() -> None:
    later = read_sheet(_sheet_2010(), 2010)

    assert later.published_total == 492
    assert [(row.make, row.model, row.vehicles) for row in later.rows] == [
        (row.make, row.model, row.vehicles) for row in read_sheet(_sheet_2004(), 2004).rows
    ]


def test_rows_that_miss_the_published_total_fail_closed() -> None:
    with pytest.raises(ValueError, match="published total"):
        read_sheet(_sheet_2004(total=493.0), 2004)


def test_a_make_subtotal_that_misses_its_rows_fails_closed() -> None:
    cells = [(1, "VW", ""), (2, "VW POLO", 10.0), (3, "ZUSAMMEN", 11.0)]

    with pytest.raises(ValueError, match="subtotal of VW"):
        parse_rows(cells)


def test_another_years_workbook_or_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="December issue"):
        read_sheet(_sheet_2004(), 2005)
    sheet = _sheet_2004()
    sheet._rows[4][2] = "Dezember 2004"
    with pytest.raises(ValueError, match="annual column"):
        read_sheet(sheet, 2004)


@pytest.mark.parametrize(
    ("header", "label", "expected"),
    [
        ("VW", "VW GOLF, JETTA", ("VW", "GOLF, JETTA")),
        ("ALFA ROMEO", "ALFA 147", ("ALFA ROMEO", "147")),
        ("BMW, MINI", "BMW 3ER", ("BMW", "3ER")),
        ("BMW, MINI", "ALPINA B 3", ("ALPINA", "B 3")),
        ("BMW, MINI", "BMW MINI", ("MINI", "MINI")),
        ("TOYOTA, LEXUS", "TOYOTA LEXUS", ("LEXUS", None)),
        ("CHRYSLER, JEEP", "DODGE CALIBER", ("DODGE", "CALIBER")),
        ("GENERAL MOTORS", "GM CADILLAC SRX", ("CADILLAC", "SRX")),
        ("GENERAL MOTORS", "GM HUMMER H3", ("HUMMER", "H3")),
        ("GENERAL MOTORS", "GM CORVETTE", ("CHEVROLET", "CORVETTE")),
        ("OPEL, GM ...", "CHEVROLET TAHOE", ("CHEVROLET", "TAHOE")),
        ("MG ROVER, AUSTIN", "MG ROVER 25, MG ZR", ("MG ROVER", "25, MG ZR")),
    ],
)
def test_a_label_names_its_make_and_model_series(header, label, expected) -> None:
    assert split_label(header, label) == expected


@pytest.mark.parametrize(
    ("header", "label"),
    [("VW", "AUDI A4"), ("BMW, MINI", "ROVER 75"), ("GENERAL MOTORS", "GM ASTRA"), ("VW", "VW")],
)
def test_a_label_that_names_no_make_of_its_header_fails_closed(header, label) -> None:
    with pytest.raises(ValueError, match="KBA archive label"):
        split_label(header, label)


def _release() -> StoredRelease:
    manifest = ReleaseManifest(
        release_id="kba-fz10-2004-12-final-v2",
        source_id="kba-fz10",
        publisher="Kraftfahrt-Bundesamt",
        source_url="https://www.kba.de/SharedDocs/Downloads/DE/Statistik/Fahrzeuge/FZ10/fz10_2004_12_xls.xls",
        retrieved_at=datetime(2026, 10, 22, tzinfo=UTC),
        published_at=datetime(2005, 1, 15, tzinfo=UTC),
        coverage_start=date(2004, 1, 1),
        coverage_end=date(2004, 12, 31),
        geography="DE",
        geography_version="DE-2004-v1",
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="kba-zfzr-archive",
        terms_url="https://www.govdata.de/dl-de/by-2-0",
        permitted_local_use="Official open-data reuse with attribution",
        artifact_path="artifact.xls",
        artifact_bytes=1,
        sha256="0" * 64,
        parser_name="kba_fz10_xls_v1",
        parser_version="v1",
        expected_schema="kba-fz10-archive-v1",
        raw_record_count=10,
        accepted_record_count=7,
        rejected_record_count=3,
        quarantined_record_count=0,
    )
    return StoredRelease(
        manifest.source_id,
        manifest.release_id,
        Path("artifact.xls"),
        Path("manifest.json"),
        manifest,
    )


def test_archived_observations_add_cohorts_but_never_teach_or_forecast() -> None:
    observations = observations_for(_release(), read_sheet(_sheet_2004(), 2004))

    assert all(
        {ARCHIVE_REGISTRATIONS_FLAG, VOCABULARY_EXCLUDED_FLAG} <= set(item.validation_flags)
        for item in observations
    )
    assert {item.period_start for item in observations} == {date(2004, 1, 1)}
    named = {
        (item.original_make, item.original_model)
        for item in observations
        if item.mapping_status is MappingStatus.UNRESOLVED
    }
    assert named == {("VW", "GOLF, JETTA"), ("VW", "POLO"), ("TOYOTA", "YARIS"), ("LEXUS", "IS")}
    assert sum(item.value for item in observations) == 492


def test_a_year_the_eea_covers_loads_only_the_makes_it_omits() -> None:
    rows = (
        ArchiveRow(7, "OPEL ASTRA", "OPEL", "ASTRA", 90),
        ArchiveRow(8, "SONSTIGE", "OPEL", None, 1, "source_model_aggregate"),
        ArchiveRow(9, "VW GOLF, JETTA", "VW", "GOLF, JETTA", 250),
        ArchiveRow(10, "HONDA CIVIC", "HONDA", "CIVIC", 9),
    )
    table = ArchiveTable(2011, "Dezember 2011", rows, 6, 2, 350)

    assert [row.row_number for row in loaded_rows(table)] == [7, 8, 10]
    assert loaded_rows(ArchiveTable(2009, "Dezember-2009", rows, 6, 2, 350)) == rows
    with pytest.raises(ValueError, match="neither archived nor supplementary"):
        loaded_rows(ArchiveTable(2010, "Dezember 2010", rows, 6, 2, 350))

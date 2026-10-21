from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from icor.evidence.lt_fleet_acquisition import (
    ARCHIVE_URL,
    EUROSTAT_STOCK_2025,
    SOURCE_HEADER,
    USER_AGENT,
    Download,
    _AllowlistedRedirects,
    acquire_fleet,
    destination_for,
    download_archive,
    profile_path,
    snapshot_date,
    variant_path,
)
from icor.evidence.register_aggregate import SCHEMA

MEMBER = "Atviri_TP_parko_duomenys.csv"
STAMP = (2026, 10, 1, 2, 31, 30)
# Fictional values in the columns that must never leave the register.
PERSONAL = {
    "VALD_TIPAS": "FIZINIS-ASMUO-QX7",
    "VALD_GIM_DAT_INT": "1971-GIMIMO-ZQ",
    "SAVIVALDYBE": "Fiktyvi-Savivaldybe-77",
    "APSKRITIS": "Fiktyvi-Apskritis-88",
    "ES_TIPO_PATVIRTINIMO_NR": "e9*2099/999*0001-FICT",
    "NAC_TIPO_PATVIRTINIMO_NR": "NAC-FICT-4242",
    "SPALVA": "VIOLETINE-FICT",
    "KILMES_SALIS": "ZZQ",
}


def _row(**values: str) -> list[str]:
    fields = dict.fromkeys(SOURCE_HEADER, "")
    fields.update(
        {
            "KATEGORIJA_KLASE": "M1",
            "KATEGORIJA_PILNAI": "M1-FICT",
            "DAE_STATUSAS": "T",
            "PIRM_REG_DATA": "2015-03-04",
            "PIRM_REG_DATA_LT": "2015-03-04",
            **PERSONAL,
        }
    )
    fields.update(values)
    return [fields[name] for name in SOURCE_HEADER]


def _csv_bytes(rows: list[list[str]], header: tuple[str, ...] = SOURCE_HEADER) -> bytes:
    from io import StringIO

    buffer = StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8")


def _archive(
    directory: Path,
    rows: list[list[str]] | None = None,
    *,
    header: tuple[str, ...] = SOURCE_HEADER,
    payload: bytes | None = None,
    member: str = MEMBER,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "register.zip"
    info = zipfile.ZipInfo(member, date_time=STAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(info, payload if payload is not None else _csv_bytes(rows or [], header))
    return path


def _rows(path: Path) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [tuple(row) for row in csv.reader(stream)]


def _fleet() -> list[list[str]]:
    return [
        _row(MARKE="AURORA", KOMERCINIS_PAV="NOVA  1", PIRM_REG_DATA_LT="2018-06-01"),
        _row(
            MARKE="AURORA",
            KOMERCINIS_PAV="NOVA 1",
            PIRM_REG_DATA="2015-09-09",
            PIRM_REG_DATA_LT="2015-09-10",
        ),
        _row(
            MARKE="NORTHSTAR",
            KOMERCINIS_PAV="SKY",
            PIRM_REG_DATA="",
            PIRM_REG_DATA_LT="2019-05-05",
        ),
        _row(
            MARKE="NORTHSTAR",
            KOMERCINIS_PAV="SKY",
            DAE_STATUSAS="N",
            PIRM_REG_DATA="2010-01-01",
            PIRM_REG_DATA_LT="2010-02-02",
        ),
        _row(MARKE="AURORA", KOMERCINIS_PAV="VAN", KATEGORIJA_KLASE="N1", KATEGORIJA_PILNAI="N1-F"),
        # The class says N1 while the full category says M1: out, and a disagreement.
        _row(MARKE="AURORA", KOMERCINIS_PAV="ODD", KATEGORIJA_KLASE="N1", KATEGORIJA_PILNAI="M1-F"),
        # The class says M1 while the full category says N1: in, and a disagreement.
        _row(
            MARKE="AURORA",
            KOMERCINIS_PAV="ODD2",
            KATEGORIJA_PILNAI="N1-F",
            PIRM_REG_DATA="2012-01-01",
        ),
        _row(MARKE="NORTHSTAR", KOMERCINIS_PAV="SKY", DAE_STATUSAS="X"),
        _row(
            MARKE="AURORA",
            KOMERCINIS_PAV="OFFROAD",
            KATEGORIJA_PILNAI="M1G-F",
            PIRM_REG_DATA="2020-01-01",
        ),
        _row(MARKE="AURORA", KOMERCINIS_PAV="BADDATE", PIRM_REG_DATA="03/04/2011"),
        _row(MARKE="AURORA", KOMERCINIS_PAV="MOTO", KATEGORIJA_KLASE="L3", KATEGORIJA_PILNAI="L3e"),
    ]


def _acquire(directory: Path, rows: list[list[str]] | None = None, **options):
    archive = _archive(directory / "in", _fleet() if rows is None else rows, **options)
    return acquire_fleet(directory / "out" / "lt-registered-cars-20261001.csv", archive)


def test_keeps_cars_of_class_m1_in_traffic_by_make_model_and_cohort(tmp_path: Path) -> None:
    result = _acquire(tmp_path)

    assert _rows(result.path) == [
        SCHEMA,
        ("AURORA", "BADDATE", "", "1"),
        ("AURORA", "NOVA 1", "2015", "2"),
        ("AURORA", "ODD2", "2012", "1"),
        ("AURORA", "OFFROAD", "2020", "1"),
        ("NORTHSTAR", "SKY", "", "1"),
    ]
    assert result.source_rows == 11
    assert result.category_rows == 8
    assert result.status_rows == 6
    assert result.category_disagreements == 2
    assert (result.group_count, result.vehicles) == (5, 6)
    assert result.blank_cohort_vehicles == 2
    assert result.blank_model_vehicles == 0
    assert result.artifact_bytes == result.path.stat().st_size
    assert result.sha256 == hashlib.sha256(result.path.read_bytes()).hexdigest()
    assert result.snapshot_date == "2026-10-01"


def test_the_variant_also_counts_registered_cars_not_in_traffic(tmp_path: Path) -> None:
    result = _acquire(tmp_path)

    variant = result.variant
    assert variant.path.name == "lt-registered-cars-20261001-with-not-in-traffic.csv"
    assert _rows(variant.path) == [
        SCHEMA,
        ("AURORA", "BADDATE", "", "1"),
        ("AURORA", "NOVA 1", "2015", "2"),
        ("AURORA", "ODD2", "2012", "1"),
        ("AURORA", "OFFROAD", "2020", "1"),
        ("NORTHSTAR", "SKY", "2010", "1"),
        ("NORTHSTAR", "SKY", "", "1"),
    ]
    assert (variant.status_rows, variant.group_count, variant.vehicles) == (7, 6, 7)
    assert variant.sha256 == hashlib.sha256(variant.path.read_bytes()).hexdigest()
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert profile["vehicles_by_status_within_category"] == {"N": 1, "T": 6, "X": 1}
    assert profile["funnel"]["variant_status_rows"] == 7
    assert profile["reconciliation"]["eurostat_lt_2025_stock"] == EUROSTAT_STOCK_2025


def test_cohort_is_the_first_registration_anywhere_not_the_national_date(tmp_path: Path) -> None:
    result = _acquire(tmp_path)

    cohorts = {(make, model): cohort for make, model, cohort, _ in _rows(result.path)[1:]}
    # Registered in Lithuania in 2018, first registered anywhere in 2015.
    assert ("AURORA", "NOVA 1") in cohorts
    assert cohorts[("AURORA", "NOVA 1")] == "2015"
    assert all(cohort != "2018" for cohort in cohorts.values())
    # No world date: the cohort stays blank, never the 2019 national date.
    assert cohorts[("NORTHSTAR", "SKY")] == ""
    assert all(cohort != "2019" for cohort in cohorts.values())


def test_the_profile_measures_the_national_date_and_the_blank_cohorts(tmp_path: Path) -> None:
    result = _acquire(tmp_path)

    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    comparison = profile["national_year_comparison_strict_2000_plus"]
    # Lithuanian dates in 2015: the second NOVA 1 plus ODD2, OFFROAD and BADDATE, whose
    # fixture default is 2015-03-04; the first NOVA 1 was registered in Lithuania in 2018.
    assert comparison["2015"] == {
        "by_cohort": 2,
        "by_national_year": 4,
        "cohort_registered_in_lt_same_year": 1,
        "cohort_registered_in_lt_later": 1,
        "cohort_without_lt_date": 0,
    }
    assert comparison["2018"]["by_national_year"] == 1
    assert profile["cohort_by_national_year_strict_2000_plus"]["2015"] == {"2015": 1, "2018": 1}
    assert profile["vehicles_by_cohort_strict"] == {
        "2012": 1,
        "2015": 2,
        "2020": 1,
        "blank": 2,
    }
    assert profile["vehicles_by_cohort_and_status_2000_plus"]["2010"] == {"N": 1}
    assert profile["national_year_of_vehicles_without_first_registration_date_by_status"] == {
        "T": {"2015": 1, "2019": 1},
    }
    assert profile["unparseable_date_values"] == 1
    # OFFROAD: first registered in 2020 anywhere, yet 2015 in Lithuania (fixture default).
    assert profile["strict_vehicles_with_national_year_before_cohort"] == 1
    assert profile["funnel"]["category_cross_check"] == {
        "class_m1_but_full_category_not_m1": 1,
        "full_category_m1_but_class_not_m1": 1,
    }
    assert profile["top_makes_strict"] == [
        {"make": "AURORA", "vehicles": 5},
        {"make": "NORTHSTAR", "vehicles": 1},
    ]


def test_masked_and_blank_labels_stay_as_buckets_and_are_counted(tmp_path: Path) -> None:
    rows = [
        _row(MARKE="Nuasmeninta", KOMERCINIS_PAV="Nuasmeninta"),
        _row(MARKE="Nuasmeninta", KOMERCINIS_PAV="Nuasmeninta", DAE_STATUSAS="N"),
        _row(MARKE="AURORA", KOMERCINIS_PAV=" ", PIRM_REG_DATA=""),
        _row(MARKE="AURORA", KOMERCINIS_PAV="NOVA"),
    ]

    result = _acquire(tmp_path, rows)

    assert ("Nuasmeninta", "Nuasmeninta", "2015", "1") in _rows(result.path)
    assert ("AURORA", "", "", "1") in _rows(result.path)
    assert (result.masked_vehicles, result.blank_model_vehicles) == (1, 1)
    assert result.blank_cohort_vehicles == 1
    assert result.variant.masked_vehicles == 2
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert profile["masked"] == {
        "label": "Nuasmeninta",
        "strict_vehicles": 1,
        "with_not_in_traffic_vehicles": 2,
    }


def test_no_personal_or_identifying_column_reaches_any_output(tmp_path: Path) -> None:
    result = _acquire(tmp_path)

    for path in (result.path, result.variant.path, result.profile_path):
        data = path.read_bytes()
        for name, value in PERSONAL.items():
            assert value.encode() not in data, (path.name, name)
            assert name.encode() not in data, (path.name, name)
    assert result.path.read_text(encoding="utf-8").splitlines()[0] == ",".join(SCHEMA)


def test_output_is_deterministic_regardless_of_row_order(tmp_path: Path) -> None:
    first = _acquire(tmp_path / "a", _fleet())
    second = _acquire(tmp_path / "b", _fleet()[::-1])

    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.variant.path.read_bytes() == second.variant.path.read_bytes()
    assert first.sha256 == second.sha256


def test_the_profile_records_the_source_and_its_download(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "in", _fleet())
    data = archive.read_bytes()
    download = Download(
        ARCHIVE_URL,
        "Wed, 21 Oct 2026 05:27:29 GMT",
        "2026-10-21T15:00:00Z",
        len(data),
        hashlib.sha256(data).hexdigest(),
    )

    result = acquire_fleet(tmp_path / "out" / "lt.csv", archive, download=download)

    profile = json.loads(profile_path(result.path).read_text(encoding="utf-8"))
    assert profile_path(result.path).name == "lt.csv.profile.json"
    assert profile["source"] == {
        "name": "Regitra, Atviri TP parko duomenys",
        "dataset": "https://data.gov.lt/datasets/1211",
        "licence": "CC BY 4.0",
        "url": ARCHIVE_URL,
        "http_last_modified": "Wed, 21 Oct 2026 05:27:29 GMT",
        "retrieved_at_utc": "2026-10-21T15:00:00Z",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "member": MEMBER,
        "snapshot_date": "2026-10-01",
    }
    assert (result.source_bytes, result.source_sha256) == (
        len(data),
        hashlib.sha256(data).hexdigest(),
    )
    assert profile["aggregates"]["strict"]["sha256"] == result.sha256
    assert profile["aggregates"]["with_not_in_traffic"]["sha256"] == result.variant.sha256


def test_an_archive_that_differs_from_its_recorded_download_is_refused(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "in", _fleet())
    download = Download(ARCHIVE_URL, "", "2026-10-21T15:00:00Z", archive.stat().st_size, "0" * 64)

    with pytest.raises(ValueError, match="recorded download"):
        acquire_fleet(tmp_path / "out" / "lt.csv", archive, download=download)
    assert not (tmp_path / "out").exists()


def test_a_register_without_passenger_cars_writes_nothing(tmp_path: Path) -> None:
    rows = [_row(KATEGORIJA_KLASE="N1", KATEGORIJA_PILNAI="N1-F")]

    with pytest.raises(ValueError, match="no passenger cars"):
        _acquire(tmp_path, rows)
    assert not (tmp_path / "out").exists()


def test_schema_drift_and_malformed_input_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        _acquire(tmp_path / "short", [], header=SOURCE_HEADER[:-1])
    swapped = (SOURCE_HEADER[1], SOURCE_HEADER[0], *SOURCE_HEADER[2:])
    with pytest.raises(ValueError, match="schema"):
        _acquire(tmp_path / "swapped", [_row()], header=swapped)
    with pytest.raises(ValueError, match="schema"):
        _acquire(tmp_path / "empty", payload=b"")

    short_row = _row()[:-1]
    with pytest.raises(ValueError, match="malformed"):
        _acquire(tmp_path / "row", [short_row])
    with pytest.raises(ValueError, match="not UTF-8"):
        _acquire(tmp_path / "latin", payload=_csv_bytes([_row()]) + b'"\xe9"\n')
    with pytest.raises(ValueError, match="members"):
        _acquire(tmp_path / "member", [_row()], member="other.csv")
    for name in ("short", "swapped", "empty", "row", "latin", "member"):
        assert not (tmp_path / name / "out").exists()


def test_a_member_larger_than_its_ceiling_is_refused(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "in", _fleet())

    with pytest.raises(ValueError, match="size limit"):
        acquire_fleet(tmp_path / "out" / "lt.csv", archive, max_member_bytes=10)


def test_the_snapshot_date_is_the_members_date_stamp(tmp_path: Path) -> None:
    archive = _archive(tmp_path, _fleet())

    snapshot = snapshot_date(archive)

    assert snapshot == date(2026, 10, 1)
    template = Path("pins/lt-registered-cars-{date}.csv")
    assert destination_for(template, snapshot) == Path("pins/lt-registered-cars-20261001.csv")
    assert variant_path(destination_for(template, snapshot)) == Path(
        "pins/lt-registered-cars-20261001-with-not-in-traffic.csv"
    )


class _Response:
    def __init__(
        self, payload: bytes, url: str = ARCHIVE_URL, headers: dict[str, str] | None = None
    ):
        self._payload = payload
        self._url = url
        self.headers = {"Last-Modified": "Wed, 21 Oct 2026 05:27:29 GMT"} | (headers or {})

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk

    def geturl(self) -> str:
        return self._url


def test_download_sends_a_browser_like_agent_and_records_bytes_and_sha256(tmp_path: Path) -> None:
    seen: list[urllib.request.Request] = []

    def opener(request: urllib.request.Request, timeout: float) -> _Response:
        seen.append(request)
        return _Response(b"abc")

    record = download_archive(
        tmp_path / "dl" / "a.zip",
        opener=opener,
        clock=lambda: datetime(2026, 10, 21, 15, 0, 0, tzinfo=UTC),
    )

    assert (tmp_path / "dl" / "a.zip").read_bytes() == b"abc"
    assert record == Download(
        ARCHIVE_URL,
        "Wed, 21 Oct 2026 05:27:29 GMT",
        "2026-10-21T15:00:00Z",
        3,
        hashlib.sha256(b"abc").hexdigest(),
    )
    assert seen[0].get_header("User-agent") == USER_AGENT
    assert USER_AGENT.startswith("Mozilla/5.0")
    assert seen[0].full_url == ARCHIVE_URL


def test_download_stays_on_the_allowlisted_origin_and_byte_ceiling(tmp_path: Path) -> None:
    def opener(payload: bytes, url: str = ARCHIVE_URL):
        return lambda request, timeout: _Response(payload, url)

    with pytest.raises(ValueError, match="byte limit"):
        download_archive(tmp_path / "b.zip", opener=opener(b"abcd"), max_bytes=3)
    with pytest.raises(ValueError, match="allowlisted"):
        download_archive(tmp_path / "c.zip", opener=opener(b"abc", "https://example.org/x.zip"))
    with pytest.raises(ValueError, match="allowlisted"):
        download_archive(tmp_path / "d.zip", url="http://www.regitra.lt/x.zip")
    with pytest.raises(ValueError, match="allowlisted"):
        download_archive(tmp_path / "e.zip", url="https://regitra.lt/x.zip")
    with pytest.raises(ValueError, match="allowlisted"):
        download_archive(tmp_path / "f.zip", url="https://www.regitra.lt.evil.example/x.zip")
    assert list(tmp_path.iterdir()) == []


def test_a_redirect_off_the_allowlist_is_refused_before_it_is_followed() -> None:
    handler = _AllowlistedRedirects()
    request = urllib.request.Request(ARCHIVE_URL)

    with pytest.raises(ValueError, match="allowlisted"):
        handler.redirect_request(request, None, 302, "Found", {}, "https://example.org/x.zip")
    with pytest.raises(ValueError, match="allowlisted"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://www.regitra.lt/x.zip")
    followed = handler.redirect_request(
        request, None, 302, "Found", {}, "https://www.regitra.lt/other.zip"
    )
    assert followed is not None
    assert followed.full_url == "https://www.regitra.lt/other.zip"

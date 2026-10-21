from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pytest

from icor.evidence.ee_fleet_acquisition import (
    METADATA_URL,
    SOURCE_HEADER,
    SOURCE_URL,
    Distribution,
    _SameOriginRedirects,
    acquire_fleet,
    latest_distribution,
    snapshot_date_of,
)
from icor.evidence.register_aggregate import SCHEMA

NOW = datetime(2026, 10, 21, 12, 30, 5, tzinfo=UTC)
USED = {"MARK", "MUDEL", "YLDINE_STAATUS", "STAATUS", "ESMANE_REG", "EESTIS_ESMAREG", "KATEGOORIA"}


def _row(**values: str) -> list[str]:
    # Every column the aggregate does not use carries a marker, so a leak is detectable.
    fields = {name: f"SENTINEL-{name}" for name in SOURCE_HEADER if name not in USED}
    fields.update(
        {
            "KATEGOORIA": "M1",
            "YLDINE_STAATUS": "REGISTREERITUD",
            "STAATUS": "REGISTRIS_OLEV_SOIDUK",
            "ESMANE_REG": "2010-05-04",
            "EESTIS_ESMAREG": "2010-05-04",
            "VKOM_MAAKOND": "Fictionalmaa maakond",
            "VKOM_TYYP": "ERAISIK-FICTIONAL",
        }
    )
    fields.update(values)
    return [fields[name] for name in SOURCE_HEADER]


def _csv(rows: list[list[str]], *, header: list[str] = SOURCE_HEADER, bom: bool = True) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter=";", lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    return (b"\xef\xbb\xbf" if bom else b"") + buffer.getvalue().encode("utf-8")


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, url: str, headers: dict[str, str]) -> None:
        super().__init__(payload)
        self._url = url
        self.headers = headers

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def geturl(self) -> str:
        return self._url


class _Opener:
    def __init__(
        self,
        payload: bytes,
        *,
        final_url: str = SOURCE_URL,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.final_url = final_url
        self.headers = {"Content-Length": str(len(payload))} if headers is None else headers
        self.requests: list[str] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> _Response:
        self.requests.append(request.full_url)
        return _Response(self.payload, self.final_url, self.headers)


def _acquire(tmp_path: Path, rows: list[list[str]], **options):
    return acquire_fleet(
        tmp_path / "out" / "ee.csv",
        opener=_Opener(_csv(rows)),
        clock=lambda: NOW,
        **options,
    )


def _read(path: Path) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [tuple(row) for row in csv.reader(stream)]


def test_keeps_registered_passenger_cars_and_counts_the_variant_with_suspended(
    tmp_path: Path,
) -> None:
    rows = [
        _row(MARK="AURORA", MUDEL="Nova  GT", ESMANE_REG="2004-03-02"),
        _row(MARK="AURORA", MUDEL="Nova GT", ESMANE_REG="2004-11-30"),
        _row(MARK="NORTHSTAR", MUDEL="Wagon", KATEGOORIA="M1G", ESMANE_REG="2012-01-01"),
        _row(MARK="AURORA", MUDEL="Pickup", KATEGOORIA="N1", ESMANE_REG="2004-03-02"),
        _row(MARK="AURORA", MUDEL="Moped", KATEGOORIA="L1e", ESMANE_REG="2004-03-02"),
        _row(
            MARK="NORTHSTAR",
            MUDEL="Wagon",
            YLDINE_STAATUS="PEATATUD",
            STAATUS="KANNE_PEATATUD",
            ESMANE_REG="2012-06-01",
        ),
        _row(
            MARK="AURORA",
            MUDEL="Gone",
            YLDINE_STAATUS="KUSTUTATUD",
            STAATUS="KUSTUTATUD",
            ESMANE_REG="2004-03-02",
        ),
    ]

    result = _acquire(tmp_path, rows)

    assert _read(result.path) == [
        SCHEMA,
        ("AURORA", "Nova GT", "2004", "2"),
        ("NORTHSTAR", "Wagon", "2012", "1"),
    ]
    assert _read(result.variant_path) == [
        SCHEMA,
        ("AURORA", "Nova GT", "2004", "2"),
        ("NORTHSTAR", "Wagon", "2012", "2"),
    ]
    assert result.variant_path.name == "ee-with-suspended.csv"
    assert (result.source_rows, result.category_rows) == (7, 5)
    assert (result.status_rows, result.variant_status_rows) == (3, 4)
    assert (result.group_count, result.vehicles) == (2, 3)
    assert (result.variant_group_count, result.variant_vehicles) == (2, 4)
    assert result.artifact_bytes == result.path.stat().st_size
    assert result.sha256 == hashlib.sha256(result.path.read_bytes()).hexdigest()
    assert result.variant_sha256 == hashlib.sha256(result.variant_path.read_bytes()).hexdigest()


def test_the_cohort_is_the_first_registration_anywhere_not_the_national_date(
    tmp_path: Path,
) -> None:
    rows = [
        _row(MARK="AURORA", MUDEL="Import", ESMANE_REG="2003-06-01", EESTIS_ESMAREG="2019-02-03"),
        _row(MARK="AURORA", MUDEL="Import", ESMANE_REG="2003-09-09", EESTIS_ESMAREG=""),
    ]

    result = _acquire(tmp_path, rows)

    assert _read(result.path)[1:] == [("AURORA", "Import", "2003", "2")]


def test_unusable_years_and_blank_or_masked_labels_stay_in_the_aggregate(tmp_path: Path) -> None:
    rows = [
        _row(MARK="AURORA", MUDEL="", ESMANE_REG="2015-01-01"),
        _row(MARK="AURORA", MUDEL="-", ESMANE_REG="2015-01-01"),
        _row(MARK="", MUDEL="Orphan", ESMANE_REG="2015-01-01"),
        _row(MARK="AURORA", MUDEL="Undated", ESMANE_REG=""),
        _row(MARK="AURORA", MUDEL="Undated", ESMANE_REG="15.01.2015"),
        _row(MARK="AURORA", MUDEL="Fine", ESMANE_REG="2015-01-01"),
    ]

    result = _acquire(tmp_path, rows)

    assert _read(result.path)[1:] == [
        ("", "Orphan", "2015", "1"),
        ("AURORA", "", "2015", "1"),
        ("AURORA", "-", "2015", "1"),
        ("AURORA", "Fine", "2015", "1"),
        ("AURORA", "Undated", "", "2"),
    ]
    assert result.masked_vehicles == 1
    assert result.blank_model_vehicles == 1
    assert result.blank_cohort_vehicles == 2
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert profile["aggregate"]["blank_make_vehicles"] == 1
    assert profile["unparseable_first_registration_vehicles"] == 1


def test_no_unused_column_reaches_the_outputs(tmp_path: Path) -> None:
    rows = [
        _row(MARK="AURORA", MUDEL="Nova", YLDINE_STAATUS=status)
        for status in (
            "REGISTREERITUD",
            "PEATATUD",
        )
    ]

    result = _acquire(tmp_path, rows)

    for path in (result.path, result.variant_path, result.profile_path):
        text = path.read_bytes().decode("utf-8")
        assert "SENTINEL" not in text
        assert "Fictionalmaa" not in text
        assert "ERAISIK" not in text
    assert result.path.read_text(encoding="utf-8").splitlines()[0] == ",".join(SCHEMA)
    assert {path.name for path in (tmp_path / "out").iterdir()} == {
        "ee.csv",
        "ee-with-suspended.csv",
        "ee.csv.profile.json",
    }


def test_the_profile_holds_counts_only_and_the_source_identity(tmp_path: Path) -> None:
    payload = _csv(
        [
            _row(MARK="AURORA", MUDEL="Nova", ESMANE_REG="1998-01-01", EESTIS_ESMAREG="1998-01-01"),
            _row(MARK="AURORA", MUDEL="Nova", ESMANE_REG="2005-01-01", EESTIS_ESMAREG="2005-02-01"),
            _row(MARK="AURORA", MUDEL="Nova", ESMANE_REG="2005-01-01", EESTIS_ESMAREG="2011-02-01"),
            _row(MARK="NORTHSTAR", MUDEL="Wagon", ESMANE_REG="2005-01-01", EESTIS_ESMAREG=""),
            _row(
                MARK="NORTHSTAR",
                MUDEL="Wagon",
                YLDINE_STAATUS="PEATATUD",
                STAATUS="KANNE_PEATATUD",
                ESMANE_REG="2005-01-01",
            ),
            _row(
                MARK="NORTHSTAR",
                MUDEL="Wagon",
                STAATUS="VOORANDATUD",
                ESMANE_REG="2007-01-01",
                EESTIS_ESMAREG="2007-01-01",
            ),
        ]
    )
    opener = _Opener(payload, headers={"Content-Length": str(len(payload)), "Last-Modified": "x"})

    result = acquire_fleet(tmp_path / "ee.csv", opener=opener, clock=lambda: NOW)

    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert opener.requests == [SOURCE_URL]
    assert profile["source"] == {
        "dataset_id": "9c5ef13d-c838-4998-acdd-e73a5da1721b",
        "licence": "CC BY-SA 4.0",
        "url": SOURCE_URL,
        "snapshot_date": "2026-10-01",
        "last_modified": "x",
        "retrieved_at": "2026-10-21T12:30:05Z",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert (result.source_bytes, result.source_sha256) == (
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    assert profile["funnel"] == {
        "source_rows": 6,
        "category_rows": 6,
        "registered_rows": 5,
        "registered_or_suspended_rows": 6,
    }
    assert profile["vehicles_by_general_status"] == {"REGISTREERITUD": 5, "PEATATUD": 1}
    assert profile["vehicles_by_status"] == {
        "REGISTRIS_OLEV_SOIDUK": 4,
        "KANNE_PEATATUD": 1,
        "VOORANDATUD": 1,
    }
    assert profile["vehicles_by_status_within_registered"] == {
        "REGISTRIS_OLEV_SOIDUK": 4,
        "VOORANDATUD": 1,
    }
    assert profile["registered_vehicles_by_cohort"] == {"1998": 1, "2005": 3, "2007": 1}
    assert profile["vehicles_by_cohort_and_status_2000_plus"] == {
        "2005": {
            "REGISTREERITUD / REGISTRIS_OLEV_SOIDUK": 3,
            "PEATATUD / KANNE_PEATATUD": 1,
        },
        "2007": {"REGISTREERITUD / VOORANDATUD": 1},
    }
    assert profile["registered_vehicles_by_national_first_registration_year_cohorts_2000_plus"] == {
        "2005": 1,
        "2007": 1,
        "2011": 1,
        "(blank)": 1,
    }
    assert profile["registered_cohorts_2000_plus_by_national_registration_vs_cohort"] == {
        "2005": {"same_year": 1, "later_year": 1, "national_blank": 1},
        "2007": {"same_year": 1},
    }
    assert profile["top_makes"] == [["AURORA", 3], ["NORTHSTAR", 2]]
    assert profile["aggregate"]["vehicles"] == 5
    assert profile["aggregate_with_suspended"]["vehicles"] == 6
    assert profile["aggregate"]["sha256"] == result.sha256
    assert profile["aggregate_with_suspended"]["sha256"] == result.variant_sha256


def test_the_profile_lists_at_most_the_forty_largest_makes(tmp_path: Path) -> None:
    rows = [_row(MARK=f"MAKE{number:02d}", MUDEL="X") for number in range(45)]
    rows += [_row(MARK="MAKE44", MUDEL="X")]

    result = _acquire(tmp_path, rows)

    makes = json.loads(result.profile_path.read_text(encoding="utf-8"))["top_makes"]
    assert len(makes) == 40
    assert makes[0] == ["MAKE44", 2]
    assert makes[1] == ["MAKE00", 1]


def test_output_is_deterministic_regardless_of_row_order(tmp_path: Path) -> None:
    rows = [
        _row(MARK="B", MUDEL="X", ESMANE_REG="2001-01-01"),
        _row(MARK="A", MUDEL="Y", ESMANE_REG="2002-01-01"),
        _row(MARK="A", MUDEL="Y", ESMANE_REG="2002-02-02", YLDINE_STAATUS="PEATATUD"),
    ]
    first = _acquire(tmp_path / "a", rows)
    second = _acquire(tmp_path / "b", rows[::-1])

    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.variant_path.read_bytes() == second.variant_path.read_bytes()
    profiles = [
        json.loads(item.profile_path.read_text(encoding="utf-8")) for item in (first, second)
    ]
    for profile in profiles:
        del profile["source"]  # the row order changes the downloaded bytes, not the counts
    assert profiles[0] == profiles[1]


def test_a_byte_order_mark_is_optional(tmp_path: Path) -> None:
    payload = _csv([_row(MARK="AURORA", MUDEL="Nova")], bom=False)

    result = acquire_fleet(tmp_path / "ee.csv", opener=_Opener(payload), clock=lambda: NOW)

    assert _read(result.path)[1:] == [("AURORA", "Nova", "2010", "1")]


def test_only_the_allowlisted_origin_is_fetched(tmp_path: Path) -> None:
    opener = _Opener(_csv([_row(MARK="A", MUDEL="B")]))
    for url in (
        SOURCE_URL.replace("pilv.transpordiamet.ee", "example.org"),
        SOURCE_URL.replace("https://", "http://"),
        SOURCE_URL.replace("pilv.transpordiamet.ee", "pilv.transpordiamet.ee.example.org"),
    ):
        with pytest.raises(ValueError, match="allowlisted"):
            acquire_fleet(tmp_path / "ee.csv", url=url, opener=opener)
    assert opener.requests == []
    assert list(tmp_path.iterdir()) == []


def test_a_redirect_off_the_allowlist_is_refused(tmp_path: Path) -> None:
    opener = _Opener(_csv([_row(MARK="A", MUDEL="B")]), final_url="https://s3.example.org/ar.csv")

    with pytest.raises(ValueError, match="allowlisted"):
        acquire_fleet(tmp_path / "ee.csv", opener=opener)
    assert list(tmp_path.iterdir()) == []

    handler = _SameOriginRedirects("pilv.transpordiamet.ee")
    request = urllib.request.Request(SOURCE_URL)
    with pytest.raises(ValueError, match="allowlisted"):
        handler.redirect_request(
            request, io.BytesIO(), 302, "Found", {}, "https://elsewhere.example.org/ar.csv"
        )
    follow = handler.redirect_request(
        request, io.BytesIO(), 302, "Found", {}, SOURCE_URL.replace("Iiee", "Other")
    )
    assert follow is not None


def test_a_download_over_the_byte_limit_writes_nothing(tmp_path: Path) -> None:
    payload = _csv([_row(MARK="AURORA", MUDEL="Nova") for _ in range(20)])

    with pytest.raises(ValueError, match="byte limit"):
        acquire_fleet(
            tmp_path / "out" / "ee.csv", opener=_Opener(payload), max_bytes=len(payload) - 1
        )
    assert not (tmp_path / "out").exists()

    # Exactly at the limit is accepted.
    result = acquire_fleet(
        tmp_path / "out" / "ee.csv", opener=_Opener(payload), max_bytes=len(payload)
    )
    assert result.vehicles == 20


def test_a_truncated_download_writes_nothing(tmp_path: Path) -> None:
    payload = _csv([_row(MARK="AURORA", MUDEL="Nova")])
    opener = _Opener(payload, headers={"Content-Length": str(len(payload) + 100)})

    with pytest.raises(ValueError, match="truncated"):
        acquire_fleet(tmp_path / "out" / "ee.csv", opener=opener)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "header",
    [
        SOURCE_HEADER[:-1],
        [*SOURCE_HEADER, "OMANIK"],
        [SOURCE_HEADER[1], SOURCE_HEADER[0], *SOURCE_HEADER[2:]],
        [name.lower() for name in SOURCE_HEADER],
        [],
    ],
)
def test_schema_drift_is_refused(tmp_path: Path, header: list[str]) -> None:
    payload = _csv([], header=header)

    with pytest.raises(ValueError, match="schema"):
        acquire_fleet(tmp_path / "out" / "ee.csv", opener=_Opener(payload))
    assert not (tmp_path / "out").exists()


def test_a_malformed_row_or_encoding_is_refused_without_echoing_the_row(tmp_path: Path) -> None:
    short = _row(MARK="AURORA", MUDEL="Nova")[:-3]
    with pytest.raises(ValueError, match="malformed") as raised:
        _acquire(tmp_path / "a", [_row(MARK="AURORA", MUDEL="Nova"), short])
    assert "SENTINEL" not in str(raised.value)
    assert not (tmp_path / "a" / "out").exists()

    unquoted = _csv([_row(MARK="AURORA", MUDEL="Nova")]).replace(b"AURORA", b'"AURORA')
    with pytest.raises(ValueError, match="malformed"):
        acquire_fleet(tmp_path / "b" / "ee.csv", opener=_Opener(unquoted))

    latin = _csv([_row(MARK="AURORA", MUDEL="Nova")]).replace(b"Nova", b"N\xf5va")
    with pytest.raises(ValueError, match="UTF-8"):
        acquire_fleet(tmp_path / "c" / "ee.csv", opener=_Opener(latin))
    assert not (tmp_path / "c").exists()


def test_a_register_without_registered_passenger_cars_is_refused(tmp_path: Path) -> None:
    rows = [
        _row(MARK="AURORA", MUDEL="Nova", KATEGOORIA="N1"),
        _row(MARK="AURORA", MUDEL="Nova", YLDINE_STAATUS="PEATATUD"),
    ]

    with pytest.raises(ValueError, match="no passenger cars"):
        _acquire(tmp_path, rows)
    assert not (tmp_path / "out").exists()


def test_the_snapshot_date_comes_from_the_file_name() -> None:
    assert snapshot_date_of(SOURCE_URL) == "2026-10-01"
    for url in (
        SOURCE_URL.replace("M1%2BM1G", "muu"),
        SOURCE_URL.replace("files=", "other="),
        "https://pilv.transpordiamet.ee/s/x/download",
    ):
        with pytest.raises(ValueError, match="snapshot"):
            snapshot_date_of(url)


def _metadata(*distributions: dict[str, object]) -> bytes:
    return json.dumps({"distributions": list(distributions)}).encode("utf-8")


def _distribution(title: str, url: str) -> dict[str, object]:
    return {"titleEn": title, "titleEt": title, "accessUrls": [url]}


def _share(name: str) -> str:
    return f"https://pilv.transpordiamet.ee/s/abc/download?path=%2F&files={name}"


def test_the_newest_m1_distribution_is_chosen_from_the_portal_metadata() -> None:
    opener = _Opener(
        _metadata(
            _distribution("2026-01-01 M1+M1G", _share("ar_2026-01-01_M1%2BM1G.csv")),
            _distribution("2026-10-01 Other", _share("ar_2026-10-01_muu.csv")),
            _distribution(" 2026-11-01 M1+M1G", _share("ar_2026-11-01_M1%2BM1G.csv")),
            _distribution("README", "https://andmed.eesti.ee/readme"),
            {"titleEn": "2027-01-01 M1+M1G", "accessUrls": []},
        ),
        final_url=METADATA_URL,
    )

    assert latest_distribution(opener) == Distribution(
        _share("ar_2026-11-01_M1%2BM1G.csv"), "2026-11-01"
    )
    assert opener.requests == [METADATA_URL]


def test_a_newest_distribution_outside_the_allowlist_or_unnamed_is_refused() -> None:
    elsewhere = _metadata(
        _distribution("2026-11-01 M1+M1G", "https://example.org/ar_2026-11-01_M1%2BM1G.csv")
    )
    with pytest.raises(ValueError, match="allowlisted"):
        latest_distribution(_Opener(elsewhere, final_url=METADATA_URL))

    mismatch = _metadata(_distribution("2026-11-01 M1+M1G", _share("ar_2026-10-01_M1%2BM1G.csv")))
    with pytest.raises(ValueError, match="disagree"):
        latest_distribution(_Opener(mismatch, final_url=METADATA_URL))

    with pytest.raises(ValueError, match="no M1\\+M1G"):
        latest_distribution(_Opener(_metadata(), final_url=METADATA_URL))
    with pytest.raises(ValueError, match="unsupported"):
        latest_distribution(_Opener(b"not json", final_url=METADATA_URL))
    with pytest.raises(ValueError, match="allowlisted"):
        latest_distribution(_Opener(_metadata(), final_url="https://example.org/api"))

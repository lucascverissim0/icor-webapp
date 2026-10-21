from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from icor.evidence.lu_fleet_acquisition import (
    DATASET_URL,
    _AllowlistedRedirects,
    acquire_fleet,
    latest_resource,
    pinned_resource,
)
from icor.evidence.register_aggregate import SCHEMA

FOLDER = "https://download.data.public.lu/resources/parc-automobile-du-luxembourg"
URL = f"{FOLDER}/20261006-072811/parc-automobile-202610.xml"
MODIFIED = "Tue, 06 Oct 2026 07:28:12 GMT"
# Fictional values of fields that must never leave the register.
SECRETS = {
    "INFOUTI": "HOLDER-ZEBRA-4417",
    "INDUTI": "ZQX7",
    "PVRNUM": "E9*FICT*7781",
    "PVRVAR": "VARIANT-QQ91",
    "TYPMOT": "ENGINE-5531",
    "COUL": "VIOLET-8820",
    "CODMRQ": "7731",
    "DATCIR": "20180906",
    "CODCAR": "BODYX",
    "MVID": "1987654",
}


def _vehicle(**values: str) -> str:
    fields = {
        "CATSTC": "1",
        "LIBCAR": "BERLINE",
        "CATEU": "M1",
        "PAYPVN": "LU",
        "LIBMRQ": "AURORA",
        "TYPCOM": "ORION",
        "DATCIRPRM": "20100315",
        "DATCIR_GD": "20100315",
        "DATHORCIR": "",
        **SECRETS,
    }
    fields.update(values)
    body = "".join(f"<{name}>{escape(value)}</{name}>" for name, value in fields.items())
    return f"\n\t<VEHICLE>{body}</VEHICLE>"


def _xml(vehicles: list[str], *, root: str = "VEHICLES") -> bytes:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><{root}>' + "".join(vehicles) + f"\n</{root}>"
    ).encode()


class _Response:
    def __init__(
        self,
        payload: bytes,
        url: str = URL,
        *,
        length: int | None = None,
        modified: str = MODIFIED,
    ) -> None:
        self._payload = payload
        self._url = url
        self.headers = {
            "Content-Length": str(len(payload) if length is None else length),
            "Last-Modified": modified,
        }

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        chunk, self._payload = (
            (self._payload, b"") if size < 0 else (self._payload[:size], self._payload[size:])
        )
        return chunk

    def geturl(self) -> str:
        return self._url


def _opener(routes: dict[str, _Response], seen: list[str] | None = None):
    def opener(request: urllib.request.Request, timeout: float) -> _Response:
        assert timeout > 0
        if seen is not None:
            seen.append(request.full_url)
        return routes[request.full_url]

    return opener


def _metadata(resources: list[dict[str, object]], *, licence: str = "cc-zero") -> _Response:
    document = {"license": licence, "resources": resources}
    return _Response(json.dumps(document).encode(), DATASET_URL)


def _resource(
    period: str, stamp: str, extension: str = "xml", **extra: object
) -> dict[str, object]:
    url = f"{FOLDER}/{stamp}/parc-automobile-{period}.{extension}"
    return {"format": extension, "url": url, "filesize": 10, "last_modified": "2026", **extra}


def _clock() -> datetime:
    return datetime(2026, 10, 21, 15, 0, tzinfo=UTC)


def _acquire(tmp_path: Path, payload: bytes, **options: object):
    return acquire_fleet(
        tmp_path / "out" / "lu.csv",
        url=URL,
        opener=_opener({URL: _Response(payload)}),
        now=_clock,
        **options,
    )


def _rows(path: Path) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [tuple(row) for row in csv.reader(stream)]


def _fleet() -> list[str]:
    return [
        _vehicle(),
        _vehicle(LIBMRQ="  AURORA ", TYPCOM="ORION  "),
        # First registered in 2014 abroad, in Luxembourg in 2017: cohort 2014, a used import.
        _vehicle(
            CATEU="M1G",
            LIBMRQ="NORTHSTAR",
            TYPCOM="COMET",
            DATCIRPRM="20140201",
            DATCIR_GD="20170620",
            PAYPVN="DE",
        ),
        _vehicle(  # off the road
            LIBMRQ="NORTHSTAR",
            TYPCOM="COMET",
            DATCIRPRM="20140201",
            DATCIR_GD="20140201",
            DATHORCIR="20240101",
        ),
        _vehicle(CATEU="N1", CATSTC="33", TYPCOM="VAN"),
        _vehicle(CATEU="", CATSTC="1", TYPCOM="OLDTIMER", DATCIRPRM="19600101"),
        _vehicle(LIBMRQ="NORTHSTAR", TYPCOM="COMET", DATCIRPRM="", DATCIR_GD="20190101"),
        _vehicle(TYPCOM="", DATCIRPRM="20050101", DATCIR_GD="20050101"),
        _vehicle(TYPCOM="*", DATCIRPRM="20230101", DATCIR_GD="20230101"),
        _vehicle(LIBMRQ="NORTHSTAR", TYPCOM="COMET", DATCIRPRM="20010505", DATCIR_GD="20010505"),
        _vehicle(
            LIBMRQ="NORTHSTAR",
            TYPCOM="COMET",
            DATCIRPRM="20080101",
            DATCIR_GD="20150101",
            PAYPVN="FR",
        ),
        _vehicle(LIBMRQ="NORTHSTAR", TYPCOM="COMET", DATCIRPRM="20080101", DATCIR_GD="20080101"),
    ]


def test_counts_passenger_cars_on_the_road_by_first_registration_anywhere(tmp_path: Path) -> None:
    result = _acquire(tmp_path, _xml(_fleet()))

    assert _rows(result.path) == [
        SCHEMA,
        ("AURORA", "", "2005", "1"),
        ("AURORA", "*", "2023", "1"),
        ("AURORA", "ORION", "2010", "2"),
        ("NORTHSTAR", "COMET", "2001", "1"),
        ("NORTHSTAR", "COMET", "2008", "2"),
        # Cohort 2014, from DATCIRPRM: not 2017 (DATCIR_GD) and not 2018 (DATCIR).
        ("NORTHSTAR", "COMET", "2014", "1"),
        ("NORTHSTAR", "COMET", "", "1"),
    ]
    assert (result.source_rows, result.category_rows, result.status_rows) == (12, 10, 9)
    assert (result.group_count, result.vehicles) == (7, 9)
    assert (result.masked_vehicles, result.blank_model_vehicles) == (1, 1)
    assert result.blank_cohort_vehicles == 1
    assert result.snapshot_date == "20261006"
    assert result.source_url == URL
    assert result.artifact_bytes == result.path.stat().st_size
    assert result.sha256 == hashlib.sha256(result.path.read_bytes()).hexdigest()


def test_profile_reports_counts_only(tmp_path: Path) -> None:
    payload = _xml(_fleet())
    result = _acquire(tmp_path, payload)

    assert result.profile_path == result.path.with_name("lu.csv.profile.json")
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert profile["source"] == {
        "dataset": DATASET_URL,
        "licence": "cc-zero",
        "url": URL,
        "snapshot_date": "20261006",
        "period": "202610",
        "http_last_modified": MODIFIED,
        "resource_last_modified": "",
        "retrieved_at_utc": "2026-10-21T15:00:00Z",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert (result.source_bytes, result.source_sha256) == (
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )
    assert profile["aggregate"] == {
        "path": "lu.csv",
        "bytes": result.artifact_bytes,
        "sha256": result.sha256,
        "groups": 7,
        "vehicles": 9,
    }
    assert profile["funnel"] == {"source_rows": 12, "after_category": 10, "after_status": 9}
    assert profile["vehicles_by_status_within_category"] == {"off_road": 1, "on_road": 9}
    assert profile["vehicles_by_cohort_strict"] == {
        "": 1,
        "2001": 1,
        "2005": 1,
        "2008": 2,
        "2010": 2,
        "2014": 1,
        "2023": 1,
    }
    assert profile["vehicles_by_cohort_and_status_from_2000"] == {
        "2001": {"on_road": 1},
        "2005": {"on_road": 1},
        "2008": {"on_road": 2},
        "2010": {"on_road": 2},
        "2014": {"off_road": 1, "on_road": 1},
        "2023": {"on_road": 1},
    }
    assert profile["national_year_by_cohort_from_2000"] == {
        "2001": {"2001": 1},
        "2005": {"2005": 1},
        "2008": {"2008": 1, "2015": 1},
        "2010": {"2010": 2},
        "2014": {"2017": 1},
        "2023": {"2023": 1},
    }
    assert profile["vehicles_by_national_year_for_cohorts_from_2000"] == {
        "2001": 1,
        "2005": 1,
        "2008": 1,
        "2010": 2,
        "2015": 1,
        "2017": 1,
        "2023": 1,
    }
    assert profile["used_imports_from_2006"] == {
        "definition": "DATCIRPRM before DATCIR_GD, strict filter, cohorts from 2006",
        "vehicles": 2,
        "by_origin_country": {"DE": 1, "FR": 1},
        "by_cohort": {"2008": 1, "2014": 1},
    }
    assert profile["unusable_vehicles"] == {
        "blank_make": 0,
        "masked": 1,
        "blank_model": 1,
        "blank_cohort": 1,
    }
    assert profile["national_category_within_category"] == {"1": 10}
    assert profile["national_cars_outside_category_by_cateu"] == {"": 1}
    assert profile["top_makes"]["makes"] == [["NORTHSTAR", 5], ["AURORA", 4]]


def test_top_makes_are_the_forty_largest(tmp_path: Path) -> None:
    vehicles = [
        _vehicle(LIBMRQ=f"MAKE{number:02d}", TYPCOM="X")
        for number in range(45)
        for _ in range(number + 1)
    ]
    result = _acquire(tmp_path, _xml(vehicles))

    makes = json.loads(result.profile_path.read_text(encoding="utf-8"))["top_makes"]["makes"]
    assert len(makes) == 40
    assert makes[0] == ["MAKE44", 45]
    assert makes[-1] == ["MAKE05", 6]


def test_no_identifying_or_unread_field_reaches_the_output(tmp_path: Path) -> None:
    result = _acquire(tmp_path, _xml(_fleet()))

    written = result.path.read_bytes() + result.profile_path.read_bytes()
    for name, value in SECRETS.items():
        assert value.encode() not in written, name
    assert b"INFOUTI" not in written
    assert b"INDUTI" not in written
    assert result.path.read_text(encoding="utf-8").splitlines()[0] == ",".join(SCHEMA)
    assert sorted(item.name for item in result.path.parent.iterdir()) == [
        "lu.csv",
        "lu.csv.profile.json",
    ]


def test_output_is_deterministic_regardless_of_row_order(tmp_path: Path) -> None:
    first = _acquire(tmp_path / "a", _xml(_fleet()))
    second = _acquire(tmp_path / "b", _xml(_fleet()[::-1]))

    assert first.path.read_bytes() == second.path.read_bytes()
    assert first.sha256 == second.sha256


def test_a_few_stray_dates_are_counted_not_fatal(tmp_path: Path) -> None:
    vehicles = [_vehicle() for _ in range(199)] + [_vehicle(DATCIRPRM="15/03/2010")]

    result = _acquire(tmp_path, _xml(vehicles))

    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert profile["malformed_dates"] == 1
    assert result.blank_cohort_vehicles == 1


def test_finds_the_newest_monthly_xml_resource() -> None:
    resources = [
        _resource("202609", "20260904-065704"),
        _resource("202610", "20261006-072707", "xlsx"),
        _resource("202610", "20261006-072811"),
        _resource("202608", "20260806-075037"),
        {"format": "pdf", "url": f"{FOLDER}/20261201-000000/description.pdf"},
        {**_resource("202611", "20261101-000000"), "url": "https://example.org/parc-202611.xml"},
    ]

    resource = latest_resource(opener=_opener({DATASET_URL: _metadata(resources)}))

    assert resource.url == URL
    assert (resource.period, resource.snapshot_date) == ("202610", "20261006")
    assert resource.declared_bytes == 10


def test_acquires_the_newest_resource_when_none_is_pinned(tmp_path: Path) -> None:
    seen: list[str] = []
    payload = _xml([_vehicle()])
    resources = [_resource("202609", "20260904-065704"), _resource("202610", "20261006-072811")]
    resources[1]["filesize"] = len(payload)
    opener = _opener({DATASET_URL: _metadata(resources), URL: _Response(payload)}, seen)

    result = acquire_fleet(tmp_path / "lu.csv", opener=opener, now=_clock)

    assert seen == [DATASET_URL, URL]
    assert result.snapshot_date == "20261006"


def test_refuses_a_dataset_without_the_cc0_licence_or_an_xml_resource() -> None:
    with pytest.raises(ValueError, match="CC0"):
        latest_resource(
            opener=_opener(
                {DATASET_URL: _metadata([_resource("202610", "20261006-072811")], licence="other")}
            )
        )
    with pytest.raises(ValueError, match="no monthly vehicle XML"):
        latest_resource(
            opener=_opener(
                {DATASET_URL: _metadata([_resource("202610", "20261006-072707", "xlsx")])}
            )
        )


def test_stays_on_the_allowlisted_origins(tmp_path: Path) -> None:
    payload = _xml([_vehicle()])
    for url in (
        URL.replace("https://", "http://"),
        "https://example.org/resources/parc-automobile-du-luxembourg/20261006-072811/"
        "parc-automobile-202610.xml",
        URL.replace("download.data.public.lu", "data.public.lu"),
    ):
        with pytest.raises(ValueError, match="allowlisted"):
            acquire_fleet(tmp_path / "x.csv", url=url, opener=_opener({}))
    with pytest.raises(ValueError, match="monthly vehicle XML"):
        pinned_resource(f"{FOLDER}/20261006-072811/other.xml")

    offsite = _Response(payload, "https://example.org/parc.xml")
    with pytest.raises(ValueError, match="allowlisted"):
        acquire_fleet(tmp_path / "y.csv", url=URL, opener=_opener({URL: offsite}))
    offsite_api = _Response(b"{}", "https://example.org/api")
    with pytest.raises(ValueError, match="allowlisted"):
        latest_resource(opener=_opener({DATASET_URL: offsite_api}))
    assert list(tmp_path.iterdir()) == []


def test_redirects_off_the_allowlist_are_refused_before_they_are_followed() -> None:
    handler = _AllowlistedRedirects()
    request = urllib.request.Request(URL)

    for target in ("https://example.org/x", "http://download.data.public.lu/x"):
        with pytest.raises(ValueError, match="allowlisted"):
            handler.redirect_request(request, None, 302, "Found", {}, target)
    followed = handler.redirect_request(
        request, None, 302, "Found", {}, "https://download.data.public.lu/x"
    )
    assert followed is not None
    assert followed.full_url == "https://download.data.public.lu/x"


def test_the_byte_limit_stops_the_stream(tmp_path: Path) -> None:
    payload = _xml(_fleet())

    with pytest.raises(ValueError, match="byte limit"):
        _acquire(tmp_path, payload, max_bytes=len(payload) - 1)
    result = _acquire(tmp_path, payload, max_bytes=len(payload))
    assert result.source_bytes == len(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_xml([_vehicle()], root="VEHICLE_LIST"), "root element"),
        (_xml([_vehicle(), "<TRUCK></TRUCK>"]), "other than VEHICLE"),
        (_xml([_vehicle().replace("<LIBMRQ>AURORA</LIBMRQ>", "")]), "schema"),
        (_xml([_vehicle().replace("<TYPCOM>ORION", "<TYPCOM>A</TYPCOM><TYPCOM>ORION")]), "repeats"),
        (_xml([_vehicle()])[:-40], "well-formed"),
        (b"", "well-formed"),
        (_xml([_vehicle(DATCIRPRM="15/03/2010") for _ in range(5)]), "date format"),
        (_xml([_vehicle(CATEU="N1")]), "no passenger cars"),
    ],
)
def test_schema_drift_and_unusable_streams_raise_without_output(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _acquire(tmp_path, payload)

    assert not (tmp_path / "out" / "lu.csv").exists()
    assert not (tmp_path / "out" / "lu.csv.profile.json").exists()


def test_a_download_whose_size_differs_from_what_was_announced_raises(tmp_path: Path) -> None:
    payload = _xml([_vehicle()])
    truncated_header = _Response(payload, length=len(payload) + 1)
    with pytest.raises(ValueError, match="incomplete"):
        acquire_fleet(tmp_path / "a.csv", url=URL, opener=_opener({URL: truncated_header}))

    resource = _resource("202610", "20261006-072811")
    resource["filesize"] = len(payload) + 1
    opener = _opener({DATASET_URL: _metadata([resource]), URL: _Response(payload)})
    with pytest.raises(ValueError, match="incomplete"):
        acquire_fleet(tmp_path / "b.csv", opener=opener)
    assert list(tmp_path.iterdir()) == []

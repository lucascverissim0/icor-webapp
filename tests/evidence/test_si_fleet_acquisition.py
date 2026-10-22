from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pytest

from icor.evidence.register_aggregate import SCHEMA
from icor.evidence.si_fleet_acquisition import (
    ARCHIVE_BYTES,
    ARCHIVE_SHA256,
    ARCHIVE_URL,
    CATEGORY,
    EXPLANATION_MEMBER,
    FIRST_REGISTRATION,
    MAKE,
    MEMBERS,
    MODEL,
    NATIONAL_REGISTRATION,
    SOURCE_HEADER,
    STATUS_ID,
    STATUS_NAME,
    AcquisitionResult,
    _AllowlistedRedirects,
    acquire_fleet,
    download_archive,
)

RETRIEVED = "2026-01-02T03:04:05Z"
MODIFIED = "Thu, 20 Jul 2023 09:47:54 GMT"
REGISTERED = ("1", "registrirano")
IN_PROCEDURE = ("3", "v postopku")


def _row(
    make: str,
    model: str,
    *,
    first: str = "03.04.2004",
    national: str | None = None,
    status: tuple[str, str] = REGISTERED,
    category: str = "M1",
) -> list[str]:
    """One fictional register row; every column that is not read holds a sentinel."""

    fields = [f"SENTINEL-{index}" for index in range(len(SOURCE_HEADER))]
    values = {
        MAKE: make,
        MODEL: model,
        CATEGORY: category,
        STATUS_ID: status[0],
        STATUS_NAME: status[1],
        FIRST_REGISTRATION: first,
        NATIONAL_REGISTRATION: first if national is None else national,
    }
    for name, value in values.items():
        fields[SOURCE_HEADER.index(name)] = value
    return fields


def _csv_bytes(rows: Sequence[Sequence[str]], encoding: str) -> bytes:
    stream = io.StringIO()
    csv.writer(stream, delimiter=";", quoting=csv.QUOTE_ALL, lineterminator="\r\n").writerows(rows)
    return stream.getvalue().encode(encoding)


def _archive(
    tmp_path: Path,
    members: Sequence[Sequence[Sequence[str]]],
    *,
    headers: Sequence[Sequence[str]] | None = None,
    header_encoding: str = "utf-8",
    names: Sequence[str] = (*MEMBERS, EXPLANATION_MEMBER),
) -> Path:
    path = tmp_path / "register.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        for index, member in enumerate(members):
            header = headers[index] if headers else SOURCE_HEADER
            data = _csv_bytes([header], header_encoding) + _csv_bytes(member, "cp1250")
            bundle.writestr(names[index], data)
        bundle.writestr(names[len(members)], b"three files together make the register")
    return path


def _three(rows: Sequence[Sequence[str]]) -> list[Sequence[Sequence[str]]]:
    """Spread rows over the three members, as the register is split."""

    return [rows[0::3], rows[1::3], rows[2::3]]


def _acquire(
    tmp_path: Path, archive: Path, name: str = "si.csv", **options: object
) -> AcquisitionResult:
    return acquire_fleet(
        tmp_path / "out" / name,
        archive,
        expected_bytes=archive.stat().st_size,
        expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_url=ARCHIVE_URL,
        last_modified=MODIFIED,
        retrieved_utc=RETRIEVED,
        **options,
    )


def _rows(path: Path) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [tuple(row) for row in csv.reader(stream)]


def test_keeps_registered_passenger_cars_by_make_model_and_first_registration_year(
    tmp_path: Path,
) -> None:
    rows = [
        _row("AURORA", "NOVA / 1.6 / TDI", first="03.04.2004", national="05.06.2012"),
        _row("AURORA", "NOVA  / 1.6 / TDI", first="09.09.2004"),
        _row("NORTHSTAR", "COMET / 2.0", first="01.01.2008", category="M1G"),
        _row("AURORA", "NOVA / 1.6 / TDI", first="01.01.2005", status=IN_PROCEDURE),
        _row("AURORA", "TRUCK / 2.0", category="N1"),
        _row("ŠIRIUS", "ORION / 1.0", first="01.01.1988"),
        _row("AURORA", "/", first="01.01.2001"),
        _row("AURORA", "", first="01.01.2001"),
        _row("NORTHSTAR", "COMET / 2.0", first=""),
        _row("AURORA", "BIKE", category="L3e"),
    ]

    result = _acquire(tmp_path, _archive(tmp_path, _three(rows)))

    assert _rows(result.path) == [
        SCHEMA,
        ("AURORA", "", "2001", "1"),
        ("AURORA", "/", "2001", "1"),
        ("AURORA", "NOVA / 1.6 / TDI", "2004", "2"),
        ("NORTHSTAR", "COMET / 2.0", "2008", "1"),
        ("NORTHSTAR", "COMET / 2.0", "", "1"),
        ("ŠIRIUS", "ORION / 1.0", "1988", "1"),
    ]
    assert (result.source_rows, result.category_rows, result.status_rows) == (10, 8, 7)
    assert (result.group_count, result.vehicles) == (6, 7)
    assert (result.masked_vehicles, result.blank_model_vehicles) == (1, 1)
    assert result.blank_cohort_vehicles == 1
    assert result.artifact_bytes == result.path.stat().st_size
    assert result.artifact_sha256 == hashlib.sha256(result.path.read_bytes()).hexdigest()
    assert result.model_head is None


def test_cohort_is_the_first_registration_anywhere_not_the_national_date(
    tmp_path: Path,
) -> None:
    rows = [_row("AURORA", "NOVA", first="15.06.2003", national="20.01.2011")]

    result = _acquire(tmp_path, _archive(tmp_path, _three(rows)))

    assert _rows(result.path)[1:] == [("AURORA", "NOVA", "2003", "1")]


def test_unusable_dates_become_a_blank_cohort(tmp_path: Path) -> None:
    rows = [
        _row("AURORA", "NOVA", first="2004"),
        _row("AURORA", "NOVA", first="31.12.04"),
        _row("AURORA", "NOVA", first="   "),
        _row("AURORA", "NOVA", first="1.2.2004"),
    ]

    result = _acquire(tmp_path, _archive(tmp_path, _three(rows)))

    assert _rows(result.path)[1:] == [
        ("AURORA", "NOVA", "2004", "1"),
        ("AURORA", "NOVA", "", "3"),
    ]
    assert result.blank_cohort_vehicles == 3


def test_no_personal_or_identifying_value_leaves_the_register(tmp_path: Path) -> None:
    rows = [_row("AURORA", "NOVA / 1.6 / TDI"), _row("NORTHSTAR", "COMET", national="01.01.2020")]

    result = _acquire(
        tmp_path,
        _archive(tmp_path, _three(rows)),
        model_head_destination=tmp_path / "out" / "head.csv",
    )

    assert result.model_head is not None
    outputs = [
        result.path.read_bytes(),
        result.model_head.path.read_bytes(),
        result.profile_path.read_bytes(),
        json.dumps(asdict(result), default=str).encode("utf-8"),
    ]
    for output in outputs:
        assert b"SENTINEL" not in output
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "head.csv",
        "si.csv",
        "si.csv.profile.json",
    ]


def test_profile_holds_counts_only_and_the_source_facts(tmp_path: Path) -> None:
    rows = [
        _row("AURORA", "NOVA", first="03.04.2004", national="05.06.2012"),
        _row("AURORA", "NOVA", first="09.09.2004", national="09.09.2004"),
        _row("AURORA", "NOVA", first="01.01.1999", national="02.02.2001"),
        _row("NORTHSTAR", "COMET", first="01.01.2010", national=""),
        _row("NORTHSTAR", "COMET", first="01.01.2010", status=IN_PROCEDURE),
        _row("NORTHSTAR", "COMET", first="05.05.2015", national="04.04.2014"),
        _row("NORTHSTAR", "COMET", first="12.12.1985"),
        _row("NORTHSTAR", "COMET", first="01.01.2004", status=("9", "odjavljeno")),
        _row("AURORA", "TRUCK", category="N1"),
    ]
    archive = _archive(tmp_path, _three(rows))

    result = _acquire(tmp_path, archive)
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))

    assert profile["source"] == {
        "dataset": "Evidenca registriranih vozil - presek stanja",
        "publisher_portal": "OPSI, podatki.gov.si",
        "licence": "CC BY 4.0",
        "snapshot_date": "2023-06-30",
        "url": ARCHIVE_URL,
        "last_modified": MODIFIED,
        "retrieved_utc": RETRIEVED,
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    assert profile["funnel"] == {"source_rows": 9, "category_rows": 8, "status_rows": 6}
    assert profile["status_values_all_rows"] == [
        {"id": "1", "description": "registrirano", "vehicles": 7},
        {"id": "3", "description": "v postopku", "vehicles": 1},
        {"id": "9", "description": "odjavljeno", "vehicles": 1},
    ]
    assert profile["vehicles_by_status_within_category"] == [
        {"id": "1", "description": "registrirano", "vehicles": 6},
        {"id": "3", "description": "v postopku", "vehicles": 1},
        {"id": "9", "description": "odjavljeno", "vehicles": 1},
    ]
    assert profile["vehicles_by_cohort_strict"] == {
        "1985": 1,
        "1999": 1,
        "2004": 2,
        "2010": 1,
        "2015": 1,
    }
    assert profile["vehicles_by_cohort_and_status_2000_plus"] == {
        "2004": {"1": 2, "9": 1},
        "2010": {"1": 1, "3": 1},
        "2015": {"1": 1},
    }
    assert profile["vehicles_by_national_year_2000_plus"] == {
        "2001": 1,
        "2004": 1,
        "2012": 1,
        "2014": 1,
    }
    assert profile["national_year_before_floor_vehicles"] == 1
    assert profile["national_year_blank_vehicles"] == 1
    zeros = {"national_later": 0, "national_earlier": 0, "national_blank": 0, "same_year": 0}
    assert profile["used_imports_by_cohort_2000_plus"] == {
        "2004": {"vehicles": 2, **zeros, "same_year": 1, "national_later": 1},
        "2010": {"vehicles": 1, **zeros, "national_blank": 1},
        "2015": {"vehicles": 1, **zeros, "national_earlier": 1},
    }
    assert (
        profile["masked_vehicles"],
        profile["blank_model_vehicles"],
        profile["blank_cohort_vehicles"],
        profile["blank_make_vehicles"],
    ) == (0, 0, 0, 0)
    assert profile["top_makes"] == [
        {"make": "AURORA", "vehicles": 3},
        {"make": "NORTHSTAR", "vehicles": 3},
    ]
    assert profile["aggregate"] == {
        "file": "si.csv",
        "bytes": result.artifact_bytes,
        "sha256": result.artifact_sha256,
        "group_count": result.group_count,
        "vehicles": result.vehicles,
    }
    assert profile["model_head_aggregate"] is None
    assert set(profile) == {
        "source",
        "filters",
        "funnel",
        "status_values_all_rows",
        "vehicles_by_status_within_category",
        "vehicles_by_cohort_strict",
        "vehicles_by_cohort_and_status_2000_plus",
        "vehicles_by_national_year_2000_plus",
        "national_year_before_floor_vehicles",
        "national_year_blank_vehicles",
        "used_imports_by_cohort_2000_plus",
        "masked_vehicles",
        "blank_model_vehicles",
        "blank_cohort_vehicles",
        "blank_make_vehicles",
        "top_makes",
        "aggregate",
        "model_head_aggregate",
    }


def test_profile_lists_the_forty_largest_makes(tmp_path: Path) -> None:
    rows = [
        _row(f"MAKE{index:02d}", "X", first="01.01.2010")
        for index in range(45)
        for _ in range(45 - index)
    ]

    result = _acquire(tmp_path, _archive(tmp_path, _three(rows)))
    top = json.loads(result.profile_path.read_text(encoding="utf-8"))["top_makes"]

    assert len(top) == 40
    assert top[0] == {"make": "MAKE00", "vehicles": 45}
    assert top[-1] == {"make": "MAKE39", "vehicles": 6}


def test_model_head_aggregate_keeps_the_text_before_the_first_slash(tmp_path: Path) -> None:
    rows = [
        _row("AURORA", "NOVA / 1.6 / TDI", first="03.04.2004"),
        _row("AURORA", "NOVA / 1.9", first="09.09.2004"),
        _row("AURORA", "NOVA", first="09.09.2004"),
        _row("AURORA", "/ 1200 / J", first="01.01.2001"),
    ]

    result = _acquire(
        tmp_path,
        _archive(tmp_path, _three(rows)),
        model_head_destination=tmp_path / "out" / "head.csv",
    )

    assert result.model_head is not None
    assert _rows(result.model_head.path) == [
        SCHEMA,
        ("AURORA", "", "2001", "1"),
        ("AURORA", "NOVA", "2004", "3"),
    ]
    assert (result.model_head.group_count, result.model_head.vehicles) == (2, 4)
    assert len(_rows(result.path)) == 5
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert profile["model_head_aggregate"]["sha256"] == result.model_head.sha256


def test_output_is_deterministic_regardless_of_row_order_and_member_split(
    tmp_path: Path,
) -> None:
    rows = [
        _row("NORTHSTAR", "COMET", first="01.01.2002"),
        _row("AURORA", "NOVA", first="01.01.2001", national="01.01.2005"),
        _row("AURORA", "NOVA", first="01.01.2001"),
        _row("AURORA", "ZETA", first="01.01.2009"),
    ]
    first_dir, second_dir = tmp_path / "a", tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()

    first = _acquire(first_dir, _archive(first_dir, _three(rows)))
    second = _acquire(second_dir, _archive(second_dir, [rows[::-1], [], []]))

    assert first.path.read_bytes() == second.path.read_bytes()
    left = json.loads(first.profile_path.read_text(encoding="utf-8"))
    right = json.loads(second.profile_path.read_text(encoding="utf-8"))
    assert left.pop("source") != right.pop("source")
    assert left == right


def test_refuses_an_archive_that_is_not_the_pinned_release(tmp_path: Path) -> None:
    archive = _archive(tmp_path, _three([_row("AURORA", "NOVA")]))

    with pytest.raises(ValueError, match="pinned release"):
        acquire_fleet(
            tmp_path / "out" / "si.csv",
            archive,
            expected_bytes=archive.stat().st_size,
            expected_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="pinned release"):
        acquire_fleet(
            tmp_path / "out" / "si.csv",
            archive,
            expected_bytes=archive.stat().st_size + 1,
            expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
    assert not (tmp_path / "out").exists()


def test_pins_describe_the_published_snapshot() -> None:
    assert ARCHIVE_BYTES == 384_940_669
    assert len(ARCHIVE_SHA256) == 64
    assert set(ARCHIVE_SHA256) <= set("0123456789abcdef")
    assert ARCHIVE_URL.startswith("https://podatki.gov.si/")
    assert len(SOURCE_HEADER) == 102
    assert len(set(SOURCE_HEADER)) == 102


def test_refuses_an_empty_register_without_writing_anything(tmp_path: Path) -> None:
    rows = [_row("AURORA", "TRUCK", category="N1"), _row("AURORA", "BIKE", status=IN_PROCEDURE)]

    with pytest.raises(ValueError, match="no passenger cars"):
        _acquire(tmp_path, _archive(tmp_path, _three(rows)))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "drift",
    [
        pytest.param(list(SOURCE_HEADER[:-1]), id="column-removed"),
        pytest.param([*SOURCE_HEADER, "Extra"], id="column-added"),
        pytest.param([*SOURCE_HEADER[1:], SOURCE_HEADER[0]], id="column-moved"),
        pytest.param(
            [name.replace("Znamka", "Marka") for name in SOURCE_HEADER], id="column-renamed"
        ),
    ],
)
def test_refuses_schema_drift_in_any_member(tmp_path: Path, drift: list[str]) -> None:
    rows = [_row("AURORA", "NOVA")]
    for position in range(3):
        headers = [list(SOURCE_HEADER)] * 3
        headers[position] = drift
        directory = tmp_path / str(position)
        directory.mkdir()
        with pytest.raises(ValueError, match="header is unsupported"):
            _acquire(directory, _archive(directory, [rows, rows, rows], headers=headers))


def test_refuses_a_header_that_is_not_utf8_and_a_malformed_row(tmp_path: Path) -> None:
    rows = [_row("AURORA", "NOVA")]
    drifted = [name.replace("Znamka", "\u0160amka") for name in SOURCE_HEADER]
    encoded = _directory(tmp_path / "e")
    with pytest.raises(ValueError, match="header is unsupported"):
        _acquire(
            encoded,
            _archive(encoded, [rows, rows, rows], headers=[drifted] * 3, header_encoding="cp1250"),
        )
    short = _directory(tmp_path / "m")
    with pytest.raises(ValueError, match="row is malformed"):
        _acquire(short, _archive(short, [rows, rows, [rows[0][:-1]]]))


def test_refuses_unexpected_archive_members(tmp_path: Path) -> None:
    rows = [_row("AURORA", "NOVA")]
    names = (*MEMBERS[:2], "other/Nio_vozila_3.csv", EXPLANATION_MEMBER)

    with pytest.raises(ValueError, match="members are unsupported"):
        _acquire(tmp_path, _archive(tmp_path, [rows, rows, rows], names=names))


def _directory(path: Path) -> Path:
    path.mkdir()
    return path


class _Response:
    def __init__(
        self,
        payload: bytes,
        url: str = ARCHIVE_URL,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self._url = url
        self.headers = headers if headers is not None else {}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk

    def geturl(self) -> str:
        return self._url


def _opener(payload: bytes, url: str = ARCHIVE_URL, headers: dict[str, str] | None = None):
    return lambda request, timeout: _Response(payload, url, headers)


def test_download_records_the_pinned_bytes_checksum_and_server_date(tmp_path: Path) -> None:
    payload = b"abc"
    digest = hashlib.sha256(payload).hexdigest()

    download = download_archive(
        tmp_path / "dl" / "a.zip",
        opener=_opener(payload, headers={"Last-Modified": MODIFIED}),
        expected_bytes=3,
        expected_sha256=digest,
    )

    assert download.path.read_bytes() == payload
    assert (download.bytes, download.sha256, download.last_modified) == (3, digest, MODIFIED)
    assert download.retrieved_utc.endswith("Z")


def test_download_stays_on_the_allowlisted_origin_and_the_pinned_size(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"abc").hexdigest()
    pinned = {"expected_bytes": 3, "expected_sha256": digest}

    with pytest.raises(ValueError, match="byte limit"):
        download_archive(tmp_path / "b.zip", opener=_opener(b"abcd"), **pinned)
    with pytest.raises(ValueError, match="pinned release"):
        download_archive(tmp_path / "c.zip", opener=_opener(b"ab"), **pinned)
    with pytest.raises(ValueError, match="pinned release"):
        download_archive(tmp_path / "d.zip", opener=_opener(b"abd"), **pinned)
    with pytest.raises(ValueError, match="allowlisted"):
        download_archive(
            tmp_path / "e.zip",
            opener=_opener(b"abc", "https://example.org/x.zip"),
            **pinned,
        )
    for url in (
        "http://podatki.gov.si/x.zip",
        "https://podatki.gov.si.example.org/x.zip",
        "https://podatki.gov.si@example.org/x.zip",
        "https://example.org/podatki.gov.si/x.zip",
    ):
        with pytest.raises(ValueError, match="allowlisted"):
            download_archive(tmp_path / "f.zip", url=url, opener=_opener(b"abc"), **pinned)
    assert list(tmp_path.iterdir()) == []


def test_a_redirect_off_the_allowlist_is_refused_before_it_is_followed() -> None:
    handler = _AllowlistedRedirects()
    request = urllib.request.Request(ARCHIVE_URL)

    with pytest.raises(ValueError, match="allowlisted"):
        handler.redirect_request(request, None, 302, "Found", {}, "https://example.org/x.zip")
    with pytest.raises(ValueError, match="allowlisted"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://podatki.gov.si/x.zip")
    followed = handler.redirect_request(
        request, None, 302, "Found", {}, "https://podatki.gov.si/other.zip"
    )
    assert followed is not None
    assert followed.full_url == "https://podatki.gov.si/other.zip"

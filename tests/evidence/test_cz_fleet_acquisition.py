from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

from icor.evidence.cz_fleet_acquisition import (
    REQUIRED_COLUMNS,
    SOURCE_URL,
    _AllowlistedRedirects,
    acquire_fleet,
)
from icor.evidence.register_aggregate import SCHEMA

# Columns the module must never read into any output: all values below are fictional.
SECRETS = {
    "VIN": "VINFICT0000000001",
    "Číslo motoru": "MOTFICT7777",
    "Výrobní číslo karoserie": "KARFICT8888",
    "Číslo TP": "TPFICT9999",
    "Číslo ORV": "ORVFICT5555",
    "PČV": "PCVFICT123456",
    "Doplňkový text na TP": "line one\r\nTEXTFICT private note",
}
HEADER = [
    "Datum 1. registrace",
    "Datum 1. registrace v ČR",
    "Druh vozidla",
    "Kategorie vozidla",
    "Tovární značka",
    "VIN",
    "Obchodní označení",
    "Číslo TP",
    "Číslo ORV",
    "Status",
    "PČV",
    "Číslo motoru",
    "Výrobní číslo karoserie",
    "Doplňkový text na TP",
    "Rok výroby",
]
STRICT = "PROVOZOVANÉ"
WITHDRAWN = "VYŘAZENO Z PROVOZU"
CLOCK = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
DISPOSITION = "attachment; filename=RSV_vypis_vozidel_20261001.csv"
HEADERS = {
    "Content-Disposition": DISPOSITION,
    "Last-Modified": "Thu, 01 Oct 2026 04:00:00 GMT",
}


def _row(
    make: str,
    model: str,
    first: str = "2010-05-05",
    national: str | None = None,
    *,
    status: str = STRICT,
    category: str = "M1",
    kind: str = "OSOBNÍ AUTOMOBIL",
    production: str = "2010",
) -> dict[str, str]:
    return {
        **SECRETS,
        "Datum 1. registrace": first,
        "Datum 1. registrace v ČR": first if national is None else national,
        "Druh vozidla": kind,
        "Kategorie vozidla": category,
        "Tovární značka": make,
        "Obchodní označení": model,
        "Status": status,
        "Rok výroby": production,
    }


def _payload(rows: list[dict[str, str]], *, header: list[str] = HEADER, bom: bool = True) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([row.get(name, "") for name in header])
    return (("﻿" if bom else "") + buffer.getvalue()).encode("utf-8")


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
        url: str = SOURCE_URL,
        chunk: int = 1 << 20,
    ) -> None:
        self._stream = io.BytesIO(payload)
        self.headers = HEADERS if headers is None else headers
        self._url = url
        self._chunk = chunk

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(self._chunk if size < 0 else min(size, self._chunk))

    def geturl(self) -> str:
        return self._url


def _serve(payload: bytes, **options):
    seen: list[urllib.request.Request] = []

    def opener(request: urllib.request.Request, timeout: float) -> _Response:
        seen.append(request)
        return _Response(payload, **options)

    opener.seen = seen  # type: ignore[attr-defined]
    return opener


def _acquire(tmp_path: Path, payload: bytes, **options):
    return acquire_fleet(
        tmp_path / "out" / "cz.csv",
        opener=_serve(payload, **options),
        now=lambda: CLOCK,
    )


def _read(path: Path) -> list[tuple[str, ...]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [tuple(row) for row in csv.reader(stream)]


FLEET = [
    _row("AURORA", "NOVA", "2010-05-05", "2015-06-01"),
    _row("AURORA", "NOVA  ", "2010-09-09"),
    _row("NORTHSTAR", "COMET, S", "2004-01-02", "2012-02-02", category="M1G"),
    _row("NORTHSTAR", "COMET, S", "2004-03-03", category=" m1 "),
    _row("AURORA", "NOVA", "2010-01-01", status=WITHDRAWN),
    _row("AURORA", "NOVA", "2010-01-01", status="ZÁNIK"),
    _row("AURORA", "NOVA", "2010-01-01", category="N1"),
    _row("AURORA", "NOVA", "2010-01-01", category=".", kind="Osobní automobil"),
    _row("AURORA", "", "2020-02-02", "2022-02-02"),
    _row("AURORA", ".", "2020-02-02"),
    _row("NORTHSTAR", "ZEPHYR", "", "2012-03-04", production="2011"),
    _row("NORTHSTAR", "ZEPHYR", "12.03.2011", "", production=""),
    _row(".", "ZEPHYR", "2015-05-05", status=WITHDRAWN),
]


def test_filters_and_cohort_come_from_the_first_registration_anywhere(tmp_path: Path) -> None:
    result = _acquire(tmp_path, _payload(FLEET))

    assert _read(result.path) == [
        SCHEMA,
        ("AURORA", "", "2020", "1"),
        ("AURORA", ".", "2020", "1"),
        ("AURORA", "NOVA", "2010", "2"),
        ("NORTHSTAR", "COMET, S", "2004", "2"),
        ("NORTHSTAR", "ZEPHYR", "", "2"),
    ]
    assert _read(result.with_withdrawn.path) == [
        SCHEMA,
        (".", "ZEPHYR", "2015", "1"),
        ("AURORA", "", "2020", "1"),
        ("AURORA", ".", "2020", "1"),
        ("AURORA", "NOVA", "2010", "3"),
        ("NORTHSTAR", "COMET, S", "2004", "2"),
        ("NORTHSTAR", "ZEPHYR", "", "2"),
    ]
    assert (result.source_rows, result.category_rows, result.status_rows) == (13, 11, 8)
    assert (result.group_count, result.vehicles) == (5, 8)
    assert (
        result.masked_vehicle_count,
        result.blank_model_vehicle_count,
        result.blank_cohort_vehicle_count,
    ) == (1, 1, 2)
    variant = result.with_withdrawn
    assert (variant.status_rows, variant.group_count, variant.vehicles) == (10, 6, 10)
    assert (
        variant.masked_vehicle_count,
        variant.blank_model_vehicle_count,
        variant.blank_cohort_vehicle_count,
    ) == (2, 1, 2)
    assert result.sha256 == hashlib.sha256(result.path.read_bytes()).hexdigest()
    assert result.artifact_bytes == result.path.stat().st_size
    assert variant.sha256 == hashlib.sha256(variant.path.read_bytes()).hexdigest()
    assert variant.path.name == "cz-with-withdrawn.csv"


def test_the_profile_holds_provenance_the_funnel_and_counts_only(tmp_path: Path) -> None:
    payload = _payload(FLEET)
    result = _acquire(tmp_path, payload)

    assert result.profile_path.name == "cz.csv.profile.json"
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert profile["source"] == {
        "url": SOURCE_URL,
        "schema_url": "https://download.dataovozidlech.cz/schema/vypisvozidel",
        "licence": profile["source"]["licence"],
        "snapshot_date": "2026-10-01",
        "last_modified": "Thu, 01 Oct 2026 04:00:00 GMT",
        "retrieved_at_utc": "2026-10-21T12:00:00Z",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "gzip": False,
    }
    assert "CC0" in profile["source"]["licence"]
    assert profile["funnel"] == {
        "source_rows": 13,
        "category_rows": 11,
        "status_rows_strict": 8,
        "status_rows_with_withdrawn": 10,
    }
    assert profile["vehicles_by_status_within_category"] == {STRICT: 8, WITHDRAWN: 2, "ZÁNIK": 1}
    assert profile["vehicles_by_cohort_strict"] == {
        "2004": 2,
        "2010": 2,
        "2020": 2,
        "(blank)": 2,
    }
    assert profile["vehicles_by_cohort_and_status_2000_plus"] == {
        "2004": {STRICT: 2},
        "2010": {STRICT: 2, WITHDRAWN: 1, "ZÁNIK": 1},
        "2015": {WITHDRAWN: 1},
        "2020": {STRICT: 2},
    }
    assert profile["vehicles_by_national_year_strict"] == {
        "2004": 1,
        "2010": 1,
        "2012": 2,
        "2015": 1,
        "2020": 1,
        "2022": 1,
        "(blank)": 1,
    }
    assert profile["national_year_against_first_registration_strict_2000_plus"] == {
        "2004": {"national_year_later": 1, "national_year_same": 1},
        "2010": {"national_year_later": 1, "national_year_same": 1},
        "2020": {"national_year_later": 1, "national_year_same": 1},
    }
    assert profile["blank_cohort"] == {
        "by_status_within_category": {STRICT: 2},
        "strict_by_vehicle_kind": {"OSOBNÍ AUTOMOBIL": 2},
        "strict_by_production_year": {"2011": 1, "(blank)": 1},
    }
    assert profile["uncategorised_passenger_kind_by_status"] == {STRICT: 1}
    assert profile["top_makes_strict"] == [
        {"make": "AURORA", "vehicles": 4},
        {"make": "NORTHSTAR", "vehicles": 4},
    ]
    assert profile["vehicle_kind_within_category_by_status"] == {
        "OSOBNÍ AUTOMOBIL": {STRICT: 8, WITHDRAWN: 2, "ZÁNIK": 1}
    }
    assert profile["aggregates"]["strict"]["sha256"] == result.sha256
    assert profile["aggregates"]["with_withdrawn"]["vehicles"] == 10
    assert profile["aggregates"]["strict"]["blank_cohort_vehicles"] == 2


def test_only_forty_makes_are_profiled(tmp_path: Path) -> None:
    rows = [_row(f"MAKE{index:02d}", "X") for index in range(45)]
    rows += [_row("MAKE44", "X")]

    profile = json.loads(_acquire(tmp_path, _payload(rows)).profile_path.read_text("utf-8"))

    assert len(profile["top_makes_strict"]) == 40
    assert profile["top_makes_strict"][0] == {"make": "MAKE44", "vehicles": 2}


def test_no_identifying_column_reaches_any_output(tmp_path: Path) -> None:
    result = _acquire(tmp_path, _payload(FLEET))

    outputs = [result.path, result.with_withdrawn.path, result.profile_path]
    blob = b"".join(path.read_bytes() for path in outputs)
    blob += json.dumps(asdict(result), default=str).encode("utf-8")
    for column, value in SECRETS.items():
        for part in value.splitlines():
            assert part.encode() not in blob
        assert column.encode() not in blob
    assert sorted(item.name for item in (tmp_path / "out").iterdir()) == [
        "cz-with-withdrawn.csv",
        "cz.csv",
        "cz.csv.profile.json",
    ]


def test_a_gzip_download_is_decompressed_while_the_served_bytes_are_hashed(
    tmp_path: Path,
) -> None:
    plain = _acquire(tmp_path / "plain", _payload(FLEET))
    compressed = gzip.compress(_payload(FLEET))

    result = _acquire(
        tmp_path / "gz",
        compressed,
        headers={**HEADERS, "Content-Encoding": "gzip"},
        chunk=7,
    )

    assert result.source_gzip is True
    assert result.source_bytes == len(compressed)
    assert result.source_sha256 == hashlib.sha256(compressed).hexdigest()
    assert result.path.read_bytes() == plain.path.read_bytes()
    assert result.with_withdrawn.sha256 == plain.with_withdrawn.sha256
    profile = json.loads(result.profile_path.read_text(encoding="utf-8"))
    assert profile["source"]["gzip"] is True
    assert profile["source"]["bytes"] == len(compressed)


def test_the_request_asks_for_gzip(tmp_path: Path) -> None:
    opener = _serve(_payload(FLEET))

    acquire_fleet(tmp_path / "cz.csv", opener=opener, now=lambda: CLOCK)

    assert opener.seen[0].get_header("Accept-encoding") == "gzip"
    assert opener.seen[0].full_url == SOURCE_URL


def test_a_truncated_or_padded_gzip_stream_is_refused(tmp_path: Path) -> None:
    compressed = gzip.compress(_payload(FLEET))
    gzip_headers = {**HEADERS, "Content-Encoding": "gzip"}

    with pytest.raises(ValueError, match="gzip stream"):
        _acquire(tmp_path / "t", compressed[:-12], headers=gzip_headers)
    with pytest.raises(ValueError, match="gzip stream"):
        _acquire(tmp_path / "p", compressed + b"padding", headers=gzip_headers)
    with pytest.raises(ValueError, match="gzip stream"):
        _acquire(tmp_path / "e", b"", headers=gzip_headers)
    assert not (tmp_path / "t").exists()


def test_the_snapshot_date_names_the_files_through_the_placeholder(tmp_path: Path) -> None:
    result = acquire_fleet(
        tmp_path / "cz-registered-cars-{date}.csv",
        opener=_serve(_payload(FLEET)),
        now=lambda: CLOCK,
    )

    assert result.snapshot_date == "20261001"
    assert result.path.name == "cz-registered-cars-20261001.csv"
    assert result.with_withdrawn.path.name == "cz-registered-cars-20261001-with-withdrawn.csv"
    assert result.profile_path.name == "cz-registered-cars-20261001.csv.profile.json"
    assert result.path.exists()


@pytest.mark.parametrize(
    "disposition",
    [
        None,
        "attachment",
        "attachment; filename=other.csv",
        "filename=RSV_vypis_vozidel_20261345.csv",
    ],
)
def test_a_download_without_a_valid_snapshot_date_is_refused(
    tmp_path: Path, disposition: str | None
) -> None:
    headers = {} if disposition is None else {"Content-Disposition": disposition}

    with pytest.raises(ValueError, match="snapshot date"):
        _acquire(tmp_path, _payload(FLEET), headers=headers)
    assert not (tmp_path / "out").exists()


def test_only_the_allowlisted_https_origin_is_fetched(tmp_path: Path) -> None:
    for url in (
        "http://download.dataovozidlech.cz/vypiszregistru/vypisvozidel",
        "https://example.org/vypiszregistru/vypisvozidel",
        "https://download.dataovozidlech.cz.example.org/x",
        "https://download.dataovozidlech.cz:8443/x",
    ):
        opener = _serve(_payload(FLEET))
        with pytest.raises(ValueError, match="allowlisted"):
            acquire_fleet(tmp_path / "cz.csv", opener=opener, url=url)
        assert opener.seen == []
    with pytest.raises(ValueError, match="allowlisted"):
        _acquire(tmp_path, _payload(FLEET), url="https://example.org/elsewhere.csv")
    assert not (tmp_path / "out").exists()


def test_a_redirect_off_the_allowlist_is_refused() -> None:
    handler = _AllowlistedRedirects()
    request = urllib.request.Request(SOURCE_URL)

    with pytest.raises(ValueError, match="allowlisted"):
        handler.redirect_request(request, None, 302, "Found", {}, "https://example.org/x.csv")
    with pytest.raises(ValueError, match="allowlisted"):
        handler.redirect_request(
            request, None, 302, "Found", {}, "http://download.dataovozidlech.cz/x.csv"
        )


def test_the_byte_limit_stops_an_oversized_download_before_anything_is_written(
    tmp_path: Path,
) -> None:
    payload = _payload(FLEET)

    with pytest.raises(ValueError, match="byte limit"):
        acquire_fleet(
            tmp_path / "out" / "cz.csv",
            opener=_serve(payload),
            byte_limit=len(payload) - 1,
        )
    assert not (tmp_path / "out").exists()

    result = acquire_fleet(
        tmp_path / "ok" / "cz.csv", opener=_serve(payload), byte_limit=len(payload)
    )
    assert result.source_bytes == len(payload)


def test_schema_drift_and_malformed_input_are_refused(tmp_path: Path) -> None:
    for index, column in enumerate(REQUIRED_COLUMNS):
        header = [name for name in HEADER if name != column]
        with pytest.raises(ValueError, match="schema"):
            _acquire(tmp_path / f"s{index}", _payload([_row("A", "B")], header=header))
    with pytest.raises(ValueError, match="schema"):
        _acquire(tmp_path / "dup", _payload([_row("A", "B")], header=[*HEADER, "Status"]))
    ragged = _payload([_row("A", "B")]).replace(b",VINFICT0000000001", b"", 1)
    with pytest.raises(ValueError, match="malformed"):
        _acquire(tmp_path / "ragged", ragged)
    with pytest.raises(ValueError, match="empty"):
        _acquire(tmp_path / "empty", b"")
    with pytest.raises(ValueError, match="no rows"):
        _acquire(tmp_path / "norows", _payload([]))
    with pytest.raises(ValueError, match="encoding"):
        _acquire(
            tmp_path / "br",
            _payload(FLEET),
            headers={**HEADERS, "Content-Encoding": "br"},
        )
    with pytest.raises(ValueError, match="no passenger cars"):
        _acquire(tmp_path / "none", _payload([_row("A", "B", category="N1")]))
    assert not any((tmp_path / name).exists() for name in ("dup", "ragged", "empty", "none"))


def test_the_columns_are_found_by_name_not_position(tmp_path: Path) -> None:
    shuffled = [*reversed(HEADER)]

    result = _acquire(tmp_path, _payload(FLEET, header=shuffled, bom=False))

    assert _read(result.path)[1] == ("AURORA", "", "2020", "1")
    assert result.vehicles == 8

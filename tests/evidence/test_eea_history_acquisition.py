from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from icor.evidence.eea_history_acquisition import ANNUAL_RELEASES, acquire_year
from icor.evidence.sources.eea import ANNUAL_AGGREGATE_SCHEMA


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload

    def geturl(self) -> str:
        return "https://discodata.eea.europa.eu/sql"


def test_acquire_year_pages_and_writes_canonical_semicolon_csv(tmp_path: Path) -> None:
    pages = [
        {
            "results": [
                {
                    "Year": 2010,
                    "Status": "F",
                    "Version_file": "v2",
                    "MS": "DE",
                    "Mk": "VW",
                    "Cn": "GOLF",
                    "TAN": None,
                    "T": "1K",
                    "Va": "A",
                    "Ve": "1",
                    "Ft": "petrol",
                    "Registrations": 2,
                    "SourceRows": 2,
                }
            ]
        },
        {"results": []},
    ]
    requested_pages: list[int] = []

    def opener(request: object, timeout: int) -> _Response:
        assert timeout == 120
        url = request.full_url  # type: ignore[attr-defined]
        requested_pages.append(int(url.split("p=")[1].split("&")[0]))
        return _Response(pages.pop(0))

    destination = tmp_path / "2010.csv"
    release = replace(
        ANNUAL_RELEASES[2010],
        expected_source_rows=2,
        expected_accepted_rows=2,
        expected_rejected_rows=0,
        expected_registrations=2,
    )
    result = acquire_year(release, destination, opener=opener, page_size=1)

    assert requested_pages == [1, 2]
    assert result.group_count == 1
    assert result.source_row_count == 2
    assert result.registration_count == 2
    with destination.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream, delimiter=";")) == [
            {
                "Year": "2010",
                "Status": "F",
                "Version_file": "v2",
                "MS": "DE",
                "Mk": "VW",
                "Cn": "GOLF",
                "TAN": "",
                "T": "1K",
                "Va": "A",
                "Ve": "1",
                "Ft": "petrol",
                "Registrations": "2",
                "SourceRows": "2",
            }
        ]


def test_release_inventory_covers_every_final_year_before_raw_2024() -> None:
    assert set(ANNUAL_RELEASES) == set(range(2010, 2024))
    assert ANNUAL_RELEASES[2010].version == "v2"
    assert ANNUAL_RELEASES[2019].version == "v20"
    assert ANNUAL_RELEASES[2020].table.endswith("co2cars_2020Fv22]")
    assert ANNUAL_RELEASES[2023].table.endswith("co2cars_2023Fv28]")
    assert ANNUAL_RELEASES[2020].expected_registrations == 11_709_621
    assert ANNUAL_RELEASES[2023].expected_registrations == 10_734_222


def test_acquisition_registration_total_excludes_rejected_identity_groups(
    tmp_path: Path,
) -> None:
    row = dict.fromkeys(ANNUAL_AGGREGATE_SCHEMA, "")
    row.update(
        Year=2010,
        Status="F",
        Version_file="v2",
        MS="DE",
        Mk="-",
        Cn="UNKNOWN",
        Registrations=9,
        SourceRows=2,
    )
    pages = [{"results": [row]}, {"results": []}]

    def opener(_: object, timeout: int) -> _Response:
        assert timeout == 120
        return _Response(pages.pop(0))

    release = replace(
        ANNUAL_RELEASES[2010],
        expected_source_rows=2,
        expected_accepted_rows=0,
        expected_rejected_rows=2,
        expected_registrations=0,
    )
    result = acquire_year(release, tmp_path / "rejected.csv", opener=opener)

    assert result.registration_count == 0

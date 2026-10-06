from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from icor.evidence.rdw_fleet_acquisition import API_URL, acquire_fleet

HEADER = "merk,handelsbenaming,cohort,vehicles\n"


class _Response:
    def __init__(self, body: str, url: str = API_URL) -> None:
        self._payload = body.encode()
        self._url = url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._payload if size < 0 else self._payload[:size]

    def geturl(self) -> str:
        return self._url


def _opener(pages: list[str], seen: list[dict[str, list[str]]] | None = None):
    def opener(request: object, timeout: int) -> _Response:
        assert timeout == 120
        url = request.full_url  # type: ignore[attr-defined]
        assert url.startswith(API_URL + "?")
        if seen is not None:
            seen.append(parse_qs(urlsplit(url).query))
        return _Response(pages.pop(0))

    return opener


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_pages_until_short_page_and_writes_sorted_canonical_csv(tmp_path: Path) -> None:
    seen: list[dict[str, list[str]]] = []
    pages = [
        HEADER + "VOLKSWAGEN,GOLF,2015,10\nVOLKSWAGEN,GOLF,2012,5\n",
        HEADER + "AUDI,A3,2020,7\n",
    ]
    destination = tmp_path / "rdw.csv"

    result = acquire_fleet(destination, opener=_opener(pages, seen), page_size=2)

    assert [query["$offset"] for query in seen] == [["0"], ["2"]]
    assert all(query["$limit"] == ["2"] for query in seen)
    assert "tenaamstellen_mogelijk='Ja'" in seen[0]["$where"][0]
    assert "voertuigsoort='Personenauto'" in seen[0]["$where"][0]
    assert destination.read_text(encoding="utf-8") == (
        HEADER + "AUDI,A3,2020,7\nVOLKSWAGEN,GOLF,2012,5\nVOLKSWAGEN,GOLF,2015,10\n"
    )
    assert result.group_count == 3
    assert result.accepted_group_count == 3
    assert result.rejected_group_count == 0
    assert result.vehicle_count == 22
    assert result.artifact_bytes == destination.stat().st_size
    assert result.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_output_is_deterministic_regardless_of_page_order(tmp_path: Path) -> None:
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    acquire_fleet(first, opener=_opener([HEADER + "B,X,2001,1\nA,Y,2002,2\n"]), page_size=5)
    acquire_fleet(second, opener=_opener([HEADER + "A,Y,2002,2\nB,X,2001,1\n"]), page_size=5)

    assert first.read_bytes() == second.read_bytes()


def test_normalises_whitespace_only_and_merges_resulting_duplicates(tmp_path: Path) -> None:
    body = HEADER + '" Mini  Cooper ","  One ",2010,3\n"Mini Cooper","One",2010,4\n'
    destination = tmp_path / "rdw.csv"

    result = acquire_fleet(destination, opener=_opener([body]), page_size=10)

    assert _rows(destination) == [
        {"merk": "Mini Cooper", "handelsbenaming": "One", "cohort": "2010", "vehicles": "7"}
    ]
    assert result.group_count == 1
    assert result.vehicle_count == 7


def test_counts_and_keeps_rejected_rows(tmp_path: Path) -> None:
    body = HEADER + (
        "A,B,,5\n"  # blank cohort
        ",B,2010,5\n"  # blank make
        "A,,2010,5\n"  # blank commercial name
        "A,B,2010,0\n"  # non-positive
        "A,B,2011,x\n"  # non-integer
        "A,B,2012,4\n"
    )
    destination = tmp_path / "rdw.csv"

    result = acquire_fleet(destination, opener=_opener([body]), page_size=10)

    assert result.group_count == 6
    assert result.accepted_group_count == 1
    assert result.rejected_group_count == 5
    assert result.vehicle_count == 4
    assert len(_rows(destination)) == 6


def test_rejects_unexpected_header_and_oversized_pages(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        acquire_fleet(
            tmp_path / "a.csv",
            opener=_opener(["merk,cohort\nA,2010\n"]),
            page_size=5,
        )
    with pytest.raises(ValueError, match="more rows"):
        acquire_fleet(
            tmp_path / "b.csv",
            opener=_opener([HEADER + "A,B,2010,1\nA,B,2011,1\n"]),
            page_size=1,
        )
    assert list(tmp_path.iterdir()) == []


def test_rejects_empty_result_and_invalid_page_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        acquire_fleet(tmp_path / "a.csv", opener=_opener([HEADER]), page_size=5)
    for size in (0, -1, 50_001):
        with pytest.raises(ValueError, match="page size"):
            acquire_fleet(tmp_path / "a.csv", opener=_opener([]), page_size=size)


def test_rejects_redirect_outside_allowlist_and_oversized_response(tmp_path: Path) -> None:
    def redirected(request: object, timeout: int) -> _Response:
        return _Response(HEADER + "A,B,2010,1\n", "https://example.com/x.csv")

    with pytest.raises(ValueError, match="allowlisted"):
        acquire_fleet(tmp_path / "a.csv", opener=redirected, page_size=5)

    def downgraded(request: object, timeout: int) -> _Response:
        return _Response(HEADER + "A,B,2010,1\n", "http://opendata.rdw.nl/x.csv")

    with pytest.raises(ValueError, match="allowlisted"):
        acquire_fleet(tmp_path / "a.csv", opener=downgraded, page_size=5)

    def huge(request: object, timeout: int) -> _Response:
        return _Response(HEADER + "A" * (65 * 1024 * 1024))

    with pytest.raises(ValueError, match="byte limit"):
        acquire_fleet(tmp_path / "a.csv", opener=huge, page_size=5)
    assert list(tmp_path.iterdir()) == []

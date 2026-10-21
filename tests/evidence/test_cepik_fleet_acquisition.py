from __future__ import annotations

import csv
import json
import ssl
import urllib.error
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from icor.evidence import cepik_fleet_acquisition as cepik
from icor.evidence.cepik_fleet_acquisition import (
    API_URL,
    PAGE_SIZE,
    Slice,
    crawl,
    plan_slices,
    write_aggregate,
)

_SLICE = Slice("SAMOCHÓD OSOBOWY", "08", date(2024, 1, 1), date(2025, 12, 31))


class _Response:
    def __init__(self, payload: bytes, url: str) -> None:
        self._payload = payload
        self._url = url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._payload if size < 0 else self._payload[:size]

    def geturl(self) -> str:
        return self._url


def _vehicle(
    identifier: str,
    make: str = "SKODA",
    model: str = "OCTAVIA",
    first: str | None = "2012-05-02",
    origin: str = "UŻYW. IMPORT INDYW",
) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "pojazdy",
        "attributes": {
            "marka": make,
            "model": model,
            "data-pierwszej-rejestracji": first,
            "pochodzenie-pojazdu": origin,
        },
    }


def _page(vehicles: list[dict[str, object]]) -> bytes:
    return json.dumps({"data": vehicles, "meta": {}, "links": {}}).encode()


def _opener(responses: list[object], seen: list[dict[str, list[str]]]):
    def opener(request: object, timeout: int) -> _Response:
        url = request.full_url  # type: ignore[attr-defined]
        assert url.startswith(API_URL + "?")
        seen.append(parse_qs(urlsplit(url).query))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _Response(response, url)  # type: ignore[arg-type]

    return opener


def _full_page(prefix: str) -> list[dict[str, object]]:
    return [_vehicle(f"{prefix}{index}") for index in range(PAGE_SIZE)]


def test_slices_cover_every_two_year_window_up_to_today() -> None:
    slices = plan_slices(date(2026, 10, 21))
    windows = sorted({(item.start, item.end) for item in slices})

    assert windows[0] == (date(1900, 1, 1), date(1901, 12, 31))
    assert windows[-1] == (date(2026, 1, 1), date(2026, 10, 21))
    assert all(
        later[0].toordinal() - earlier[1].toordinal() == 1
        for earlier, later in zip(windows, windows[1:], strict=False)
    )
    assert all((end.year - start.year) <= 1 for start, end in windows)
    assert len({item.name for item in slices}) == len(slices)


def test_a_later_start_keeps_the_same_windows_and_names() -> None:
    full = plan_slices(date(2026, 10, 21))
    recent = plan_slices(date(2026, 10, 21), from_year=2006)

    assert min(item.start for item in recent) == date(2006, 1, 1)
    assert set(recent) == {item for item in full if item.start.year >= 2006}
    with pytest.raises(ValueError, match="even years"):
        plan_slices(date(2026, 10, 21), from_year=2007)


def test_a_slice_pages_until_a_short_page_and_counts_each_vehicle_once(tmp_path: Path) -> None:
    seen: list[dict[str, list[str]]] = []
    first = _full_page("a")
    # The second page repeats one vehicle from the first, as an unstable order would.
    second = [
        _vehicle("a0"),
        _vehicle("b1", "TOYOTA", "YARIS", "2024-03-01", "NOWY ZAKUPIONY W KRAJU"),
    ]
    responses: list[object] = [_page(first), _page(second)]

    crawl([_SLICE], tmp_path, opener=_opener(responses, seen), sleep=lambda _s: None)

    assert [query["page"] for query in seen] == [["1"], ["2"]]
    query = seen[0]
    assert query["wojewodztwo"] == ["08"]
    assert query["data-od"] == ["20240101"] and query["data-do"] == ["20251231"]
    assert query["typ-daty"] == ["1"] and query["tylko-zarejestrowane"] == ["true"]
    assert query["filter[rodzaj-pojazdu]"] == ["SAMOCHÓD OSOBOWY"]
    document = json.loads((tmp_path / f"{_SLICE.name}.json").read_text(encoding="utf-8"))
    assert document["rows"] == PAGE_SIZE + 1
    assert document["duplicates"] == 1
    assert document["groups"] == [
        ["SKODA", "OCTAVIA", "2012", "UŻYW. IMPORT INDYW", PAGE_SIZE],
        ["TOYOTA", "YARIS", "2024", "NOWY ZAKUPIONY W KRAJU", 1],
    ]


def test_truncated_and_throttled_pages_are_retried(tmp_path: Path) -> None:
    seen: list[dict[str, list[str]]] = []
    throttled = urllib.error.HTTPError(API_URL, 429, "busy", None, None)  # type: ignore[arg-type]
    good = _page([_vehicle("x1")])
    responses: list[object] = [
        throttled,
        good[: len(good) // 2],
        b'{"errors": [{"error-code": "API-ERR-100001"}]}',
        good,
    ]
    sleeps: list[float] = []

    crawl([_SLICE], tmp_path, opener=_opener(responses, seen), sleep=sleeps.append)

    assert len(seen) == 4 and len(sleeps) == 3
    document = json.loads((tmp_path / f"{_SLICE.name}.json").read_text(encoding="utf-8"))
    assert document["rows"] == 1


def test_a_slice_that_keeps_failing_is_not_written(tmp_path: Path) -> None:
    seen: list[dict[str, list[str]]] = []
    responses: list[object] = [b"not json"] * 3

    later = Slice("SAMOCHÓD OSOBOWY", "14", date(2024, 1, 1), date(2025, 12, 31))
    responses.append(_page([_vehicle("y1")]))
    messages: list[str] = []

    with pytest.raises(RuntimeError, match="1 CEPiK slices failed"):
        crawl(
            [_SLICE, later],
            tmp_path,
            opener=_opener(responses, seen),
            sleep=lambda _s: None,
            max_attempts=3,
            workers=1,
            log=messages.append,
        )

    assert not (tmp_path / f"{_SLICE.name}.json").exists()
    assert (tmp_path / f"{later.name}.json").exists()
    assert any("after 3 attempts" in message for message in messages)


def test_a_non_retryable_status_is_raised(tmp_path: Path) -> None:
    seen: list[dict[str, list[str]]] = []
    responses: list[object] = [urllib.error.HTTPError(API_URL, 400, "bad", None, None)]  # type: ignore[arg-type]

    with pytest.raises(urllib.error.HTTPError):
        crawl([_SLICE], tmp_path, opener=_opener(responses, seen), sleep=lambda _s: None)


def test_finished_slices_are_skipped_on_resume(tmp_path: Path) -> None:
    seen: list[dict[str, list[str]]] = []
    crawl(
        [_SLICE], tmp_path, opener=_opener([_page([_vehicle("x1")])], seen), sleep=lambda _s: None
    )

    crawl([_SLICE], tmp_path, opener=_opener([], seen), sleep=lambda _s: None)

    assert len(seen) == 1


def test_a_response_from_another_host_is_refused(tmp_path: Path) -> None:
    def opener(request: object, timeout: int) -> _Response:
        return _Response(_page([]), "https://example.org/pojazdy")

    with pytest.raises(ValueError, match="allowlisted"):
        crawl([_SLICE], tmp_path, opener=opener, sleep=lambda _s: None)


def test_the_aggregate_merges_slices_and_keeps_counts_only(tmp_path: Path) -> None:
    other = Slice("SAMOCHÓD OSOBOWY", "14", date(2024, 1, 1), date(2025, 12, 31))
    seen: list[dict[str, list[str]]] = []
    responses: list[object] = [
        _page([_vehicle("a1"), _vehicle("a2", first=None)]),
        _page(
            [
                _vehicle("b1"),
                _vehicle("b2", "FIAT", " 126 P ", "1990-01-01", "NOWY ZAKUPIONY W KRAJU"),
            ]
        ),
    ]
    crawl(
        [_SLICE, other],
        tmp_path / "state",
        opener=_opener(responses, seen),
        sleep=lambda _s: None,
        workers=1,
    )

    result = write_aggregate(tmp_path / "state", [_SLICE, other], tmp_path / "cepik.csv")

    with (tmp_path / "cepik.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows == [
        list(cepik.SCHEMA),
        ["FIAT", "126 P", "1990", "NOWY ZAKUPIONY W KRAJU", "1"],
        ["SKODA", "OCTAVIA", "", "UŻYW. IMPORT INDYW", "1"],
        ["SKODA", "OCTAVIA", "2012", "UŻYW. IMPORT INDYW", "2"],
    ]
    assert (result.slice_count, result.group_count, result.vehicle_count) == (2, 3, 4)
    assert "a1" not in (tmp_path / "cepik.csv").read_text(encoding="utf-8")


def test_the_aggregate_refuses_an_unfinished_crawl(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unfinished"):
        write_aggregate(tmp_path, [_SLICE], tmp_path / "cepik.csv")


def test_a_refused_certificate_is_raised_not_retried(tmp_path: Path) -> None:
    seen: list[dict[str, list[str]]] = []
    refused = urllib.error.URLError(ssl.SSLCertVerificationError("self-signed"))
    sleeps: list[float] = []

    with pytest.raises(urllib.error.URLError):
        crawl([_SLICE], tmp_path, opener=_opener([refused], seen), sleep=sleeps.append)

    assert sleeps == []


def test_the_default_opener_refuses_finite_field_dhe() -> None:
    context = cepik._tls_context()

    assert context.verify_mode is ssl.CERT_REQUIRED
    assert all(
        "DHE-" not in item["name"] or "ECDHE" in item["name"] for item in context.get_ciphers()
    )

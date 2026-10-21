"""Resumable acquisition of Poland's registered passenger cars from the CEPiK API.

CEPiK publishes no current bulk file (its CSVs stop at 2022-04-17), so the stock is
read from the paged ``/pojazdy`` API: one voivodeship and at most two years of
first-registration-in-Poland dates per query, 500 vehicles a page. Each slice
(vehicle kind × voivodeship × window) is crawled in full, de-duplicated on the API's
opaque vehicle id, aggregated to counts and written atomically before the next one
starts, so an interrupted crawl resumes at the first unfinished slice. No row-level
field leaves memory: the slice files hold make, model, the year of first registration
anywhere, the vehicle's origin class and a count.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import ssl
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.parse import urlencode, urlsplit

import certifi

API_URL = "https://api.cepik.gov.pl/pojazdy"
_ALLOWED_HOST = "api.cepik.gov.pl"
PAGE_SIZE = 500
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
#: The sixteen voivodeship TERYT codes of the ``wojewodztwa`` dictionary. Vehicles
#: CEPiK holds without a voivodeship ("XX") cannot be queried through the API.
VOIVODESHIPS = (
    "02",
    "04",
    "06",
    "08",
    "10",
    "12",
    "14",
    "16",
    "18",
    "20",
    "22",
    "24",
    "26",
    "28",
    "30",
    "32",
)
#: Every spelling of a passenger car in the ``rodzaj-pojazdu`` dictionary
#: (2026-10-21). Crew-cab vans ("SAM.CIĘŻAROWO-OSOBOWY" and variants) are goods
#: vehicles and stay out.
PASSENGER_KINDS = (
    "SAMOCHÓD OSOBOWY",
    "OSOBOWY",
    "SAMOCHODY OSOBOWE",
    "SAMOCHOD OSOBOWY",
    "SAMOCHED OSOBOWY",
    "SAMOCHAD OSOBOWY",
    "SAM.OSOBOWY",
)
FIELDS = ("marka", "model", "data-pierwszej-rejestracji", "pochodzenie-pojazdu")
SCHEMA = ("marka", "model", "cohort", "origin", "vehicles")
EARLIEST_WINDOW_YEAR = 1900
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _tls_context() -> ssl.SSLContext:
    # CEPiK offers finite-field DHE with a key OpenSSL now refuses as too small;
    # its ECDHE suites are fine. The certificate is still verified against certifi.
    context = ssl.create_default_context(cafile=certifi.where())
    context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL")
    return context


def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
    return urllib.request.urlopen(request, timeout=timeout, context=_tls_context())  # type: ignore[no-any-return]


class _Response(Protocol):
    status: int

    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> object: ...

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Slice:
    kind: str
    voivodeship: str
    start: date
    end: date

    @property
    def name(self) -> str:
        kind = hashlib.sha256(self.kind.encode("utf-8")).hexdigest()[:8]
        return f"{kind}-{self.voivodeship}-{self.start:%Y%m%d}-{self.end:%Y%m%d}"


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    path: Path
    artifact_bytes: int
    sha256: str
    slice_count: int
    group_count: int
    vehicle_count: int
    duplicate_count: int
    first_slice_finished: str
    last_slice_finished: str


def plan_slices(today: date, kinds: Iterable[str] = PASSENGER_KINDS) -> tuple[Slice, ...]:
    """Every kind × voivodeship × two-year window from 1900 to ``today``."""

    unknown = set(kinds) - set(PASSENGER_KINDS)
    if unknown:
        raise ValueError(f"not a CEPiK passenger-car kind: {sorted(unknown)}")

    windows = []
    year = EARLIEST_WINDOW_YEAR
    while year <= today.year:
        end = min(date(year + 1, 12, 31), today)
        windows.append((date(year, 1, 1), end))
        year += 2
    return tuple(
        Slice(kind, voivodeship, start, end)
        for kind in kinds
        for voivodeship in VOIVODESHIPS
        for start, end in windows
    )


def crawl(
    slices: Iterable[Slice],
    state_dir: Path,
    *,
    opener: Callable[..., _Response] = urlopen,
    workers: int = 2,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = 12,
    log: Callable[[str], None] = lambda _message: None,
) -> None:
    """Crawl every slice not yet finished in ``state_dir``; finished ones are skipped."""

    state_dir.mkdir(parents=True, exist_ok=True)
    pending = [item for item in slices if not (state_dir / f"{item.name}.json").exists()]
    log(f"{len(pending)} slices pending")

    def run(item: Slice) -> None:
        counts, rows, duplicates = _crawl_slice(
            item, opener=opener, sleep=sleep, max_attempts=max_attempts, log=log
        )
        _write_json(
            state_dir / f"{item.name}.json",
            {
                "kind": item.kind,
                "voivodeship": item.voivodeship,
                "start": item.start.isoformat(),
                "end": item.end.isoformat(),
                "finished_at": date.today().isoformat(),
                "rows": rows,
                "duplicates": duplicates,
                "groups": [[*key, count] for key, count in sorted(counts.items())],
            },
        )
        log(f"{item.name} rows={rows} duplicates={duplicates}")

    failures: list[str] = []

    def attempt(item: Slice) -> None:
        # A slice that keeps failing is left unfinished for the next run instead of
        # stopping the slices queued behind it.
        try:
            run(item)
        except RuntimeError as error:
            failures.append(item.name)
            log(f"{item.name} failed: {error}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(attempt, pending):
            pass
    if failures:
        raise RuntimeError(f"{len(failures)} CEPiK slices failed; rerun to resume")


def _crawl_slice(
    item: Slice,
    *,
    opener: Callable[..., _Response],
    sleep: Callable[[float], None],
    max_attempts: int,
    log: Callable[[str], None],
) -> tuple[dict[tuple[str, str, str, str], int], int, int]:
    counts: dict[tuple[str, str, str, str], int] = {}
    seen: set[str] = set()
    rows = duplicates = 0
    page = 1
    while True:
        records = _fetch_page(
            item, page, opener=opener, sleep=sleep, max_attempts=max_attempts, log=log
        )
        for identifier, attributes in records:
            if identifier in seen:
                duplicates += 1
                continue
            seen.add(identifier)
            rows += 1
            first = attributes.get("data-pierwszej-rejestracji") or ""
            key = (
                _collapse(attributes.get("marka") or ""),
                _collapse(attributes.get("model") or ""),
                first[:4] if len(first) >= 4 and first[:4].isdigit() else "",
                _collapse(attributes.get("pochodzenie-pojazdu") or ""),
            )
            counts[key] = counts.get(key, 0) + 1
        if len(records) < PAGE_SIZE:
            return counts, rows, duplicates
        page += 1


def page_url(item: Slice, page: int) -> str:
    query = {
        "wojewodztwo": item.voivodeship,
        "data-od": f"{item.start:%Y%m%d}",
        "data-do": f"{item.end:%Y%m%d}",
        "typ-daty": "1",
        "tylko-zarejestrowane": "true",
        "limit": str(PAGE_SIZE),
        "page": str(page),
        "sort": "id",
        "fields": ",".join(FIELDS),
        "filter[rodzaj-pojazdu]": item.kind,
    }
    return f"{API_URL}?{urlencode(query)}"


def _fetch_page(
    item: Slice,
    page: int,
    *,
    opener: Callable[..., _Response],
    sleep: Callable[[float], None],
    max_attempts: int,
    log: Callable[[str], None] = lambda _message: None,
) -> list[tuple[str, dict[str, str | None]]]:
    url = page_url(item, page)
    _require_allowlisted(url)
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return _parse_page(_get(url, opener), item.kind)
        except urllib.error.HTTPError as error:
            if error.code not in _RETRYABLE_STATUS:
                raise
            last_error = error
        except urllib.error.URLError as error:
            # A refused certificate or handshake will not heal by waiting.
            if isinstance(error.reason, ssl.SSLError):
                raise
            last_error = error
        except (TimeoutError, ConnectionError, _MalformedPage) as error:
            last_error = error
        log(f"{item.name} page {page} attempt {attempt + 1}: {str(last_error)[:160]}")
        sleep(min(120.0, 2.0 * 2**attempt))
    raise RuntimeError(
        f"CEPiK page failed after {max_attempts} attempts: {item.name} page {page}"
    ) from last_error


class _MalformedPage(ValueError):
    pass


def _get(url: str, opener: Callable[..., _Response]) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ICOR-evidence/1"})
    with opener(request, timeout=180) as response:
        _require_allowlisted(response.geturl())
        payload = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise ValueError("CEPiK API response exceeds the byte limit")
    return payload


def _parse_page(payload: bytes, kind: str) -> list[tuple[str, dict[str, str | None]]]:
    # The API sometimes ends a 200 response early; a page that is not complete JSON
    # is retried rather than read as the last page.
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _MalformedPage("CEPiK API response is not complete JSON") from error
    if not isinstance(document, dict):
        raise _MalformedPage("CEPiK API response is not an object")
    if "errors" in document:
        raise _MalformedPage(f"CEPiK API returned errors: {document['errors']!r}"[:500])
    data = document.get("data")
    if not isinstance(data, list) or len(data) > PAGE_SIZE:
        raise _MalformedPage("CEPiK API response has no usable data list")
    records = []
    for entry in data:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise _MalformedPage("CEPiK API vehicle entry is malformed")
        attributes = entry.get("attributes")
        if not isinstance(attributes, dict) or not set(FIELDS) <= set(attributes):
            raise _MalformedPage("CEPiK API vehicle attributes are incomplete")
        records.append((entry["id"], {field: attributes[field] for field in FIELDS}))
    return records


def write_aggregate(
    state_dir: Path, slices: Iterable[Slice], destination: Path
) -> AcquisitionResult:
    """Merge every finished slice into one sorted canonical CSV of counts only."""

    merged: dict[tuple[str, str, str, str], int] = {}
    slice_count = duplicates = 0
    finished: list[str] = []
    for item in slices:
        path = state_dir / f"{item.name}.json"
        if not path.exists():
            raise ValueError(f"CEPiK slice is unfinished: {item.name}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if (document["kind"], document["voivodeship"], document["start"], document["end"]) != (
            item.kind,
            item.voivodeship,
            item.start.isoformat(),
            item.end.isoformat(),
        ):
            raise ValueError(f"CEPiK slice file does not match its slice: {item.name}")
        slice_count += 1
        duplicates += document["duplicates"]
        finished.append(document["finished_at"])
        for make, model, cohort, origin, count in document["groups"]:
            key = (make, model, cohort, origin)
            merged[key] = merged.get(key, 0) + int(count)
    records = sorted((*key, str(count)) for key, count in merged.items())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=destination.parent
        ) as output:
            temporary = Path(output.name)
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(SCHEMA)
            writer.writerows(records)
            output.flush()
            os.fsync(output.fileno())
        artifact_bytes = temporary.stat().st_size
        sha256 = _sha256(temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return AcquisitionResult(
        destination,
        artifact_bytes,
        sha256,
        slice_count,
        len(records),
        sum(merged.values()),
        duplicates,
        min(finished) if finished else "",
        max(finished) if finished else "",
    )


def _write_json(path: Path, document: object) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as output:
            temporary = Path(output.name)
            json.dump(document, output, ensure_ascii=False, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _require_allowlisted(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != _ALLOWED_HOST:
        raise ValueError("CEPiK API URL is outside its allowlisted origin")


def _collapse(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_slice_documents(state_dir: Path) -> Iterator[dict[str, object]]:
    for path in sorted(state_dir.glob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))

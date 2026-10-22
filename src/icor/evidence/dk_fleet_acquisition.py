"""Counts-only acquisition of Denmark's registered passenger cars from the Motorstyrelsen DMR.

Source: the weekly "Statistikudtræk" of the Danish Motor Register (DMR), published by
Motorstyrelsen as a ZIP on an FTP server that answers anonymous logins:
``ftp://5.44.137.84/ESStatistikListeModtag/ESStatistikListeModtag-YYYYMMDD-HHMMSS.zip``.
The ZIP is about 7.1 GB (ZIP64, deflate) and holds one entry, ``ESStatistikListeModtag.xml``
of about 129 GB with one ``<Statistik>`` element per vehicle and leasing period.

Terms: the extract's page (https://motorst.dk/erhverv/motorregistret-for-virksomheder/
faa-adgang-til-motorregistret/andre-adgange) says use is free ("Det koster ikke noget at
anvende udtrækket") and states no licence. **No explicit reuse licence is stated.** The page
also shows an FTP login; it is not used, because the anonymous login lists and serves the
same files.

The file is never stored. It is streamed over FTP (binary ``RETR``, resumed with ``REST``
after a dropped connection) through ``HashingReader``, so the byte count and SHA-256 are
those of the exact bytes served. The ZIP local header is parsed here, the deflate stream is
inflated with ``zlib.decompressobj(-15)`` and checked against the central directory's CRC-32
and sizes at its end, and the XML goes through ``xml.parsers.expat`` handlers that follow
only the paths below. A reader thread downloads and inflates while the main thread parses.

What is read, in the XML namespace ``http://skat.dk/dmr/2007/05/31/``, from
``ESStatistikListeModtag_I/StatistikSamling/Statistik``:

- ``KoeretoejArtNummer``: category; 1 (Personbil) is kept, every number and name is profiled;
- ``KoeretoejOplysningGrundStruktur/KoeretoejOplysningStatus``: the strict filter is
  ``Registreret``; ``KoeretoejRegistreringStatus`` and the other values are profiled;
- make ``.../KoeretoejBetegnelseStruktur/KoeretoejMaerkeTypeNavn`` and model
  ``.../KoeretoejBetegnelseStruktur/Model/KoeretoejModelTypeNavn``;
- the cohort: the year of ``KoeretoejOplysningFoersteRegistreringDato`` (``YYYY-MM-DD+TZ``);
- profiled only: ``KoeretoejOplysningOprettetUdFra``, the sparse ``KoeretoejOplysningModelAar``,
  ``KoeretoejRegistreringStatusDato``, ``KoeretoejOplysningStatusDato`` and the inspection
  type and date, to measure whether used imports carry their foreign first-registration date.

Never output: the VIN (``KoeretoejOplysningStelNummer``), the plate
(``RegistreringNummerNummer``), ``KoeretoejIdent`` (it is only hashed in memory, to count
repeated vehicles), ``KoeretoejOplysningTypeAnmeldelseNummer``, every comment, and any owner or
user element. The extract holds none of the last kind; the profile lists every element name
seen so that this can be checked.

Verified on the 2026-10-20 extract (14,120,965 records, 129,464,303,405 inflated bytes):

- A vehicle is repeated once per leasing period (1,030,747 of 9,364,908 passenger-car
  records), with identical status, origin, labels and dates; the first record of each
  ``KoeretoejIdent`` is kept, matched in memory only.
- The strict count is 2,973,464 registered passenger cars, 98.5% of Eurostat's 3,018,026.
- Used imports carry their foreign first-registration date. Of the 182,459 registered cars
  created from a registration inspection (``Registreringssyn``) that have a customs
  inspection date, 93.2% were first registered more than 180 days before it, and 1.0%
  within 30 days; the cohort is therefore the year of first registration anywhere.
"""

from __future__ import annotations

import contextlib
import ftplib
import json
import os
import queue
import re
import struct
import threading
import time
import zlib
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO
from urllib.parse import urlsplit
from xml.parsers import expat

from icor.evidence.normalization import normalize_vehicle_label
from icor.evidence.register_aggregate import (
    HashingReader,
    cohort_of,
    collapse,
    write_aggregate,
)

FTP_HOST = "5.44.137.84"
DIRECTORY = "/ESStatistikListeModtag/"
DIRECTORY_URL = f"ftp://{FTP_HOST}{DIRECTORY}"
TERMS_URL = (
    "https://motorst.dk/erhverv/motorregistret-for-virksomheder/"
    "faa-adgang-til-motorregistret/andre-adgange"
)
_ALLOWED_HOST = FTP_HOST
SNAPSHOT_NAME = re.compile(r"ESStatistikListeModtag-(\d{8})-(\d{6})\.zip\Z", re.ASCII)
ENTRY_NAME = "ESStatistikListeModtag.xml"
NAMESPACE = "http://skat.dk/dmr/2007/05/31/"
ROOT = "ESStatistikListeModtag_I"
SAMLING = "StatistikSamling"
RECORD = "Statistik"

PASSENGER_CAR_NUMBER = "1"
PASSENGER_CAR_NAME = "Personbil"
REGISTERED = "Registreret"
PROFILE_FIRST_COHORT = 2000
TOP_MAKES = 40
#: Every element name that could name a vehicle's owner or user. None exists in the extract.
_OWNER_LIKE = re.compile(
    r"ejer|bruger|cpr|cvr|adress|foedsel|koen|kunde|virksomhed|person|firma|holder",
    re.IGNORECASE,
)
_INSPECTION_TYPES = frozenset(
    {"RegistreringsSyn", "RegistreringssynToldsyn", "PeriodiskSyn", "KontrolSyn", "MOT"}
)
_FIRST_REGISTRATION = re.compile(r"(\d{4})-\d{2}-\d{2}(?:Z|[+-]\d{2}:\d{2})?\Z", re.ASCII)
#: Buckets of the days from the first-registration date to a later date of the same record.
_GAP_EDGES = (-181, -1, 30, 180, 365, 730, 1825)
_GAP_BUCKETS = (
    "<=-181",
    "-180..-1",
    "0..30",
    "31..180",
    "181..365",
    "366..730",
    "731..1825",
    ">1825",
)
_EARLIER_THAN_180_DAYS = frozenset(_GAP_BUCKETS[4:])
_MODEL_YEAR_CLIP = (-5, 15)

_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")
_CENTRAL_HEADER = struct.Struct("<4sHHHHHHIIIHHHHHII")
_READ_SIZE = 256 * 1024
_MAX_OUTPUT = 16 * 1024 * 1024
_MAX_TAIL = 1024 * 1024
_ZIP64_MARKER = 0xFFFFFFFF
_SCHEMA_PROBE_ROWS = 20_000
_QUEUE_DEPTH = 6


@dataclass(frozen=True, slots=True)
class Snapshot:
    name: str
    snapshot_date: date
    size: int
    modified: datetime | None


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    path: Path
    artifact_bytes: int
    sha256: str
    profile_path: Path
    source_name: str
    source_date: str
    source_bytes: int
    source_sha256: str
    #: Every ``<Statistik>`` record, then those that are passenger cars (a vehicle with
    #: several leasing periods has several), then the repeats of an already counted vehicle.
    source_rows: int
    category_rows: int
    duplicate_category_rows: int
    #: Distinct registered passenger cars: the vehicles in the aggregate.
    registered_rows: int
    group_count: int
    vehicles: int
    masked_vehicles: int
    blank_make_vehicles: int
    blank_model_vehicles: int
    blank_cohort_vehicles: int


def _require_allowlisted(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "ftp" or parts.netloc != _ALLOWED_HOST:
        raise ValueError("DMR source URL is outside its allowlisted origin")


def connect_ftp(host: str = FTP_HOST, *, timeout: float = 60.0) -> ftplib.FTP:
    """An anonymous binary session in the extract's directory; any other host is refused."""

    _require_allowlisted(f"ftp://{host}{DIRECTORY}")
    ftp = ftplib.FTP(timeout=timeout)
    try:
        ftp.connect(host, 21)
        ftp.login()
        ftp.voidcmd("TYPE I")
        ftp.cwd(DIRECTORY)
    except BaseException:
        ftp.close()
        raise
    return ftp


def newest_snapshot(ftp: Any, *, name: str | None = None) -> Snapshot:
    """The newest ``ESStatistikListeModtag-YYYYMMDD-HHMMSS.zip``, or the pinned ``name``."""

    if getattr(ftp, "host", None) != _ALLOWED_HOST:
        raise ValueError("DMR source URL is outside its allowlisted origin")
    candidates: list[tuple[datetime, str]] = []
    for listed in ftp.nlst():
        base = listed.rsplit("/", 1)[-1]
        match = SNAPSHOT_NAME.match(base)
        if match is None:
            continue
        try:
            stamp = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            continue
        candidates.append((stamp, base))
    # A listing leaves the session in ASCII mode, where the server refuses REST.
    ftp.voidcmd("TYPE I")
    if name is not None:
        if name not in {base for _, base in candidates}:
            raise ValueError("the pinned DMR snapshot is not in the directory")
        chosen = next(item for item in candidates if item[1] == name)
    elif candidates:
        chosen = max(candidates)
    else:
        raise ValueError("the DMR directory lists no ESStatistikListeModtag snapshot")
    size = ftp.size(chosen[1])
    if not isinstance(size, int) or size <= 0:
        raise ValueError("the DMR server did not report the snapshot's size")
    return Snapshot(chosen[1], chosen[0].date(), size, _modified(ftp, chosen[1]))


def _modified(ftp: Any, name: str) -> datetime | None:
    try:
        reply = ftp.sendcmd(f"MDTM {name}")
        return datetime.strptime(reply.split()[-1][:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except (ftplib.Error, OSError, ValueError, IndexError):
        return None


class FtpFile:
    """A readable binary ``RETR`` that reconnects and resumes with ``REST`` when it drops."""

    def __init__(
        self,
        connect: Callable[[], Any],
        name: str,
        size: int,
        *,
        max_failures: int = 20,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = lambda _message: None,
    ) -> None:
        self._connect = connect
        self._name = name
        self._size = size
        self._max_failures = max_failures
        self._sleep = sleep
        self._log = log
        self._position = 0
        self._ftp: Any = None
        self._data: Any = None
        self.reconnects = 0

    def read(self, size: int = -1) -> bytes:
        want = 1 << 20 if size is None or size < 0 else size
        failures = 0
        while True:
            if self._position >= self._size:
                self._close()
                return b""
            try:
                if self._data is None:
                    self._open()
                # One byte past the end exposes a server that sends more than it listed.
                chunk = self._data.recv(min(want, self._size - self._position + 1))
                if not chunk:
                    raise ConnectionError("the data connection closed before the file ended")
                self._position += len(chunk)
                return bytes(chunk)
            except (OSError, EOFError, ftplib.Error) as error:
                self._close()
                failures += 1
                if failures > self._max_failures:
                    raise RuntimeError(
                        f"DMR download failed {failures} times at byte {self._position}"
                    ) from error
                self.reconnects += 1
                self._log(
                    f"download interrupted at byte {self._position}: "
                    f"{type(error).__name__}; reconnecting"
                )
                self._sleep(min(60.0, 2.0 * 2**failures))

    def readable(self) -> bool:
        return True

    def _open(self) -> None:
        self._ftp = self._connect()
        if getattr(self._ftp, "host", None) != _ALLOWED_HOST:
            raise ValueError("DMR source URL is outside its allowlisted origin")
        rest = self._position or None
        self._data = self._ftp.transfercmd(f"RETR {self._name}", rest=rest)

    def _close(self) -> None:
        data, ftp = self._data, self._ftp
        finished = self._position >= self._size
        self._data = self._ftp = None
        steps: list[Callable[[], object]] = []
        if data is not None:
            steps.append(data.close)
        if ftp is not None:
            if finished and data is not None:
                steps += [ftp.voidresp, ftp.quit]
            steps.append(ftp.close)
        for step in steps:
            with contextlib.suppress(OSError, EOFError, ftplib.Error):
                step()


class DeflatedEntry:
    """The single deflated entry of a ZIP read as a stream, with its integrity checked.

    The local file header is parsed from the stream (so no seeking is needed), the entry is
    inflated until its own deflate stream ends, and the rest of the file is read to prove it
    is the central directory of this one entry: same name, CRC-32 and sizes.
    """

    def __init__(self, source: BinaryIO, name: str = ENTRY_NAME) -> None:
        self._source = source
        self._name = name
        self.inflated_bytes = 0
        self.compressed_bytes = 0
        self.crc32 = 0

    def chunks(self) -> Iterator[bytes]:
        source = self._source
        header = _read_exact(source, _LOCAL_HEADER.size)
        signature, _, flags, method, _, _, _, _, _, name_length, extra_length = (
            _LOCAL_HEADER.unpack(header)
        )
        if signature != b"PK\x03\x04":
            raise ValueError("DMR download is not a ZIP file")
        if flags & 0x41:
            raise ValueError("DMR ZIP entry is encrypted")
        if method != 8:
            raise ValueError("DMR ZIP entry is not deflate-compressed")
        name = _read_exact(source, name_length)
        _read_exact(source, extra_length)
        if name != self._name.encode("ascii"):
            raise ValueError("DMR ZIP entry has an unexpected name")

        decompressor = zlib.decompressobj(-15)
        fed = 0
        while not decompressor.eof:
            block = source.read(_READ_SIZE)
            if not block:
                raise ValueError("DMR ZIP entry ends before its deflate stream does")
            fed += len(block)
            pending = block
            while True:
                data = decompressor.decompress(pending, _MAX_OUTPUT)
                if data:
                    self.crc32 = zlib.crc32(data, self.crc32)
                    self.inflated_bytes += len(data)
                    yield data
                if decompressor.eof:
                    break
                pending = decompressor.unconsumed_tail
                if not pending and len(data) < _MAX_OUTPUT:
                    break
        self.compressed_bytes = fed - len(decompressor.unused_data)
        tail = bytearray(decompressor.unused_data)
        while block := source.read(_READ_SIZE):
            tail += block
            if len(tail) > _MAX_TAIL:
                raise ValueError("DMR ZIP holds more than its one entry")
        self._verify(bytes(tail))

    def _verify(self, tail: bytes) -> None:
        expected = self._name.encode("ascii")
        index = tail.find(b"PK\x01\x02")
        while index != -1:
            fixed = tail[index : index + _CENTRAL_HEADER.size]
            if len(fixed) == _CENTRAL_HEADER.size:
                (
                    _,
                    _,
                    _,
                    _,
                    method,
                    _,
                    _,
                    crc,
                    compressed,
                    uncompressed,
                    name_length,
                    extra_length,
                    comment_length,
                    *_,
                ) = _CENTRAL_HEADER.unpack(fixed)
                start = index + _CENTRAL_HEADER.size
                if tail[start : start + name_length] == expected:
                    extra = tail[start + name_length : start + name_length + extra_length]
                    after = tail[start + name_length + extra_length + comment_length :]
                    self._check_directory(method, crc, compressed, uncompressed, extra, after[:4])
                    return
            index = tail.find(b"PK\x01\x02", index + 1)
        raise ValueError("DMR ZIP central directory does not describe the entry")

    def _check_directory(
        self,
        method: int,
        crc: int,
        compressed: int,
        uncompressed: int,
        extra: bytes,
        following: bytes,
    ) -> None:
        if method != 8:
            raise ValueError("DMR ZIP central directory disagrees on the compression method")
        if following not in (b"PK\x06\x06", b"PK\x05\x06"):
            raise ValueError("DMR ZIP holds more than its one entry")
        if crc != self.crc32:
            raise ValueError("DMR ZIP entry fails its CRC-32 check")
        cursor = 0
        zip64 = _zip64_values(extra)
        sizes: list[int] = []
        for value in (uncompressed, compressed):
            if value == _ZIP64_MARKER:
                if cursor + 1 > len(zip64):
                    raise ValueError("DMR ZIP central directory lacks its ZIP64 sizes")
                value = zip64[cursor]
                cursor += 1
            sizes.append(value)
        # A writer that truncated a size to 32 bits is still compared modulo 2**32.
        modulus = 1 << 32
        if sizes[0] % modulus != self.inflated_bytes % modulus or (
            sizes[0] >= modulus and sizes[0] != self.inflated_bytes
        ):
            raise ValueError("DMR ZIP entry has an unexpected inflated size")
        if sizes[1] % modulus != self.compressed_bytes % modulus or (
            sizes[1] >= modulus and sizes[1] != self.compressed_bytes
        ):
            raise ValueError("DMR ZIP entry has an unexpected compressed size")


def _zip64_values(extra: bytes) -> list[int]:
    position = 0
    while position + 4 <= len(extra):
        identifier, length = struct.unpack_from("<HH", extra, position)
        body = extra[position + 4 : position + 4 + length]
        if identifier == 1:
            return [
                struct.unpack_from("<Q", body, offset)[0] for offset in range(0, len(body) - 7, 8)
            ]
        position += 4 + length
    return []


def _read_exact(source: BinaryIO, count: int) -> bytes:
    parts: list[bytes] = []
    remaining = count
    while remaining:
        chunk = source.read(remaining)
        if not chunk:
            raise ValueError("DMR ZIP ends inside a header")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


_END = object()


def prefetched(chunks: Iterable[bytes], depth: int = _QUEUE_DEPTH) -> Iterator[bytes]:
    """Run ``chunks`` in a thread, so that downloading overlaps with parsing."""

    buffer: queue.Queue[object] = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(item: object) -> bool:
        while not stop.is_set():
            try:
                buffer.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for chunk in chunks:
                if not put(chunk):
                    return
            put(_END)
        except BaseException as error:  # handed to the consumer, which re-raises it
            put(error)

    thread = threading.Thread(target=produce, name="dmr-download", daemon=True)
    thread.start()
    try:
        while True:
            item = buffer.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]
    finally:
        stop.set()


def _ordinal(value: str) -> int | None:
    try:
        return date.fromisoformat(value[:10]).toordinal()
    except ValueError:
        return None


def _gap_bucket(days: int) -> str:
    return _GAP_BUCKETS[bisect_left(_GAP_EDGES, days)]


def _model_year_bucket(gap: int) -> str:
    low, high = _MODEL_YEAR_CLIP
    if gap <= low:
        return f"<={low}"
    if gap >= high:
        return f">={high}"
    return str(gap)


class RecordCounter:
    """Everything counted from the vehicles' fields; nothing identifying is kept."""

    def __init__(self) -> None:
        self.source_rows = 0
        self.category_rows = 0
        self.category_vehicles = 0
        self.registered_rows = 0
        self.categories: dict[tuple[str, str], int] = defaultdict(int)
        self.status: dict[str, int] = defaultdict(int)
        self.registration_status: dict[str, int] = defaultdict(int)
        self.status_cross: dict[tuple[str, str], int] = defaultdict(int)
        self.cohort_status: dict[tuple[str, str], int] = defaultdict(int)
        self.origin_status: dict[tuple[str, str], int] = defaultdict(int)
        self.origin_cohort: dict[tuple[str, str], int] = defaultdict(int)
        self.counts: dict[tuple[str, str, str], int] = defaultdict(int)
        self.evidence: dict[tuple[str, str, str, str], int] = defaultdict(int)
        self.present: dict[str, int] = defaultdict(int)
        self.unparseable_first_registration = 0
        self.repeated_fields: dict[str, int] = defaultdict(int)
        self.element_names: set[str] = set()
        #: One DMR record per leasing period repeats a vehicle: ident -> fingerprint of its fields.
        self._seen: dict[int, int] = {}
        self.duplicate_category_rows = 0
        self.conflicting_duplicate_rows = 0
        self.category_rows_without_ident = 0

    def add(self, record: dict[str, str]) -> None:
        self.source_rows += 1
        get = record.get
        number = get("KoeretoejArtNummer", "").strip()
        name = get("KoeretoejArtNavn", "").strip()
        self.categories[(number, name)] += 1
        if number != PASSENGER_CAR_NUMBER:
            return
        self.category_rows += 1
        status = get("KoeretoejOplysningStatus", "").strip()
        registration_status = get("KoeretoejRegistreringStatus", "").strip()
        origin = get("KoeretoejOplysningOprettetUdFra", "").strip()
        make = get("KoeretoejMaerkeTypeNavn")
        model = get("KoeretoejModelTypeNavn")
        first = get("KoeretoejOplysningFoersteRegistreringDato", "").strip()
        ident = get("KoeretoejIdent", "").strip()
        if ident:
            key = (
                int(ident)
                if ident.isascii() and ident.isdigit() and len(ident) < 19
                else hash(ident)
            )
            fingerprint = hash((status, registration_status, origin, make, model, first))
            previous = self._seen.get(key)
            if previous is not None:
                self.duplicate_category_rows += 1
                if previous != fingerprint:
                    self.conflicting_duplicate_rows += 1
                return
            self._seen[key] = fingerprint
        else:
            self.category_rows_without_ident += 1
        self.category_vehicles += 1
        self.status[status] += 1
        self.registration_status[registration_status] += 1
        self.status_cross[(status, registration_status)] += 1
        self.origin_status[(origin, status)] += 1
        matched = _FIRST_REGISTRATION.match(first)
        cohort = cohort_of(matched.group(1)) if matched else ""
        if first and not matched:
            self.unparseable_first_registration += 1
        self.cohort_status[(cohort, status)] += 1
        present = self.present
        present["status"] += status != ""
        present["first_registration"] += first != ""
        present["make"] += make is not None
        present["model"] += model is not None
        strict = status == REGISTERED
        self._evidence(record, origin, cohort, first, strict)
        if not strict:
            return
        self.registered_rows += 1
        self.origin_cohort[(origin, cohort)] += 1
        self.counts[(collapse(make), collapse(model), cohort)] += 1

    def _evidence(
        self, record: dict[str, str], origin: str, cohort: str, first: str, strict: bool
    ) -> None:
        get = record.get
        first_ordinal = _ordinal(first) if cohort else None
        entries: list[tuple[str, str]] = []
        model_year = get("KoeretoejOplysningModelAar", "").strip()
        if not model_year:
            entries.append(("model_year_gap", "no_model_year"))
        elif first_ordinal is None or cohort_of(model_year) == "":
            entries.append(("model_year_gap", "unusable"))
        else:
            entries.append(("model_year_gap", _model_year_bucket(int(cohort) - int(model_year))))
        inspection_type = get("SynResultatSynsType", "").strip()
        if not inspection_type:
            inspection = "inspection:none"
        elif inspection_type in _INSPECTION_TYPES:
            inspection = f"inspection:{inspection_type}"
        else:
            inspection = "inspection:other"
        for proxy, value in (
            ("registration_status_date", get("KoeretoejRegistreringStatusDato", "")),
            ("information_status_date", get("KoeretoejOplysningStatusDato", "")),
            (inspection, get("SynResultatSynsDato", "")),
        ):
            if first_ordinal is None:
                bucket = "unusable_first_registration"
            elif not value.strip():
                bucket = "missing"
            else:
                later = _ordinal(value.strip())
                bucket = "unparseable" if later is None else _gap_bucket(later - first_ordinal)
            entries.append((f"first_registration_vs:{proxy}", bucket))
        evidence = self.evidence
        for scope in ("registered", "all_statuses") if strict else ("all_statuses",):
            evidence[(scope, origin, "vehicles", "")] += 1
            for kind, bucket in entries:
                evidence[(scope, origin, kind, bucket)] += 1

    def require_schema(self, *, final: bool) -> None:
        if self.category_rows == 0:
            if final:
                raise ValueError("DMR schema drift: no Personbil row (KoeretoejArtNummer 1)")
            return
        missing = [
            field
            for field in ("status", "first_registration", "make", "model")
            if self.present[field] == 0
        ]
        if missing:
            raise ValueError(
                "DMR schema drift: no passenger car holds " + ", ".join(sorted(missing))
            )
        names = {name for (number, name) in self.categories if number == PASSENGER_CAR_NUMBER}
        if names != {PASSENGER_CAR_NAME}:
            raise ValueError("DMR schema drift: category 1 is no longer named Personbil")


#: Element -> (field, depth below the document root counted from 1, parent element).
_FIELD_PATHS: dict[str, tuple[str, int, str]] = {
    "KoeretoejIdent": ("KoeretoejIdent", 4, RECORD),
    "KoeretoejArtNummer": ("KoeretoejArtNummer", 4, RECORD),
    "KoeretoejArtNavn": ("KoeretoejArtNavn", 4, RECORD),
    "KoeretoejRegistreringStatus": ("KoeretoejRegistreringStatus", 4, RECORD),
    "KoeretoejRegistreringStatusDato": ("KoeretoejRegistreringStatusDato", 4, RECORD),
    "SynResultatSynsType": ("SynResultatSynsType", 5, "SynResultatStruktur"),
    "SynResultatSynsDato": ("SynResultatSynsDato", 5, "SynResultatStruktur"),
    "KoeretoejOplysningOprettetUdFra": (
        "KoeretoejOplysningOprettetUdFra",
        5,
        "KoeretoejOplysningGrundStruktur",
    ),
    "KoeretoejOplysningStatus": (
        "KoeretoejOplysningStatus",
        5,
        "KoeretoejOplysningGrundStruktur",
    ),
    "KoeretoejOplysningStatusDato": (
        "KoeretoejOplysningStatusDato",
        5,
        "KoeretoejOplysningGrundStruktur",
    ),
    "KoeretoejOplysningFoersteRegistreringDato": (
        "KoeretoejOplysningFoersteRegistreringDato",
        5,
        "KoeretoejOplysningGrundStruktur",
    ),
    "KoeretoejOplysningModelAar": (
        "KoeretoejOplysningModelAar",
        5,
        "KoeretoejOplysningGrundStruktur",
    ),
    "KoeretoejMaerkeTypeNavn": ("KoeretoejMaerkeTypeNavn", 6, "KoeretoejBetegnelseStruktur"),
    "KoeretoejModelTypeNavn": ("KoeretoejModelTypeNavn", 7, "Model"),
}


def count_records(
    chunks: Iterable[bytes],
    counter: RecordCounter,
    *,
    on_chunk: Callable[[RecordCounter], None] | None = None,
) -> None:
    """Parse the XML as it arrives, adding each vehicle to ``counter``."""

    parser = expat.ParserCreate()
    parser.buffer_text = True
    stack: list[str] = []
    record: dict[str, str] = {}
    pieces: list[str] = []
    wanted: dict[str, tuple[str, int, str]] = {}
    state = {"capture": "", "target": "", "record": "", "probed": False}
    names_add = counter.element_names.add

    def start(name: str, attributes: dict[str, str]) -> None:
        stack.append(name)
        names_add(name)
        depth = len(stack)
        if depth == 1:
            _begin_document(name, attributes)
            return
        spec = wanted.get(name)
        if spec is None or spec[1] != depth or stack[-2] != spec[2]:
            return
        field = spec[0]
        if field == RECORD:
            record.clear()
            return
        if field in record:
            counter.repeated_fields[field] += 1
            return
        state["capture"] = field
        state["target"] = name
        pieces.clear()
        parser.CharacterDataHandler = pieces.append

    def end(name: str) -> None:
        stack.pop()
        if state["capture"]:
            if name == state["target"]:
                record[state["capture"]] = "".join(pieces)
                state["capture"] = ""
                parser.CharacterDataHandler = None
            return
        if name == state["record"] and len(stack) == 2:
            counter.add(record)
            if counter.source_rows == _SCHEMA_PROBE_ROWS:
                counter.require_schema(final=False)

    def _begin_document(name: str, attributes: dict[str, str]) -> None:
        if not name.endswith(ROOT):
            raise ValueError("DMR schema drift: unexpected document root")
        prefix = name[: -len(ROOT)]
        declaration = "xmlns" if prefix == "" else "xmlns:" + prefix.rstrip(":")
        if attributes.get(declaration) != NAMESPACE:
            raise ValueError("DMR schema drift: unexpected XML namespace")
        for element, (field, depth, parent) in _FIELD_PATHS.items():
            wanted[prefix + element] = (field, depth, prefix + parent)
        wanted[prefix + RECORD] = (RECORD, 3, prefix + SAMLING)
        state["record"] = prefix + RECORD

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    try:
        for chunk in chunks:
            parser.Parse(chunk, False)
            if on_chunk is not None:
                on_chunk(counter)
        parser.Parse(b"", True)
    except expat.ExpatError as error:
        raise ValueError(f"DMR XML is not well formed: {error}") from error
    if not state["record"]:
        raise ValueError("DMR schema drift: the XML has no document root")
    counter.element_names = {
        name.split(":", 1)[-1] if ":" in name else name for name in counter.element_names
    }
    counter.require_schema(final=True)


def aggregate_zip(
    source: BinaryIO,
    *,
    limit: int | None = None,
    on_progress: Callable[[RecordCounter, HashingReader, DeflatedEntry], None] | None = None,
    interval: float = 180.0,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[RecordCounter, HashingReader, DeflatedEntry]:
    """Stream a DMR ZIP into counts; the ZIP is hashed as served and nothing is stored."""

    reader = HashingReader(source, limit=limit)  # type: ignore[arg-type]
    entry = DeflatedEntry(reader)  # type: ignore[arg-type]
    counter = RecordCounter()
    next_report = clock() + interval

    def report(_: RecordCounter) -> None:
        nonlocal next_report
        if on_progress is not None and clock() >= next_report:
            next_report = clock() + interval
            on_progress(counter, reader, entry)

    count_records(prefetched(entry.chunks()), counter, on_chunk=report)
    return counter, reader, entry


def acquire_fleet(
    destination: Path,
    *,
    connect: Callable[[], Any] = connect_ftp,
    name: str | None = None,
    log: Callable[[str], None] = lambda _message: None,
    interval: float = 180.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> AcquisitionResult:
    """Stream the newest extract into a sorted canonical CSV of counts and a profile.

    ``destination`` may contain ``{date}``, which becomes the snapshot's YYYYMMDD.
    """

    started = datetime.now(UTC)
    ftp = connect()
    try:
        snapshot = newest_snapshot(ftp, name=name)
    finally:
        ftp.close()
    stamp = f"{snapshot.snapshot_date:%Y%m%d}"
    destination = Path(str(destination).replace("{date}", stamp))
    log(f"snapshot {snapshot.name} ({snapshot.size} bytes, modified {snapshot.modified})")
    remote = FtpFile(connect, snapshot.name, snapshot.size, sleep=sleep, log=log)

    def progress(counter: RecordCounter, reader: HashingReader, entry: DeflatedEntry) -> None:
        log(
            f"bytes={reader.bytes_read} ({100 * reader.bytes_read / snapshot.size:.1f}%) "
            f"inflated={entry.inflated_bytes} records={counter.source_rows} "
            f"passenger_cars={counter.category_rows} registered={counter.registered_rows} "
            f"reconnects={remote.reconnects}"
        )

    counter, reader, entry = aggregate_zip(
        remote,  # type: ignore[arg-type]
        limit=snapshot.size,
        on_progress=progress,
        interval=interval,
        clock=clock,
    )
    if reader.bytes_read != snapshot.size:
        raise ValueError("DMR download is incomplete")
    log(
        f"stream complete: bytes={reader.bytes_read} sha256={reader.sha256} "
        f"inflated={entry.inflated_bytes} records={counter.source_rows}"
    )
    aggregate = write_aggregate(counter.counts, destination)
    stats = _label_stats(counter.counts)
    if aggregate.vehicles != counter.registered_rows:
        raise ValueError("DMR aggregate does not hold every registered passenger car")
    profile_path = destination.with_name(destination.name + ".profile.json")
    profile = build_profile(
        counter,
        snapshot=snapshot,
        source_bytes=reader.bytes_read,
        source_sha256=reader.sha256,
        entry=entry,
        retrieved_at=started,
        aggregate_name=destination.name,
        aggregate_sha256=aggregate.sha256,
        aggregate_bytes=aggregate.artifact_bytes,
        group_count=aggregate.group_count,
        stats=stats,
        reconnects=remote.reconnects,
    )
    _write_json(profile_path, profile)
    log(f"wrote {destination} sha256={aggregate.sha256} groups={aggregate.group_count}")
    return AcquisitionResult(
        destination,
        aggregate.artifact_bytes,
        aggregate.sha256,
        profile_path,
        snapshot.name,
        snapshot.snapshot_date.isoformat(),
        reader.bytes_read,
        reader.sha256,
        counter.source_rows,
        counter.category_rows,
        counter.duplicate_category_rows,
        counter.registered_rows,
        aggregate.group_count,
        aggregate.vehicles,
        stats["masked_vehicles"],
        stats["blank_make_vehicles"],
        stats["blank_model_vehicles"],
        stats["blank_cohort_vehicles"],
    )


def _is_masked(label: str) -> bool:
    return "¡" in label or (label != "" and set(label) <= {"*"})


def _label_stats(counts: dict[tuple[str, str, str], int]) -> dict[str, int]:
    stats = {
        "masked_vehicles": 0,
        "blank_make_vehicles": 0,
        "blank_model_vehicles": 0,
        "blank_cohort_vehicles": 0,
        "vin_like_label_vehicles": 0,
    }
    vin = re.compile(r"[A-HJ-NPR-Z0-9]{17}\Z", re.ASCII)
    for (make, model, cohort), vehicles in counts.items():
        if _is_masked(make) or _is_masked(model):
            stats["masked_vehicles"] += vehicles
        if normalize_vehicle_label(make) is None:
            stats["blank_make_vehicles"] += vehicles
        if normalize_vehicle_label(model) is None:
            stats["blank_model_vehicles"] += vehicles
        if cohort == "":
            stats["blank_cohort_vehicles"] += vehicles
        if vin.match(make) or vin.match(model):
            stats["vin_like_label_vehicles"] += vehicles
    return stats


def _shown(value: str) -> str:
    return value if value else "(missing)"


def build_profile(
    counter: RecordCounter,
    *,
    snapshot: Snapshot,
    source_bytes: int,
    source_sha256: str,
    entry: DeflatedEntry,
    retrieved_at: datetime,
    aggregate_name: str,
    aggregate_sha256: str,
    aggregate_bytes: int,
    group_count: int,
    stats: dict[str, int],
    reconnects: int,
) -> dict[str, object]:
    """Counts only: no row-level field, label of one vehicle or identifier is in it."""

    cohort_totals: dict[str, int] = defaultdict(int)
    make_totals: dict[str, int] = defaultdict(int)
    for (make, _, cohort), vehicles in counter.counts.items():
        cohort_totals[cohort] += vehicles
        make_totals[make] += vehicles
    by_cohort_status: dict[str, dict[str, int]] = {}
    for (cohort, status), vehicles in sorted(counter.cohort_status.items()):
        if cohort == "" or int(cohort) >= PROFILE_FIRST_COHORT:
            by_cohort_status.setdefault(_shown(cohort), {})[_shown(status)] = vehicles
    top_makes = sorted(make_totals.items(), key=lambda item: (-item[1], item[0]))[:TOP_MAKES]
    return {
        "source": {
            "country": "dk",
            "publisher": "Motorstyrelsen, Danish Motor Register (DMR) Statistikudtræk",
            "url": f"{DIRECTORY_URL}{snapshot.name}",
            "terms_url": TERMS_URL,
            "terms": (
                'Use is free ("Det koster ikke noget at anvende udtrækket"); no explicit '
                "reuse licence is stated."
            ),
            "access": "anonymous FTP; the login shown on the terms page is not used",
            "file_name": snapshot.name,
            "file_date": snapshot.snapshot_date.isoformat(),
            "last_modified_utc": snapshot.modified.isoformat() if snapshot.modified else None,
            "retrieved_at_utc": retrieved_at.isoformat(),
            "bytes": source_bytes,
            "sha256": source_sha256,
            "zip_entry": ENTRY_NAME,
            "entry_inflated_bytes": entry.inflated_bytes,
            "entry_compressed_bytes": entry.compressed_bytes,
            "entry_crc32": f"{entry.crc32:08x}",
            "reconnects": reconnects,
        },
        "aggregate": {
            "name": aggregate_name,
            "bytes": aggregate_bytes,
            "sha256": aggregate_sha256,
            "groups": group_count,
            "vehicles": counter.registered_rows,
        },
        "filters": {
            "category": f"KoeretoejArtNummer == {PASSENGER_CAR_NUMBER} ({PASSENGER_CAR_NAME})",
            "status": f"KoeretoejOplysningStatus == {REGISTERED}",
            "funnel": {
                "source_rows": counter.source_rows,
                "after_category_filter": counter.category_rows,
                "repeated_vehicle_rows_dropped": counter.duplicate_category_rows,
                "after_category_filter_unique_vehicles": counter.category_vehicles,
                "after_status_filter": counter.registered_rows,
            },
            "repeated_vehicles": {
                "note": (
                    "The extract repeats a vehicle once per leasing period; the first record "
                    "of each KoeretoejIdent is kept, matched in memory and never written."
                ),
                "rows_dropped": counter.duplicate_category_rows,
                "dropped_rows_that_differ_in_status_origin_labels_or_date": (
                    counter.conflicting_duplicate_rows
                ),
                "passenger_car_rows_without_ident": counter.category_rows_without_ident,
            },
        },
        "categories_in_source": {
            f"{_shown(number)} {_shown(name)}": vehicles
            for (number, name), vehicles in sorted(counter.categories.items())
        },
        "passenger_cars_by_status": {
            _shown(status): vehicles for status, vehicles in sorted(counter.status.items())
        },
        "passenger_cars_by_registration_status": {
            _shown(status): vehicles
            for status, vehicles in sorted(counter.registration_status.items())
        },
        "passenger_cars_by_status_and_registration_status": {
            f"{_shown(status)} | {_shown(registration)}": vehicles
            for (status, registration), vehicles in sorted(counter.status_cross.items())
        },
        "passenger_cars_by_origin_and_status": {
            f"{_shown(origin)} | {_shown(status)}": vehicles
            for (origin, status), vehicles in sorted(counter.origin_status.items())
        },
        "registered_by_cohort": {
            _shown(cohort): vehicles for cohort, vehicles in sorted(cohort_totals.items())
        },
        f"passenger_cars_by_cohort_and_status_from_{PROFILE_FIRST_COHORT}": by_cohort_status,
        "national_first_registration_year": {
            "available": False,
            "note": (
                "The extract holds one first-registration date and no separate Danish one; "
                "see used_import_evidence for how far it is the foreign date."
            ),
        },
        "used_import_evidence": _evidence_profile(counter),
        "registered_by_origin_and_cohort": {
            origin_label: {
                _shown(cohort): vehicles
                for (origin, cohort), vehicles in sorted(counter.origin_cohort.items())
                if origin == origin_label
            }
            for origin_label in sorted({origin for origin, _ in counter.origin_cohort})
        },
        "labels": {
            "masked_vehicles": stats["masked_vehicles"],
            "blank_make_vehicles": stats["blank_make_vehicles"],
            "blank_model_vehicles": stats["blank_model_vehicles"],
            "blank_cohort_vehicles": stats["blank_cohort_vehicles"],
            "vin_like_label_vehicles": stats["vin_like_label_vehicles"],
            "note": (
                "Blank means empty or a missing marker such as '-', as the loader treats it; "
                "the register does not mask labels."
            ),
        },
        "first_registration_values_not_matching_yyyy_mm_dd": (
            counter.unparseable_first_registration
        ),
        "field_presence_among_passenger_cars": dict(sorted(counter.present.items())),
        "fields_seen_twice_in_one_vehicle": dict(sorted(counter.repeated_fields.items())),
        "element_names_seen": sorted(counter.element_names),
        "owner_like_element_names": sorted(
            name for name in counter.element_names if _OWNER_LIKE.search(name)
        ),
        "top_makes_registered": [{"make": make, "vehicles": n} for make, n in top_makes],
    }


def _evidence_profile(counter: RecordCounter) -> dict[str, object]:
    """Per scope and origin: model-year gap and first-registration-versus-entry date gaps."""

    scopes: dict[str, dict[str, dict[str, object]]] = {}
    for (scope, origin, kind, bucket), vehicles in sorted(counter.evidence.items()):
        entry = scopes.setdefault(scope, {}).setdefault(_shown(origin), {})
        if kind == "vehicles":
            entry["vehicles"] = vehicles
        else:
            group = entry.setdefault(kind, {})
            assert isinstance(group, dict)
            group[bucket] = vehicles
    for origins in scopes.values():
        for entry in origins.values():
            gaps = entry.get("model_year_gap", {})
            assert isinstance(gaps, dict)
            entry["model_year_summary"] = _model_year_summary(gaps)
            for kind in [key for key in entry if str(key).startswith("first_registration_vs:")]:
                buckets = entry[kind]
                assert isinstance(buckets, dict)
                usable = sum(n for bucket, n in buckets.items() if bucket in _GAP_BUCKETS)
                entry[kind] = {
                    "buckets": buckets,
                    "usable": usable,
                    "first_registration_more_than_180_days_earlier": sum(
                        n for bucket, n in buckets.items() if bucket in _EARLIER_THAN_180_DAYS
                    ),
                }
    return {
        "definitions": {
            "scopes": (
                "registered = passenger cars with the strict status; all_statuses = passenger "
                "cars of every status"
            ),
            "model_year_gap": (
                "first-registration cohort year minus KoeretoejOplysningModelAar, clipped to "
                f"{_MODEL_YEAR_CLIP[0]}..{_MODEL_YEAR_CLIP[1]}"
            ),
            "first_registration_vs": (
                "days from the first-registration date to the named later date, in buckets; "
                "a positive gap means the first registration is earlier. inspection:<type> is "
                "the date of the vehicle's recorded inspection of that type"
            ),
        },
        "scopes": scopes,
    }


def _model_year_summary(gaps: dict[str, int]) -> dict[str, int]:
    summary = {"with_model_year": 0, "cohort_2_or_more_years_after": 0, "cohort_1_year_after": 0}
    summary.update({"cohort_same_year": 0, "cohort_before_model_year": 0})
    for bucket, vehicles in gaps.items():
        if bucket in ("no_model_year", "unusable"):
            continue
        low, high = _MODEL_YEAR_CLIP
        gap = low if bucket.startswith("<=") else high if bucket.startswith(">=") else int(bucket)
        summary["with_model_year"] += vehicles
        if gap >= 2:
            summary["cohort_2_or_more_years_after"] += vehicles
        elif gap == 1:
            summary["cohort_1_year_after"] += vehicles
        elif gap == 0:
            summary["cohort_same_year"] += vehicles
        else:
            summary["cohort_before_model_year"] += vehicles
    return summary


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=path.parent
        ) as output:
            temporary = Path(output.name)
            json.dump(document, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

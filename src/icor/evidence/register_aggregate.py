"""The counts-only aggregate every national register acquisition writes.

A register is read once, as a stream, and only make, model, the year of first
registration anywhere and a count leave it: no VIN, plate, owner or row-level
field is ever written. The aggregate is a sorted UTF-8 CSV, so the same counts
always give the same bytes and checksum.
"""

from __future__ import annotations

import csv
import hashlib
import os
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

SCHEMA = ("make", "model", "cohort", "vehicles")


@dataclass(frozen=True, slots=True)
class AggregateFile:
    path: Path
    artifact_bytes: int
    sha256: str
    group_count: int
    vehicles: int


def collapse(value: str | None) -> str:
    """A label as the register wrote it, in NFC with runs of whitespace as one space."""

    return " ".join(unicodedata.normalize("NFC", value or "").split())


def cohort_of(year: str | None) -> str:
    """A four-digit year, or "" when the register gives none that is usable."""

    value = (year or "").strip()
    return value if len(value) == 4 and value.isascii() and value.isdigit() else ""


def write_aggregate(counts: Mapping[tuple[str, str, str], int], destination: Path) -> AggregateFile:
    """Write ``(make, model, cohort) -> vehicles`` atomically as the canonical CSV."""

    if not counts:
        raise ValueError("the register holds no passenger cars")
    if any(count <= 0 for count in counts.values()):
        raise ValueError("register counts must be positive")
    records = sorted(((*key, str(count)) for key, count in counts.items()), key=_sort_key)
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
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return AggregateFile(destination, artifact_bytes, digest, len(records), sum(counts.values()))


class HashingReader:
    """Pass bytes through from a stream while counting and hashing them."""

    def __init__(self, stream: BinaryIO, *, limit: int | None = None) -> None:
        self._stream = stream
        self._limit = limit
        self._digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        self.bytes_read += len(chunk)
        if self._limit is not None and self.bytes_read > self._limit:
            raise ValueError("the register download exceeds its byte limit")
        self._digest.update(chunk)
        return chunk

    def readable(self) -> bool:
        return True

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sort_key(record: tuple[str, ...]) -> tuple[object, ...]:
    make, model, cohort, vehicles = record
    numeric = cohort.isascii() and cohort.isdigit()
    return (make, model, 0 if numeric else 1, int(cohort) if numeric else 0, cohort, vehicles)

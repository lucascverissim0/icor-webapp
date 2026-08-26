"""Canonical serialization helpers for reproducible evidence artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from pathlib import Path, PurePath


def canonical_json_bytes(value: object) -> bytes:
    """Serialize supported contract values to canonical UTF-8 JSON bytes."""
    normalized = _normalize(value)
    document = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return (document + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    """Return the lower-case SHA-256 digest of a file's exact bytes."""
    digest = sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: object) -> object:
    if isinstance(value, float):
        raise TypeError("float values are forbidden in canonical JSON")
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _normalize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("canonical JSON datetimes must be UTC")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("canonical JSON mapping keys must be strings")
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, bool | int | str):
        return value
    raise TypeError(f"{type(value).__name__} is unsupported in canonical JSON")

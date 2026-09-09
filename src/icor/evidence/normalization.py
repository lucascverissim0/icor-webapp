"""Conservative source-label normalization and deterministic identifiers."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_MISSING_MARKERS = frozenset({"", "-", "[c]", "[x]", "[z]", "n/a"})
_ID_PREFIX = re.compile(r"[a-z0-9][a-z0-9._-]{0,42}\Z")
_DISPLAY_ACRONYMS = {
    "bmw": "BMW",
    "byd": "BYD",
    "daf": "DAF",
    "ds": "DS",
    "man": "MAN",
    "mg": "MG",
    "vw": "VW",
}


def normalize_vehicle_label(value: str) -> str | None:
    """Remove presentation noise while retaining the publisher's lexical identity."""

    if type(value) is not str:
        raise ValueError("vehicle label must be text")
    normalized = " ".join(unicodedata.normalize("NFC", value).split()).casefold()
    return None if normalized in _MISSING_MARKERS else normalized


def source_vehicle_display_label(value: str) -> str:
    """Normalize whitespace and casing without changing lexical identity."""

    if type(value) is not str or not value.strip():
        raise ValueError("vehicle display label must be text")
    compact = " ".join(unicodedata.normalize("NFC", value).split())
    if not (compact.islower() or compact.isupper()):
        return compact
    return " ".join(
        _DISPLAY_ACRONYMS.get(token.casefold(), token.title())
        for token in compact.split()
    )


def stable_evidence_id(prefix: str, *parts: str) -> str:
    """Build an identifier-safe digest with length-prefixed, order-preserving input."""

    if type(prefix) is not str or _ID_PREFIX.fullmatch(prefix) is None:
        raise ValueError("evidence identifier prefix is invalid")
    if not parts or any(type(part) is not str for part in parts):
        raise ValueError("evidence identifier parts must be text")
    digest = hashlib.sha256()
    for part in parts:
        encoded = unicodedata.normalize("NFC", part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"{prefix}-{digest.hexdigest()}"[:80]

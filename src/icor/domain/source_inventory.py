"""Immutable discovery outcomes for official vehicle evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from re import fullmatch


class InventoryStatus(StrEnum):
    ACQUIRED_VALIDATED = "acquired_validated"
    ACQUIRED_QUARANTINED = "acquired_quarantined"
    UNAVAILABLE = "unavailable"
    SUPERSEDED = "superseded"
    EXCLUDED_INCOMPATIBLE = "excluded_incompatible"
    EXCLUDED_LICENCE = "excluded_licence"
    PENDING_REVIEW = "pending_review"


@dataclass(frozen=True, slots=True)
class SourceInventoryEntry:
    source_key: str
    period_start: date
    period_end: date
    status: InventoryStatus
    release_id: str | None
    revision_state: str
    licence_status: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.source_key) is not str
            or fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", self.source_key) is None
        ):
            raise ValueError("source key is invalid")
        if type(self.period_start) is not date or type(self.period_end) is not date:
            raise ValueError("inventory periods must be dates")
        if self.period_start > self.period_end:
            raise ValueError("inventory coverage is reversed")
        if not isinstance(self.status, InventoryStatus):
            raise ValueError("inventory status is unsupported")
        for value, label in (
            (self.revision_state, "revision state"),
            (self.licence_status, "licence status"),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} is required")
        if not isinstance(self.reason_codes, tuple) or any(
            type(code) is not str or fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", code) is None
            for code in self.reason_codes
        ):
            raise ValueError("inventory reason code is invalid")
        if self.status is InventoryStatus.ACQUIRED_VALIDATED:
            if type(self.release_id) is not str or not self.release_id.strip():
                raise ValueError("validated inventory outcome requires a release ID")
        elif not self.reason_codes:
            raise ValueError("unacquired inventory outcome requires a reason code")

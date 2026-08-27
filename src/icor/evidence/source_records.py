"""Source-neutral parser result contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedRelease[Record]:
    records: tuple[Record, ...]
    raw_count: int
    accepted_count: int
    rejected_count: int
    quarantined_count: int
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (
            self.raw_count,
            self.accepted_count,
            self.rejected_count,
            self.quarantined_count,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("record counts must be non-negative integers")
        if self.accepted_count != len(self.records):
            raise ValueError("accepted count must match parsed records")
        if self.accepted_count + self.rejected_count + self.quarantined_count != self.raw_count:
            raise ValueError("record counts must reconcile")
        if not isinstance(self.warnings, tuple) or any(
            type(warning) is not str or not warning.strip() for warning in self.warnings
        ):
            raise ValueError("warnings must be nonblank text")

"""Canonical persistence for the official-source discovery inventory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from icor.domain.source_inventory import InventoryStatus, SourceInventoryEntry
from icor.evidence.serialization import canonical_json_bytes


@dataclass(frozen=True, slots=True)
class SourceInventory:
    _entries: tuple[SourceInventoryEntry, ...]

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(self._entries, key=lambda item: (item.source_key, item.period_start))
        )
        identities = [(item.source_key, item.period_start, item.period_end) for item in ordered]
        if len(identities) != len(set(identities)):
            raise ValueError("source inventory contains a duplicate source period")
        object.__setattr__(self, "_entries", ordered)

    def entries(self) -> tuple[SourceInventoryEntry, ...]:
        return self._entries

    def write(self, path: Path) -> None:
        path.write_bytes(canonical_json_bytes({"entries": self._entries, "schema_version": 1}))

    @classmethod
    def load(cls, path: Path) -> SourceInventory:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if set(payload) != {"entries", "schema_version"} or payload["schema_version"] != 1:
                raise ValueError
            entries = tuple(
                SourceInventoryEntry(
                    source_key=item["source_key"],
                    period_start=date.fromisoformat(item["period_start"]),
                    period_end=date.fromisoformat(item["period_end"]),
                    status=InventoryStatus(item["status"]),
                    release_id=item["release_id"],
                    revision_state=item["revision_state"],
                    licence_status=item["licence_status"],
                    reason_codes=tuple(item["reason_codes"]),
                )
                for item in payload["entries"]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("source inventory is invalid") from error
        return cls(entries)

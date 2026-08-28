"""Validated in-memory view of versioned generation evidence."""

from __future__ import annotations

from dataclasses import dataclass

from icor.domain.generations import GenerationEntry


@dataclass(frozen=True, slots=True)
class GenerationRegistry:
    entries: tuple[GenerationEntry, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.entries, key=lambda item: item.generation_id))
        identifiers = tuple(item.generation_id for item in ordered)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("generation registry contains a duplicate generation ID")
        object.__setattr__(self, "entries", ordered)

    def candidates(
        self, canonical_vehicle_id: str, market: str, registration_year: int
    ) -> tuple[GenerationEntry, ...]:
        return tuple(
            item
            for item in self.entries
            if item.canonical_vehicle_id == canonical_vehicle_id
            and item.market == market
            and item.start_month.year <= registration_year
            and (item.end_month is None or item.end_month.year >= registration_year)
        )

    def get(self, generation_id: str) -> GenerationEntry | None:
        return next(
            (item for item in self.entries if item.generation_id == generation_id),
            None,
        )

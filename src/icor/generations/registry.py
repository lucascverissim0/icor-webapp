"""Validated in-memory view of versioned generation evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

from icor.domain.generations import GenerationEntry


@dataclass(frozen=True, slots=True)
class GenerationRegistry:
    entries: tuple[GenerationEntry, ...]
    _entries_by_vehicle_market: dict[
        tuple[str, str], tuple[GenerationEntry, ...]
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.entries, key=lambda item: item.generation_id))
        identifiers = tuple(item.generation_id for item in ordered)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("generation registry contains a duplicate generation ID")
        object.__setattr__(self, "entries", ordered)
        grouped: dict[tuple[str, str], list[GenerationEntry]] = {}
        for item in ordered:
            grouped.setdefault(
                (item.canonical_vehicle_id, item.market), []
            ).append(item)
        object.__setattr__(
            self,
            "_entries_by_vehicle_market",
            {key: tuple(values) for key, values in grouped.items()},
        )

    def candidates(
        self, canonical_vehicle_id: str, market: str, registration_year: int
    ) -> tuple[GenerationEntry, ...]:
        return tuple(
            item
            for item in self._entries_by_vehicle_market.get(
                (canonical_vehicle_id, market), ()
            )
            if item.start_month.year <= registration_year
            and (item.end_month is None or item.end_month.year >= registration_year)
        )

    def get(self, generation_id: str) -> GenerationEntry | None:
        return next(
            (item for item in self.entries if item.generation_id == generation_id),
            None,
        )

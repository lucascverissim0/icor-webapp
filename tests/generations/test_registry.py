from __future__ import annotations

from datetime import date

from icor.generations.registry import GenerationRegistry


class _CountingEntry:
    def __init__(self, generation_id: str, vehicle_id: str) -> None:
        self.generation_id = generation_id
        self._vehicle_id = vehicle_id
        self.market = "EU27"
        self.start_month = date(2010, 1, 1)
        self.end_month = date(2031, 12, 1)
        self.vehicle_reads = 0

    @property
    def canonical_vehicle_id(self) -> str:
        self.vehicle_reads += 1
        return self._vehicle_id


def test_candidates_do_not_rescan_unrelated_generations() -> None:
    unrelated = tuple(
        _CountingEntry(f"generation-{index:04d}", f"vehicle-{index:04d}")
        for index in range(1_000)
    )
    selected = _CountingEntry("generation-selected", "vehicle-selected")
    registry = GenerationRegistry((*unrelated, selected))  # type: ignore[arg-type]
    for entry in (*unrelated, selected):
        entry.vehicle_reads = 0

    assert registry.candidates("vehicle-selected", "EU27", 2024) == (selected,)
    assert sum(entry.vehicle_reads for entry in unrelated) == 0

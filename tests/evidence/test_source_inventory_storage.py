from datetime import date
from pathlib import Path

from icor.domain.source_inventory import InventoryStatus, SourceInventoryEntry
from icor.evidence.source_inventory import SourceInventory


def test_inventory_round_trip_is_canonical_and_sorted(tmp_path: Path) -> None:
    inventory = SourceInventory(
        (
            SourceInventoryEntry(
                source_key="uk-veh0160-gb",
                period_start=date(2001, 1, 1),
                period_end=date(2025, 12, 31),
                status=InventoryStatus.ACQUIRED_VALIDATED,
                release_id="uk-dft-veh0160-gb-2025-final-20260713",
                revision_state="final",
                licence_status="permitted",
                reason_codes=(),
            ),
            SourceInventoryEntry(
                source_key="eea-2025-provisional",
                period_start=date(2025, 1, 1),
                period_end=date(2025, 12, 31),
                status=InventoryStatus.EXCLUDED_INCOMPATIBLE,
                release_id=None,
                revision_state="provisional",
                licence_status="permitted",
                reason_codes=("final_release_required",),
            ),
        )
    )

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    inventory.write(first)
    SourceInventory.load(first).write(second)

    assert first.read_bytes() == second.read_bytes()
    assert [entry.source_key for entry in SourceInventory.load(first).entries()] == [
        "eea-2025-provisional",
        "uk-veh0160-gb",
    ]


def test_inventory_rejects_duplicate_source_periods() -> None:
    entry = SourceInventoryEntry(
        source_key="eea-2024-final",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        status=InventoryStatus.ACQUIRED_VALIDATED,
        release_id="eea-co2cars-2024-final-v30-r1",
        revision_state="final",
        licence_status="permitted",
        reason_codes=(),
    )

    try:
        SourceInventory((entry, entry))
    except ValueError as error:
        assert str(error) == "source inventory contains a duplicate source period"
    else:
        raise AssertionError("duplicate source period was accepted")

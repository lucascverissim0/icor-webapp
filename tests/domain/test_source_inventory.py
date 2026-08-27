from datetime import date

import pytest

from icor.domain.source_inventory import InventoryStatus, SourceInventoryEntry


def test_unacquired_inventory_outcome_requires_an_explanation() -> None:
    with pytest.raises(ValueError, match="reason code"):
        SourceInventoryEntry(
            source_key="eea-2010-final",
            period_start=date(2010, 1, 1),
            period_end=date(2010, 12, 31),
            status=InventoryStatus.UNAVAILABLE,
            release_id=None,
            revision_state="final",
            licence_status="permitted",
            reason_codes=(),
        )


def test_validated_inventory_outcome_requires_a_release_identity() -> None:
    with pytest.raises(ValueError, match="release ID"):
        SourceInventoryEntry(
            source_key="eea-2010-final",
            period_start=date(2010, 1, 1),
            period_end=date(2010, 12, 31),
            status=InventoryStatus.ACQUIRED_VALIDATED,
            release_id=None,
            revision_state="final",
            licence_status="permitted",
            reason_codes=(),
        )


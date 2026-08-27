from dataclasses import FrozenInstanceError

import pytest

from icor.evidence.source_records import ParsedRelease


def test_parsed_release_requires_counts_to_reconcile() -> None:
    with pytest.raises(ValueError, match="record counts must reconcile"):
        ParsedRelease(
            records=("accepted",),
            raw_count=3,
            accepted_count=1,
            rejected_count=1,
            quarantined_count=0,
            warnings=(),
        )


def test_parsed_release_is_immutable_and_matches_accepted_records() -> None:
    result = ParsedRelease(
        records=("one", "two"),
        raw_count=3,
        accepted_count=2,
        rejected_count=0,
        quarantined_count=1,
        warnings=("one suppressed record",),
    )

    assert result.records == ("one", "two")
    with pytest.raises(FrozenInstanceError):
        result.raw_count = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    "field", ["raw_count", "accepted_count", "rejected_count", "quarantined_count"]
)
def test_parsed_release_rejects_negative_or_boolean_counts(field: str) -> None:
    values: dict[str, object] = {
        "records": (),
        "raw_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "quarantined_count": 0,
        "warnings": (),
    }
    values[field] = -1
    with pytest.raises(ValueError, match="non-negative integer"):
        ParsedRelease(**values)  # type: ignore[arg-type]
    values[field] = True
    with pytest.raises(ValueError, match="non-negative integer"):
        ParsedRelease(**values)  # type: ignore[arg-type]

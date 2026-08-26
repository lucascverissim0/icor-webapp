from dataclasses import replace
from datetime import UTC, datetime

import pytest

from icor.domain.opportunities import CoverageMatchType, ProductionCoverage


def coverage(**overrides: object) -> ProductionCoverage:
    values: dict[str, object] = {
        "coverage_id": "coverage-1",
        "match_type": CoverageMatchType.EXACT_CONFIGURATION,
        "configuration_id": "demo-config-1",
        "brand": "Aurora Mobility",
        "model": "A1 Horizon",
        "model_year": 2025,
        "sku": "DEMO-AUR-A1-CAM",
        "note": "Production line confirmed.",
        "created_at": datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        "updated_at": datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
    }
    values.update(overrides)
    return ProductionCoverage(**values)  # type: ignore[arg-type]


def test_exact_coverage_requires_a_configuration_identity() -> None:
    with pytest.raises(ValueError, match="exact coverage requires"):
        coverage(configuration_id=None)


def test_fallback_coverage_cannot_claim_configuration_or_sku() -> None:
    with pytest.raises(ValueError, match="fallback coverage cannot"):
        coverage(
            match_type=CoverageMatchType.VEHICLE_YEAR_FALLBACK,
            configuration_id="demo-config-1",
            sku="DEMO-AUR-A1-CAM",
        )


@pytest.mark.parametrize("note", ["line\x00break", "line\nbreak", "x" * 501])
def test_coverage_note_rejects_controls_and_oversize_text(note: str) -> None:
    with pytest.raises(ValueError, match="note"):
        coverage(note=note)


def test_coverage_requires_utc_timestamps_in_creation_order() -> None:
    with pytest.raises(ValueError, match="timestamps"):
        replace(
            coverage(),
            updated_at=datetime(2026, 8, 26, 9, 59, tzinfo=UTC),
        )


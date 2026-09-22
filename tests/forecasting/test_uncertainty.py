from decimal import Decimal
from pathlib import Path

import pytest

from icor.forecasting.uncertainty import DEFAULT_DRAW_COUNT, OpportunityUncertaintyModel


def test_uncertainty_is_seeded_ordered_and_reproducible() -> None:
    model = OpportunityUncertaintyModel(draw_count=1000)
    inputs = {
        "active_fleet_p10": Decimal("800"),
        "active_fleet_p50": Decimal("900"),
        "active_fleet_p90": Decimal("1000"),
        "hazard_p10": Decimal("0.03"),
        "hazard_p50": Decimal("0.04"),
        "hazard_p90": Decimal("0.05"),
        "seed": 20260827,
    }

    first = model.estimate(**inputs)

    assert first == model.estimate(**inputs)
    assert first.p10 <= first.p50 <= first.p90


def test_point_mass_hazard_preserves_the_fleet_interval() -> None:
    """With a fixed hazard the product is a rescaling, so deciles must carry through."""
    model = OpportunityUncertaintyModel(draw_count=20000)

    result = model.estimate(
        active_fleet_p10=Decimal("800"),
        active_fleet_p50=Decimal("900"),
        active_fleet_p90=Decimal("1000"),
        hazard_p10=Decimal("0.04"),
        hazard_p50=Decimal("0.04"),
        hazard_p90=Decimal("0.04"),
        seed=20260827,
    )

    assert float(result.p10) == pytest.approx(32.0, rel=0.02)
    assert float(result.p50) == pytest.approx(36.0, rel=0.02)
    assert float(result.p90) == pytest.approx(40.0, rel=0.02)


def test_propagated_interval_is_not_narrower_than_its_inputs() -> None:
    """P10/P90 are quantiles of the inputs, not the support of a bounded distribution."""
    model = OpportunityUncertaintyModel(draw_count=20000)

    result = model.estimate(
        active_fleet_p10=Decimal("800"),
        active_fleet_p50=Decimal("900"),
        active_fleet_p90=Decimal("1000"),
        hazard_p10=Decimal("0.04"),
        hazard_p50=Decimal("0.04"),
        hazard_p90=Decimal("0.04"),
        seed=20260827,
    )

    fleet_relative_width = (1000.0 - 800.0) / 900.0
    propagated_relative_width = float(result.p90 - result.p10) / float(result.p50)

    assert propagated_relative_width >= fleet_relative_width * 0.98


def test_build_and_query_paths_draw_the_same_number_of_samples() -> None:
    """A split draw count made the ranking page and the forecast page disagree."""
    from icor.application.generation_planning import GenerationPlanningService
    from icor.infrastructure.snapshot_vehicle_forecast_repository import (
        SnapshotVehicleForecastRepository,
    )

    build_path = GenerationPlanningService().uncertainty
    query_path = SnapshotVehicleForecastRepository(Path("unused.sqlite3"), "v")._uncertainty

    assert build_path.draw_count == query_path.draw_count == DEFAULT_DRAW_COUNT


def test_uncertainty_method_is_per_instance_not_a_class_attribute() -> None:
    default = OpportunityUncertaintyModel()
    variant = OpportunityUncertaintyModel(method="narrow-band-experiment-v1")

    assert default.method == "quantile-matched-split-normal-propagation-v2"
    assert variant.method == "narrow-band-experiment-v1"
    assert "method" not in vars(OpportunityUncertaintyModel)


def test_an_empty_uncertainty_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="uncertainty method is required"):
        OpportunityUncertaintyModel(method="   ")

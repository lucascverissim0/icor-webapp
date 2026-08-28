from decimal import Decimal

from icor.forecasting.uncertainty import OpportunityUncertaintyModel


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

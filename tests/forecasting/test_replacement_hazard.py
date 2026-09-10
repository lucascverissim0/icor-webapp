from decimal import Decimal

from icor.forecasting.replacement_hazard import ReplacementHazardModel


def test_hazard_uses_the_evidence_anchored_french_fleet_rate() -> None:
    model = ReplacementHazardModel()

    assert model.annual_probability(age_years=2, geography="FR") == Decimal("0.043026")
    assert model.annual_probability(age_years=14, geography="FR") == Decimal("0.043026")


def test_hazard_interval_is_an_explicit_twenty_percent_scenario_band() -> None:
    model = ReplacementHazardModel()

    assert model.interval(age_years=8, geography="BE") == (
        Decimal("0.0344208"),
        Decimal("0.043026"),
        Decimal("0.0516312"),
    )


def test_no_unsupported_cross_market_premium_is_applied() -> None:
    model = ReplacementHazardModel()

    assert model.annual_probability(age_years=8, geography="GB") == (
        model.annual_probability(age_years=8, geography="FR")
    )

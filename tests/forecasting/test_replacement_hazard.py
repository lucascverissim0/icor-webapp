from icor.forecasting.replacement_hazard import ReplacementHazardModel


def test_older_vehicle_has_higher_replacement_hazard() -> None:
    model = ReplacementHazardModel()

    assert model.annual_probability(age_years=12, geography="DE") > model.annual_probability(
        age_years=2, geography="DE"
    )


def test_explicit_geography_multiplier_is_applied() -> None:
    model = ReplacementHazardModel(geography_multipliers={"GB": "1.10"})

    assert model.annual_probability(age_years=8, geography="GB") > model.annual_probability(
        age_years=8, geography="DE"
    )

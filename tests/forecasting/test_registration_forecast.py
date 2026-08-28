from decimal import Decimal

from icor.forecasting.registration_forecast import RegistrationForecaster


def test_stable_history_selects_constant_candidate() -> None:
    result = RegistrationForecaster().forecast(
        {2020: Decimal("100"), 2021: Decimal("100"), 2022: Decimal("100")},
        horizon_year=2024,
    )

    assert result.method == "rolling-origin-constant-v1"
    assert result.values == ((2023, Decimal("100")), (2024, Decimal("100")))


def test_forecast_never_emits_negative_registrations() -> None:
    result = RegistrationForecaster().forecast(
        {2020: Decimal("100"), 2021: Decimal("50"), 2022: Decimal("5")},
        horizon_year=2025,
    )

    assert all(value >= 0 for _, value in result.values)

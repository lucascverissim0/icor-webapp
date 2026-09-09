from decimal import Decimal

from icor.forecasting.registration_forecast import RegistrationForecaster


def test_stable_history_uses_validated_ensemble_without_inventing_growth() -> None:
    result = RegistrationForecaster().forecast(
        {2020: Decimal("100"), 2021: Decimal("100"), 2022: Decimal("100")},
        horizon_year=2024,
    )

    assert result.method == "validated-recency-damped-ensemble-v2"
    assert result.values == ((2023, Decimal("100")), (2024, Decimal("100")))


def test_forecast_never_emits_negative_registrations() -> None:
    result = RegistrationForecaster().forecast(
        {2020: Decimal("100"), 2021: Decimal("50"), 2022: Decimal("5")},
        horizon_year=2025,
    )

    assert all(value >= 0 for _, value in result.values)


def test_ensemble_damps_trend_and_reports_multistep_backtest_error() -> None:
    result = RegistrationForecaster().forecast(
        {
            2019: Decimal("80"),
            2020: Decimal("90"),
            2021: Decimal("100"),
            2022: Decimal("110"),
            2023: Decimal("120"),
        },
        horizon_year=2025,
    )

    assert result.values == (
        (2024, Decimal("124.0000")),
        (2025, Decimal("127.2000")),
    )
    assert result.backtest_error >= 0


def test_materialization_can_skip_diagnostic_backtest_without_changing_values() -> None:
    history = {
        2020: Decimal("100"),
        2021: Decimal("110"),
        2022: Decimal("120"),
        2023: Decimal("130"),
    }

    evaluated = RegistrationForecaster().forecast(history, horizon_year=2025)
    materialized = RegistrationForecaster().forecast(
        history,
        horizon_year=2025,
        evaluate_backtest=False,
    )

    assert materialized.values == evaluated.values
    assert materialized.method == evaluated.method
    assert materialized.backtest_error == 0

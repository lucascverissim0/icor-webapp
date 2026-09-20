from decimal import Decimal

from scripts.benchmark_registration_forecasts import _contiguous_runs


def test_observed_history_is_split_at_gaps_instead_of_interpolated() -> None:
    history = {
        2018: Decimal("10"),
        2019: Decimal("11"),
        2021: Decimal("13"),
        2022: Decimal("14"),
    }

    assert _contiguous_runs(history) == (
        {2018: Decimal("10"), 2019: Decimal("11")},
        {2021: Decimal("13"), 2022: Decimal("14")},
    )

"""The hidden-register backtest scores a register country estimated without its register."""

from __future__ import annotations

import pytest

from icor.forecasting.hidden_register import (
    Cell,
    code_scores,
    cohort_errors,
    l1_gate,
    make_scores,
    per_code,
    project,
    registrations_times_survival,
    rescale_to_cohorts,
    spearman,
)

# A fictional curve: half the cars are gone at age 10, a quarter at age 5.
SHARES = {0: 1.0, 1: 0.95, 2: 0.9, 5: 0.75, 6: 0.7, 10: 0.5, 13: 0.4}


def survival(age: int) -> tuple[float, float, float]:
    p50 = SHARES[age]
    return (p50 - 0.05, p50, min(1.0, p50 + 0.05))


GOLF_16 = Cell("zephyr|arrow", "zephyr", "ARROW", 2016)
GOLF_20 = Cell("zephyr|arrow", "zephyr", "ARROW", 2020)
BOLT_16 = Cell("nova|bolt", "nova", "BOLT", 2016)


def test_the_current_fallback_is_registrations_times_survival_at_the_register_date() -> None:
    registrations = {GOLF_16: 1000.0, GOLF_20: 400.0, Cell("x|y", "x", "Y", 2027): 50.0}

    predicted = registrations_times_survival(registrations, survival, year=2026)

    # Age 10 and age 6 at a 2026 register; a cohort after the register is not predicted.
    assert predicted.keys() == {GOLF_16, GOLF_20}
    assert predicted[GOLF_16] == pytest.approx((450.0, 500.0, 550.0))
    assert predicted[GOLF_20] == pytest.approx((260.0, 280.0, 300.0))


def test_projection_ages_the_register_date_count_by_the_curves_own_ratio() -> None:
    # 500 cars at age 10 in 2026 are 400 at age 13 in 2029: 500 * 0.4 / 0.5.
    projected = project({GOLF_16: (450.0, 500.0, 550.0)}, survival, year=2026, horizon=2029)

    # Each quantile is aged by its own curve, as the planner anchors stock.
    assert projected[GOLF_16] == pytest.approx((450 * 0.35 / 0.45, 400.0, 550 * 0.45 / 0.55))


def test_cohort_errors_skip_the_incomplete_register_year_and_name_the_age() -> None:
    truth = {2016: 1000.0, 2020: 500.0, 2026: 80.0}
    errors = cohort_errors(truth, {2016: 900.0, 2026: 10.0}, year=2026)

    assert [(e.cohort, e.age, e.truth, e.predicted) for e in errors] == [
        (2016, 10, 1000.0, 900.0),
        # A cohort the hidden profile cannot see at all is predicted as zero, not skipped.
        (2020, 6, 500.0, 0.0),
    ]
    assert errors[0].error == pytest.approx(-0.1)
    assert errors[1].error == pytest.approx(-1.0)


def test_the_l1_gate_holds_young_cohorts_to_8_percent_and_old_ones_to_15() -> None:
    errors = cohort_errors(
        {2020: 100.0, 2010: 100.0}, {2020: 107.0, 2010: 86.0}, year=2026
    )  # age 6: +7%; age 16: -14%

    gate = l1_gate(errors)

    assert gate["passes"] is True
    assert gate["groups"]["0-15"]["max_abs_error"] == pytest.approx(0.07)
    assert gate["groups"]["16-25"]["max_abs_error"] == pytest.approx(0.14)

    failing = l1_gate(cohort_errors({2020: 100.0}, {2020: 91.0}, year=2026))
    assert failing["passes"] is False


def test_make_scores_use_the_top_makes_and_separate_level_from_mix() -> None:
    truth = {("zephyr", 2016): 600.0, ("nova", 2016): 400.0, ("tiny", 2016): 1.0}
    # Right mix, every make 20% short: the level is wrong, the mix is not.
    predicted = {("zephyr", 2016): 480.0, ("nova", 2016): 320.0}

    scores = make_scores(truth, predicted, top=2)

    assert scores["makes"] == ["zephyr", "nova"]
    assert scores["wape"] == pytest.approx(0.2)
    assert scores["mix_wape"] == pytest.approx(0.0)


def test_rescaling_to_true_cohort_totals_keeps_each_cohorts_mix() -> None:
    cells = {
        GOLF_16: (90.0, 100.0, 110.0),
        BOLT_16: (270.0, 300.0, 330.0),
        GOLF_20: (5.0, 10.0, 15.0),
    }

    scaled = rescale_to_cohorts(cells, {2016: 800.0, 2020: 10.0})

    assert scaled[GOLF_16] == pytest.approx((180.0, 200.0, 220.0))
    assert scaled[BOLT_16] == pytest.approx((540.0, 600.0, 660.0))
    assert scaled[GOLF_20] == pytest.approx((5.0, 10.0, 15.0))


def test_shared_units_split_evenly_and_unallocated_units_keep_their_status() -> None:
    codes = per_code(
        [(("Q101",), "single", 100.0), (("Q101", "Q102"), "shared", 50.0), ((), "unreviewed", 7.0)]
    )

    assert codes == {"Q101": 125.0, "Q102": 25.0, "(unreviewed)": 7.0}


def test_spearman_is_one_for_the_same_order_and_handles_ties() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)


def test_code_scores_measure_large_codes_bias_rank_and_interval_coverage() -> None:
    truth = {"Q101": 2000.0, "Q102": 1000.0, "Q103": 500.0, "(unreviewed)": 300.0}
    low = {"Q101": 1500.0, "Q102": 1100.0, "Q103": 300.0}
    mid = {"Q101": 1800.0, "Q102": 1200.0, "Q103": 400.0}
    high = {"Q101": 2100.0, "Q102": 1300.0, "Q103": 500.0}

    scores = code_scores(truth, low, mid, high, min_units=1000, top=3)

    # Large codes: |1800-2000| + |1200-1000| over 3000.
    assert scores["large_codes"] == 2
    assert scores["wape_large"] == pytest.approx(400 / 3000)
    # Bias over every real code; pseudo-codes are reported apart.
    assert scores["bias"] == pytest.approx(3400 / 3500 - 1)
    assert scores["unallocated_truth"] == pytest.approx(300.0)
    # Q101 is inside its interval, Q102 is not.
    assert scores["coverage_large"] == pytest.approx(0.5)
    assert scores["spearman_top"] == pytest.approx(1.0)

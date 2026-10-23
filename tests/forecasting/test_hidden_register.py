"""The hidden-register backtest scores a register country estimated without its register."""

from __future__ import annotations

import pytest

from icor.forecasting.hidden_register import (
    Cell,
    band_level_factors,
    coarse_names,
    code_scores,
    cohort_errors,
    donor_mix,
    drop_cells,
    l1_gate,
    make_scores,
    per_code,
    per_code_certain,
    perturb_cells,
    pre_profile_share,
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


def test_certain_units_never_split_a_shared_registration() -> None:
    certain, other = per_code_certain(
        [(("Q1",), "single", 10.0), (("Q1", "Q2"), "shared", 6.0), ((), "outside", 4.0)]
    )

    assert certain == {"Q1": 10.0}
    assert other == {"shared": 6.0, "outside": 4.0}


def test_pre_profile_share_counts_production_years_before_the_profile() -> None:
    assert pre_profile_share(2005, 2014, profile_first_year=2010, year=2026) == 0.5
    assert pre_profile_share(2012, None, profile_first_year=2010, year=2026) == 0.0
    assert pre_profile_share(2000, 2005, profile_first_year=2010, year=2026) == 1.0
    assert pre_profile_share(2030, None, profile_first_year=2010, year=2026) == 0.0


def test_band_factors_use_only_bands_the_profile_holds_whole() -> None:
    layout = (("Y_LT2", 0, 1), ("Y2-5", 2, 4), ("Y5-10", 5, 9), ("Y10-20", 10, 19))
    predicted = {year: 100.0 for year in range(2010, 2026)}

    result = band_level_factors(
        {"Y_LT2": 300.0, "Y2-5": 300.0, "Y5-10": 600.0, "Y10-20": 9999.0},
        layout,
        predicted,
        band_year=2024,
        profile_first_year=2010,
    )

    factor = result["factor"]
    assert factor[2024] == factor[2023] == 1.5
    assert factor[2021] == 1.0
    assert factor[2016] == 1.2
    # 2005-2014 reaches before the profile: its cohorts take the next younger band.
    assert factor[2012] == 1.2
    # Registered after the office's count: the youngest band.
    assert factor[2025] == 1.5


def test_donor_mix_leaves_the_target_out_and_weighs_countries_equally() -> None:
    shares = {"AA": {"Q1": 1.0}, "BB": {"Q2": 1.0}, "CC": {"Q1": 0.5, "Q2": 0.5}}

    assert donor_mix(shares, "CC") == {"Q1": 0.5, "Q2": 0.5}


def test_degradations_are_deterministic_and_keep_the_cells_they_should() -> None:
    cells = {Cell(f"make|m{i}", "make", f"Model {i} Sport", 2015): 100.0 for i in range(400)}

    dropped = drop_cells(cells, 0.17, seed="s")
    perturbed = perturb_cells(cells, 0.05, seed="s")

    assert dropped == drop_cells(cells, 0.17, seed="s")
    assert 0.10 < 1 - len(dropped) / len(cells) < 0.24
    assert all(95.0 <= units <= 105.0 for units in perturbed.values())
    assert {cell.model for cell in coarse_names(cells)} == {"Model"}


def test_a_shared_allocation_with_one_code_is_not_certain() -> None:
    certain, other = per_code_certain([(("Q1",), "shared", 5.0), (("Q1",), "single", 2.0)])

    assert certain == {"Q1": 2.0}
    assert other == {"shared": 5.0}


def test_a_lost_label_takes_every_cohort_of_the_vehicle() -> None:
    cells = {
        Cell(f"id{vehicle}", "Make", f"Model {vehicle}", cohort): 1.0
        for vehicle in range(200)
        for cohort in range(2010, 2025)
    }

    kept = drop_cells(cells, 0.17, seed="s")
    perturbed = perturb_cells(cells, 0.05, seed="s")

    cohorts_kept: dict[str, int] = {}
    for cell in kept:
        cohorts_kept[cell.identity] = cohorts_kept.get(cell.identity, 0) + 1
    assert set(cohorts_kept.values()) == {15}
    assert 0.10 < 1 - len(cohorts_kept) / 200 < 0.25
    for vehicle in range(200):
        factors = {value for cell, value in perturbed.items() if cell.identity == f"id{vehicle}"}
        assert len(factors) == 1


def test_a_band_missing_between_usable_ones_takes_the_younger_neighbour() -> None:
    layout = [("Y_LT2", 0, 1), ("Y2-5", 2, 4), ("Y5-10", 5, 9)]
    predicted = {cohort: 100.0 for cohort in range(2014, 2025)}

    result = band_level_factors(
        {"Y_LT2": 300.0, "Y5-10": 600.0},
        layout,
        predicted,
        band_year=2024,
        profile_first_year=2010,
    )

    assert result["factor"][2023] == pytest.approx(1.5)
    assert result["factor"][2021] == pytest.approx(1.5)  # Y2-5 is absent
    assert result["factor"][2017] == pytest.approx(1.2)

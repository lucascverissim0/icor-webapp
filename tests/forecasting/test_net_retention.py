import json
from decimal import Decimal

import pytest

from icor.application.generation_planning import GenerationPlanningService
from icor.forecasting import net_retention
from icor.forecasting.net_retention import MarketYear
from icor.forecasting.survival import (
    NET_RETENTION_PATH,
    NetRetainedSurvivalModel,
    load_net_retained_survival_model,
    load_promoted_survival_model,
    load_survival_model_for,
)

SUPPORT = 30


def _share(age: int) -> float:
    return 0.95**age


def _market(year: int, retention, *, shift: int = 0, official=None) -> MarketYear:
    registrations = {item: 1_000 for item in range(year - SUPPORT, year + 1)}
    computed = {}
    for name, low, high in net_retention.BANDS:
        ages = net_retention.band_ages(low, high, shift)
        computed[name] = round(sum(1_000 * _share(age) * retention(age) for age in ages))
    return MarketYear(year, shift, official or computed, registrations)


def _exporting(age: int) -> float:
    return 1.0 if age < 5 else 0.6


def test_a_market_the_curve_matches_gets_no_retention():
    years = [_market(year, lambda _age: 1.0) for year in (2023, 2024, 2025)]

    goal = net_retention.targets(years, _share, until=2025)

    assert set(goal.values()) == {1.0}
    assert net_retention.fit_curve(goal, years[-1], _share, support=SUPPORT) is None


def test_a_shortfall_within_tolerance_is_left_alone_and_a_surplus_never_raises():
    years = [_market(2025, lambda age: 0.98 if age >= 5 else 1.2)]

    assert net_retention.targets(years, _share, until=2025) == {
        "Y_LT2": 1.0,
        "Y2-5": 1.0,
        "Y5-10": 1.0,
        "Y10-20": 1.0,
    }


def test_the_target_is_the_largest_recent_ratio_so_one_bad_year_cannot_lower_it():
    years = [
        _market(2023, lambda age: 0.7 if age >= 5 else 1.0),
        _market(2024, _exporting),
        _market(2025, lambda age: 0.5 if age >= 5 else 1.0),
        _market(2010, lambda age: 0.9 if age >= 5 else 1.0),
    ]

    goal = net_retention.targets(years, _share, until=2025)

    assert goal["Y5-10"] == pytest.approx(0.7, abs=0.002)
    assert goal["Y_LT2"] == 1.0


def _declining(age: int) -> float:
    return max(0.4, 1 - 0.05 * max(0, age - 4))


def test_the_fitted_curve_reproduces_the_bands_and_never_increases_or_exceeds_one():
    years = [_market(year, _declining) for year in (2023, 2024, 2025)]
    goal = net_retention.targets(years, _share, until=2025)

    curve = net_retention.fit_curve(goal, years[-1], _share, support=SUPPORT)

    assert curve is not None and len(curve) == SUPPORT
    assert all(later <= earlier for earlier, later in zip(curve, curve[1:], strict=False))
    assert max(curve) == 1.0
    # Bands kept whole stay whole: the decline starts after their oldest age.
    assert curve[:5] == (1.0,) * 5
    for name, low, high in net_retention.BANDS[2:]:
        ages = net_retention.band_ages(low, high, 0)
        modelled = sum(1_000 * _share(age) for age in ages)
        fitted = sum(1_000 * _share(age) * curve[age] for age in ages)
        assert fitted / modelled == pytest.approx(goal[name], abs=0.02)


def test_a_rising_shortfall_pools_into_a_non_increasing_curve():
    years = [_market(2025, lambda age: 0.8 if age < 2 else 1.0)]
    goal = net_retention.targets(years, _share, until=2025)

    curve = net_retention.fit_curve(goal, years[-1], _share, support=SUPPORT)

    assert curve is None or all(
        later <= earlier for earlier, later in zip(curve, curve[1:], strict=False)
    )


def test_the_backtest_keeps_a_curve_only_where_it_predicts_later_years_better():
    exporting = [_market(year, _exporting) for year in range(2018, 2026)]
    curve, test = net_retention.backtest(exporting, _share, fit_until=2022, support=SUPPORT)
    assert curve is not None and test.passes
    assert test.error(4) < test.error(3)

    # A shortfall that vanished after the fit predicts the later years worse.
    changed = [
        *(_market(year, _exporting) for year in range(2018, 2023)),
        *(_market(year, lambda _age: 1.0) for year in range(2023, 2026)),
    ]
    _curve, worse = net_retention.backtest(changed, _share, fit_until=2022, support=SUPPORT)
    assert not worse.passes


def test_a_three_year_convention_moves_every_band_edge_but_the_first():
    assert list(net_retention.band_ages(0, 1, 1)) == [0, 1, 2]
    assert list(net_retention.band_ages(2, 4, 1)) == [3, 4, 5]
    assert list(net_retention.band_ages(10, 19, 0)) == list(range(10, 20))


def test_the_retained_model_lowers_only_a_calibrated_market_and_every_quantile_alike():
    base = load_promoted_survival_model()
    curve = tuple(Decimal("1") if age < 5 else Decimal("0.5") for age in range(41))
    model = NetRetainedSurvivalModel(base, {"BE": curve}, method="test-retention-v1")

    plain = base.interval(Decimal(1000), age_years=8)
    lowered = model.interval(Decimal(1000), age_years=8, geography="BE")

    assert model.interval(Decimal(1000), age_years=8, geography="FR") == plain
    assert model.interval(Decimal(1000), age_years=8) == plain
    assert model.interval(Decimal(1000), age_years=3, geography="BE") == base.interval(
        Decimal(1000), age_years=3
    )
    assert lowered.p10 == (plain.p10 * Decimal("0.5")).quantize(Decimal("0.0001"))
    assert lowered.p50 == (plain.p50 * Decimal("0.5")).quantize(Decimal("0.0001"))
    assert lowered.p90 == (plain.p90 * Decimal("0.5")).quantize(Decimal("0.0001"))
    assert model.retention("BE", age_years=99) == Decimal("0.5")
    assert model.method == f"{base.method}+test-retention-v1"
    assert model.reason_code("BE") == base.reason_code("BE")


@pytest.mark.parametrize(
    "curve",
    [
        (Decimal("1"), Decimal("1.01")),
        (Decimal("0.9"), Decimal("0.95")),
        (Decimal("0"),),
        (),
    ],
)
def test_the_retained_model_refuses_a_curve_that_could_add_cars(curve):
    with pytest.raises(ValueError):
        NetRetainedSurvivalModel(load_promoted_survival_model(), {"BE": curve}, method="m")


def test_the_committed_artifact_calibrates_on_the_promoted_curve_and_only_lowers():
    document = json.loads(NET_RETENTION_PATH.read_text(encoding="utf-8"))
    model = load_net_retained_survival_model()

    assert document["base_method"] == load_promoted_survival_model().method
    assert "GB" not in document["markets"]
    assert set(model.multipliers) == set(document["markets"])
    for market in document["markets"].values():
        assert market["backtest"]["calibrated_error"] < market["backtest"]["uncalibrated_error"]


def test_a_snapshot_is_read_with_the_model_its_method_names():
    base = load_promoted_survival_model()
    retained = load_net_retained_survival_model()

    assert load_survival_model_for(base.method).method == base.method
    assert load_survival_model_for(retained.method).method == retained.method
    with pytest.raises(ValueError, match="no committed survival model"):
        load_survival_model_for("constant-annual-retention-v1")


def test_planning_names_the_retention_a_cohort_got():
    base = load_promoted_survival_model()
    curve = tuple(Decimal("1") if age < 5 else Decimal("0.5") for age in range(41))
    planner = GenerationPlanningService(
        survival=NetRetainedSurvivalModel(base, {"BE": curve}, method="m")
    )

    young = planner._conditional_fleet(Decimal(100), age_years=4, stock_age_years=2, geography="BE")
    crossing = planner._conditional_fleet(
        Decimal(100), age_years=6, stock_age_years=4, geography="BE"
    )
    elsewhere = planner._conditional_fleet(
        Decimal(100), age_years=6, stock_age_years=4, geography="FR"
    )

    assert young is not None and crossing is not None and elsewhere is not None
    assert crossing.p50 == pytest.approx(elsewhere.p50 * Decimal("0.5"), abs=Decimal("0.0002"))
    assert young.p50 > crossing.p50

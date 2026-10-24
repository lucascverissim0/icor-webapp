"""The age anchor reproduces an office's bands, may add cars, and is backtested."""

from __future__ import annotations

import json
from decimal import Decimal
from re import fullmatch

import pytest

from icor.application.generation_planning import GenerationPlanningService
from icor.domain.cohorts import _IDENTIFIER
from icor.forecasting.age_anchor import age_curve, backtest, band_anchors, fit
from icor.forecasting.net_retention import MarketYear
from icor.forecasting.survival import (
    AGE_ANCHOR_PATH,
    AgeAnchoredSurvivalModel,
    load_age_anchored_survival_model,
    load_net_retained_survival_model,
    load_survival_model_for,
)


def share(age: int) -> float:
    return max(0.0, 1 - 0.04 * age)


def _year(year: int, official: dict[str, int], *, first: int = 2004, open_ended=False):
    return MarketYear(
        year=year,
        convention="2y",
        official=official,
        registrations={cohort: 1000 for cohort in range(first, year + 1)},
        open_ended=open_ended,
    )


def _modelled(year: int, ages: range) -> float:
    return sum(1000 * share(age) for age in ages)


def test_anchors_reproduce_each_band_and_may_exceed_one() -> None:
    young, mid = _modelled(2025, range(0, 2)), _modelled(2025, range(2, 5))
    item = _year(2025, {"Y_LT2": round(young * 0.9), "Y2-5": round(mid * 1.5)})

    anchors = band_anchors(item, share)

    assert anchors["Y_LT2"] == pytest.approx(0.9, abs=1e-3)
    assert anchors["Y2-5"] == pytest.approx(1.5, abs=1e-3)


def test_a_band_the_registrations_do_not_cover_is_not_anchored() -> None:
    item = _year(
        2025, {"Y10-20": 1, "Y5-10": round(_modelled(2025, range(5, 10)) * 1.2)}, first=2010
    )

    anchors = band_anchors(item, share)

    assert "Y10-20" not in anchors
    assert anchors["Y5-10"] == pytest.approx(1.2, abs=1e-3)


def test_an_open_ended_last_band_is_never_anchored() -> None:
    item = _year(2025, {"Y10-20": 10**9}, open_ended=True)

    assert band_anchors(item, share) == {}


def test_anchors_within_tolerance_of_one_are_one() -> None:
    item = _year(2025, {"Y_LT2": round(_modelled(2025, range(0, 2)) * 1.02)})

    assert band_anchors(item, share) == {"Y_LT2": 1.0}
    assert fit([item], share, until=2025, support=30) is None


def test_ages_outside_the_usable_bands_take_the_nearest_band() -> None:
    item = _year(2025, {}, first=2010)

    curve = age_curve(item, {"Y2-5": 1.5, "Y5-10": 1.2}, support=25)

    assert curve[0] == curve[1] == 1.5  # younger than the youngest usable band
    assert curve[3] == 1.5 and curve[7] == 1.2
    assert curve[15] == curve[24] == 1.2  # older than the oldest usable band


def test_backtest_scores_later_years_with_anchors_fitted_earlier() -> None:
    def official(year: int) -> dict[str, int]:
        return {
            "Y_LT2": round(_modelled(year, range(0, 2)) * 1.3),
            "Y2-5": round(_modelled(year, range(2, 5)) * 1.3),
        }

    items = [_year(year, official(year)) for year in range(2019, 2026)]

    curve, test = backtest(items, share, fit_until=2022, support=30)

    assert curve is not None and curve[0] == pytest.approx(1.3, abs=1e-3)
    assert {cell[0] for cell in test.cells} == {2023, 2024, 2025}
    assert test.error(4) < test.error(3)
    assert test.passes


def _anchored(curve: tuple[Decimal, ...]) -> AgeAnchoredSurvivalModel:
    return AgeAnchoredSurvivalModel(
        load_net_retained_survival_model(), {"AT": curve}, method="test-anchor-v1"
    )


def test_the_anchored_model_scales_only_its_market_and_may_add_cars() -> None:
    curve = tuple(Decimal("1.5") if age < 5 else Decimal("0.8") for age in range(41))
    model = _anchored(curve)
    plain = model.unanchored.interval(Decimal(1000), age_years=3, geography="AT")

    raised = model.interval(Decimal(1000), age_years=3, geography="AT")

    assert raised.p10 == (plain.p10 * Decimal("1.5")).quantize(Decimal("0.0001"))
    assert raised.p50 == (plain.p50 * Decimal("1.5")).quantize(Decimal("0.0001"))
    assert raised.p90 == (plain.p90 * Decimal("1.5")).quantize(Decimal("0.0001"))
    assert model.interval(Decimal(1000), age_years=3, geography="FR") == (
        model.unanchored.interval(Decimal(1000), age_years=3, geography="FR")
    )
    assert model.interval(Decimal(1000), age_years=3) == model.unanchored.interval(
        Decimal(1000), age_years=3
    )
    assert model.anchor("AT", age_years=99) == Decimal("0.8")
    assert model.retention("AT", age_years=8) == model.unanchored.retention("AT", age_years=8)
    assert model.method == f"{model.unanchored.method}+test-anchor-v1"


@pytest.mark.parametrize("curve", [(), (Decimal("0"),), (Decimal("1"), Decimal("-1"))])
def test_the_anchored_model_refuses_a_non_positive_anchor(curve) -> None:
    with pytest.raises(ValueError):
        _anchored(curve)


def test_cars_a_register_observed_are_aged_without_the_anchor() -> None:
    curve = tuple(Decimal("1.5") if age < 5 else Decimal("0.8") for age in range(41))
    anchored = GenerationPlanningService(survival=_anchored(curve))
    plain = GenerationPlanningService(survival=load_net_retained_survival_model())

    for age, stock_age in ((4, 2), (8, 3)):
        assert anchored._conditional_fleet(
            Decimal(100), age_years=age, stock_age_years=stock_age, geography="AT"
        ) == plain._conditional_fleet(
            Decimal(100), age_years=age, stock_age_years=stock_age, geography="AT"
        )


def test_the_committed_anchor_excludes_gb_and_keeps_only_markets_it_predicted_better() -> None:
    document = json.loads(AGE_ANCHOR_PATH.read_text(encoding="utf-8"))
    model = load_age_anchored_survival_model()

    assert document["base_method"] == load_net_retained_survival_model().method
    for excluded in ("GB", "HR"):
        assert excluded in document["excluded_markets"]
        assert excluded not in document["markets"]
        assert excluded not in document["rejected_markets"]
    assert set(model.anchors) == set(document["markets"])
    for market in document["markets"].values():
        assert market["backtest"]["anchored_error"] < market["backtest"]["unanchored_error"]
    assert load_survival_model_for(model.method).method == model.method
    for item in model.assumption_ids:
        assert fullmatch(_IDENTIFIER, item), item


def test_the_anchored_method_names_the_exact_anchor_artifact(tmp_path) -> None:
    # A snapshot id derives from the method names, so an edited anchor file must
    # change the method, or two different ledgers would share one snapshot id.
    committed = load_age_anchored_survival_model()
    document = json.loads(AGE_ANCHOR_PATH.read_text(encoding="utf-8"))
    edited = dict(document, markets=dict(document["markets"]))
    edited["markets"].pop(next(iter(edited["markets"])))
    path = tmp_path / "anchor.json"
    path.write_text(json.dumps(edited), encoding="utf-8")

    changed = load_age_anchored_survival_model(path)

    assert changed.method != committed.method
    assert committed.method.startswith(changed.unanchored.method + "+eurostat-age-anchor-v1-")

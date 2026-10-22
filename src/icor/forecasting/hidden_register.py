"""Score a fleet estimate made for a register country as if it had no register.

A country with a complete national register is the only truth for how many of
each vehicle are on its roads. Hiding the register and estimating the country
from what a register-less country has (new registrations, official totals)
measures how wrong the estimate is where no register exists. The estimate is
scored at three levels (Mission 4 plan, Phase 1):

- L1, cars per registration cohort, against the register's own cohorts;
- L2, cars per make and cohort, for the largest makes;
- L3, cars per Eurocode at the forecast horizon, which is what the ranking uses.
  The replacement rate does not depend on age, so demand per code is
  proportional to fleet per code and this is also the demand error.

Every function is pure: the caller reads the snapshot. Rescaling to the true
cohort totals is an oracle, not an estimator: it uses the truth to separate the
error in the mix of vehicles from the error in the level.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

#: (p10, p50, p90).
Interval = tuple[float, float, float]
#: The share of a cohort still on the road at an age, per quantile.
SurvivalShares = Callable[[int], Interval]

#: Mission 4 L1 gate: the largest cohort error allowed at each band of ages.
L1_GATES: tuple[tuple[int, int, float], ...] = ((0, 15, 0.08), (16, 25, 0.15))


@dataclass(frozen=True, slots=True)
class Cell:
    """One vehicle's registration cohort in the hidden country."""

    #: The key its registrations are allocated to Eurocodes under.
    identity: str
    make: str
    #: The publisher's model label, which reviewed spelling variants match.
    model: str
    cohort: int


@dataclass(frozen=True, slots=True)
class CohortError:
    cohort: int
    age: int
    truth: float
    predicted: float

    @property
    def error(self) -> float:
        return self.predicted / self.truth - 1


def registrations_times_survival(
    registrations: Mapping[Cell, float], survival: SurvivalShares, *, year: int
) -> dict[Cell, Interval]:
    """The current fallback: each cohort's registrations times the curve at its age."""

    predicted = {}
    for cell, units in registrations.items():
        if cell.cohort > year or not units:
            continue
        shares = survival(year - cell.cohort)
        predicted[cell] = (units * shares[0], units * shares[1], units * shares[2])
    return predicted


def project(
    cells: Mapping[Cell, Interval], survival: SurvivalShares, *, year: int, horizon: int
) -> dict[Cell, Interval]:
    """Age counts made at ``year`` to ``horizon``, each quantile by its own curve.

    This is how the planner ages a register's stock, so the truth and an
    estimate at the register date reach the horizon the same way.
    """

    projected = {}
    for cell, values in cells.items():
        now, later = survival(year - cell.cohort), survival(horizon - cell.cohort)
        projected[cell] = tuple(  # type: ignore[assignment]
            value * after / before if before > 0 else 0.0
            for value, before, after in zip(values, now, later, strict=True)
        )
    return projected


def cohort_errors(
    truth: Mapping[int, float], predicted: Mapping[int, float], *, year: int
) -> list[CohortError]:
    """Each complete cohort's error; the register's own year is still being registered."""

    return [
        CohortError(cohort, year - cohort, units, predicted.get(cohort, 0.0))
        for cohort, units in sorted(truth.items())
        if cohort < year and units > 0
    ]


def l1_gate(errors: Iterable[CohortError]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    passes = True
    errors = list(errors)
    for low, high, limit in L1_GATES:
        inside = [abs(e.error) for e in errors if low <= e.age <= high]
        worst = max(inside, default=0.0)
        groups[f"{low}-{high}"] = {
            "cohorts": len(inside),
            "limit": limit,
            "max_abs_error": worst,
            "mean_abs_error": sum(inside) / len(inside) if inside else 0.0,
            "passes": worst <= limit,
        }
        passes = passes and worst <= limit
    return {"passes": passes, "groups": groups}


def _wape(truth: Mapping[Any, float], predicted: Mapping[Any, float], keys: Iterable[Any]) -> float:
    keys = list(keys)
    denominator = sum(truth.get(key, 0.0) for key in keys)
    if not denominator:
        return 0.0
    return sum(abs(predicted.get(key, 0.0) - truth.get(key, 0.0)) for key in keys) / denominator


def make_scores(
    truth: Mapping[tuple[str, int], float],
    predicted: Mapping[tuple[str, int], float],
    *,
    top: int = 20,
) -> dict[str, Any]:
    """WAPE over the largest makes' cohorts, and the same with each cohort's level corrected."""

    by_make: dict[str, float] = defaultdict(float)
    for (make, _cohort), units in truth.items():
        by_make[make] += units
    makes = sorted(by_make, key=lambda make: (-by_make[make], make))[:top]
    keys = [key for key in {*truth, *predicted} if key[0] in makes]
    true_cohort: dict[int, float] = defaultdict(float)
    predicted_cohort: dict[int, float] = defaultdict(float)
    for key in keys:
        true_cohort[key[1]] += truth.get(key, 0.0)
        predicted_cohort[key[1]] += predicted.get(key, 0.0)
    mixed = {
        key: predicted.get(key, 0.0) * true_cohort[key[1]] / predicted_cohort[key[1]]
        for key in keys
        if predicted_cohort[key[1]]
    }
    return {
        "makes": makes,
        "wape": _wape(truth, predicted, keys),
        "mix_wape": _wape(truth, mixed, keys),
    }


def rescale_to_cohorts(
    cells: Mapping[Cell, Interval], totals: Mapping[int, float]
) -> dict[Cell, Interval]:
    """Oracle: scale each cohort so its p50 total equals the true total; the mix is kept."""

    current: dict[int, float] = defaultdict(float)
    for cell, values in cells.items():
        current[cell.cohort] += values[1]
    scaled = {}
    for cell, values in cells.items():
        base = current[cell.cohort]
        factor = totals.get(cell.cohort, 0.0) / base if base else 0.0
        scaled[cell] = (values[0] * factor, values[1] * factor, values[2] * factor)
    return scaled


def per_code(allocated: Iterable[tuple[tuple[str, ...], str, float]]) -> dict[str, float]:
    """Units per Eurocode; a shared allocation is split evenly between its codes.

    Units no code can take are kept under their status in parentheses, so they
    are counted but never scored as a code.
    """

    totals: dict[str, float] = defaultdict(float)
    for codes, status, units in allocated:
        if codes:
            for code in codes:
                totals[code] += units / len(codes)
        else:
            totals[f"({status})"] += units
    return dict(totals)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(first: list[float], second: list[float]) -> float:
    """Rank correlation, ties given their average rank."""

    a, b = _ranks(list(first)), _ranks(list(second))
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    spread = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return covariance / spread if spread else 0.0


def code_scores(
    truth: Mapping[str, float],
    low: Mapping[str, float],
    mid: Mapping[str, float],
    high: Mapping[str, float],
    *,
    min_units: float = 1000,
    top: int = 200,
) -> dict[str, Any]:
    """The L3 measures: error on large codes, overall bias, rank order and interval coverage."""

    codes = [code for code in {*truth, *mid} if not code.startswith("(")]
    large = [code for code in codes if truth.get(code, 0.0) >= min_units]
    ranked = sorted(codes, key=lambda code: (-truth.get(code, 0.0), code))[:top]
    true_total = sum(truth.get(code, 0.0) for code in codes)
    inside = [code for code in large if low.get(code, 0.0) <= truth[code] <= high.get(code, 0.0)]
    return {
        "codes": len(codes),
        "large_codes": len(large),
        "wape_large": _wape(truth, mid, large),
        "bias": sum(mid.get(code, 0.0) for code in codes) / true_total - 1 if true_total else 0.0,
        "spearman_top": spearman(
            [truth.get(code, 0.0) for code in ranked], [mid.get(code, 0.0) for code in ranked]
        )
        if len(ranked) > 1
        else 0.0,
        "coverage_large": len(inside) / len(large) if large else 0.0,
        "unallocated_truth": sum(
            units for code, units in truth.items() if code.startswith("(")
        ),
    }

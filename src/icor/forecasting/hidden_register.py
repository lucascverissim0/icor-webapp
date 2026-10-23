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

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
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
        "unallocated_truth": sum(units for code, units in truth.items() if code.startswith("(")),
    }


def per_code_certain(
    allocated: Iterable[tuple[tuple[str, ...], str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Units each Eurocode certainly carries, and every other unit under its outcome.

    This is how the ranking counts a code: only a ``single`` allocation is
    certain, and a registration that may carry several codes is never split
    between them. A ``shared`` allocation stays shared even with one code (an
    uncoded sibling may be the car). The second mapping holds the shared units
    (``shared``) and those no code takes, under their status.
    """

    certain: dict[str, float] = defaultdict(float)
    other: dict[str, float] = defaultdict(float)
    for codes, status, units in allocated:
        if status == "single" and len(codes) == 1:
            certain[codes[0]] += units
        else:
            other["shared" if codes else status] += units
    return dict(certain), dict(other)


def pre_profile_share(
    first_year: int, last_year: int | None, *, profile_first_year: int, year: int
) -> float:
    """The share of a code's production years up to ``year`` before the profile starts.

    It is known from the catalogue alone, so it can describe a code wherever a
    range is applied. A code still in production is counted to the year before
    ``year``, whose registrations are complete.
    """

    end = min(last_year if last_year is not None else year - 1, year - 1)
    if end < first_year:
        return 0.0
    before = max(0, min(end, profile_first_year - 1) - first_year + 1)
    return before / (end - first_year + 1)


def band_level_factors(
    official: Mapping[str, float],
    layout: Iterable[tuple[str, int, int | None]],
    predicted_by_cohort: Mapping[int, float],
    *,
    band_year: int,
    profile_first_year: int,
) -> dict[str, Any]:
    """Each cohort's level correction from official cars by age band.

    A register-less country still has its statistics office's count of cars by
    age band, which holds the used imports and the scrapping its new-car
    registrations never show. Within a band every cohort is scaled by the
    official count over the estimate's count at ``band_year``. A band whose
    cohorts reach before the profile is not used, since the estimate holds only
    part of it; its cohorts, and those of a band missing between usable ones,
    take the nearest younger usable band's factor, and cohorts registered after
    ``band_year`` take the youngest band's.
    """

    usable: list[tuple[int, int, float]] = []
    for name, low, high in layout:
        if high is None or name not in official:
            continue
        cohorts = range(band_year - high, band_year - low + 1)
        if cohorts.start < profile_first_year:
            continue
        estimate = sum(predicted_by_cohort.get(cohort, 0.0) for cohort in cohorts)
        if estimate > 0:
            usable.append((cohorts.start, cohorts.stop - 1, official[name] / estimate))
    if not usable:
        return {"bands": [], "factor": {}}
    usable.sort()
    factors: dict[int, float] = {}
    for cohort in predicted_by_cohort:
        inside = [factor for first, last, factor in usable if first <= cohort <= last]
        if inside:
            factors[cohort] = inside[0]
        elif cohort > usable[-1][1]:
            factors[cohort] = usable[-1][2]
        else:
            factors[cohort] = min(band for band in usable if band[0] > cohort)[2]
    return {
        "bands": [
            {"first_cohort": first, "last_cohort": last, "factor": factor}
            for first, last, factor in usable
        ],
        "factor": factors,
    }


def donor_mix(shares: Mapping[str, Mapping[str, float]], target: str) -> dict[str, float]:
    """Baseline (iii): the other countries' share of cars per code, each country weighing one."""

    donors = [country for country in shares if country != target]
    mix: dict[str, float] = defaultdict(float)
    for country in donors:
        for code, share in shares[country].items():
            mix[code] += share / len(donors)
    return dict(mix)


def _unit_interval(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _vehicle(cell: Cell) -> tuple[str, str, str]:
    """A label is lost or misread for every cohort of a vehicle at once."""

    return (cell.identity, cell.make, cell.model)


def drop_cells(
    registrations: Mapping[Cell, float], share: float, *, seed: str
) -> dict[Cell, float]:
    """Degradation: lose whole vehicles' registrations, as a blank model label does."""

    return {
        cell: units
        for cell, units in registrations.items()
        if _unit_interval(seed, *_vehicle(cell)) >= share
    }


def perturb_cells(
    registrations: Mapping[Cell, float], spread: float, *, seed: str
) -> dict[Cell, float]:
    """Degradation: each vehicle off by up to ``spread`` either way, as two publishers disagree."""

    return {
        cell: units * (1 + spread * (2 * _unit_interval(seed, *_vehicle(cell)) - 1))
        for cell, units in registrations.items()
    }


def coarse_names(registrations: Mapping[Cell, float]) -> dict[Cell, float]:
    """Degradation: a publisher that names only the model's first word."""

    coarse: dict[Cell, float] = defaultdict(float)
    for cell, units in registrations.items():
        words = cell.model.split()
        coarse[replace(cell, model=words[0] if words else cell.model)] += units
    return dict(coarse)

"""Measure an 80% range for a code's estimate from how wrong such estimates were.

Where a country has a complete national register, its true count per Eurocode is
known. Hiding it and estimating the country the way a register-less country is
estimated gives, per code, a ratio of truth to prediction. Those ratios are the
evidence for a measured range: a code estimated at P50 elsewhere is likely to
lie within P50 times the 10th and 90th percentile ratios of comparable codes.

"Comparable" uses only what is known when the range is applied: the predicted
size and the share of the code's life before the estimate's first source year.
Each held-out country is scored with quantiles learned from the other countries
only (leave one country out), and every donor country carries equal weight, so
one large register cannot set the range for all.

The gate (Part B step 2): on codes predicted at 1,000 cars or more, held-out
P10-P90 coverage must be 72-88% pooled and at least 60% in every held-out
country. A range that fails is reported, never shipped.

Every function is pure.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Predicted cars per code at which a code is scored and a range is applied.
MIN_PREDICTED = 1000.0
#: Predicted-size band edges, in cars.
SIZE_EDGES: tuple[float, ...] = (5_000.0, 20_000.0)
#: A code with at least this share of its life before the first source year.
PRE_PROFILE_EDGE = 0.25
#: Fewer donor codes than this in a bin and the bin falls back to all donors.
MIN_BIN = 50
QUANTILES = (0.10, 0.90)
GATE_POOLED = (0.72, 0.88)
GATE_PER_COUNTRY = 0.60


@dataclass(frozen=True, slots=True)
class CodeObservation:
    """One code in one held-out country: the truth and the estimate's P50."""

    country: str
    code: str
    truth: float
    predicted: float
    #: Share of the code's life before the estimate's first source year.
    pre_profile_share: float

    @property
    def log_ratio(self) -> float:
        """log(truth / predicted); a code with no true cars is infinitely over-predicted."""

        if self.predicted <= 0:
            raise ValueError("a scored code must have a positive prediction")
        return math.log(self.truth / self.predicted) if self.truth > 0 else -math.inf


def size_band(predicted: float) -> int:
    return sum(predicted >= edge for edge in SIZE_EDGES)


def bin_of(observation: CodeObservation) -> tuple[int, int]:
    return size_band(observation.predicted), int(observation.pre_profile_share >= PRE_PROFILE_EDGE)


def weighted_quantile(values: Sequence[tuple[float, float]], quantile: float) -> float:
    """The smallest value whose cumulative weight reaches ``quantile`` of the total."""

    ordered = sorted(values)
    total = sum(weight for _, weight in ordered)
    if not ordered or total <= 0:
        raise ValueError("no weighted values")
    target = quantile * total
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= target - 1e-12:
            return value
    return ordered[-1][0]


def _equal_country_weights(
    observations: Iterable[CodeObservation],
) -> list[tuple[float, float]]:
    """Each country's observations share one unit of weight between them."""

    by_country: dict[str, list[CodeObservation]] = defaultdict(list)
    for observation in observations:
        by_country[observation.country].append(observation)
    return [(item.log_ratio, 1.0 / len(items)) for items in by_country.values() for item in items]


def ratio_table(donors: Sequence[CodeObservation], *, min_bin: int = MIN_BIN) -> dict[str, Any]:
    """P10/P90 log ratios per bin from these donors, and the pooled fallback."""

    pooled = _equal_country_weights(donors)
    table: dict[str, Any] = {
        "pooled": tuple(weighted_quantile(pooled, q) for q in QUANTILES),
        "bins": {},
    }
    grouped: dict[tuple[int, int], list[CodeObservation]] = defaultdict(list)
    for observation in donors:
        grouped[bin_of(observation)].append(observation)
    for key, members in grouped.items():
        if len(members) >= min_bin:
            weighted = _equal_country_weights(members)
            table["bins"][key] = tuple(weighted_quantile(weighted, q) for q in QUANTILES)
    return table


def interval(table: Mapping[str, Any], observation: CodeObservation) -> tuple[float, float]:
    """The measured P10-P90 range of a code's true count, in cars."""

    low, high = table["bins"].get(bin_of(observation), table["pooled"])
    return observation.predicted * math.exp(low), observation.predicted * math.exp(high)


def held_out_intervals(
    observations: Sequence[CodeObservation], *, min_bin: int = MIN_BIN
) -> dict[tuple[str, str], tuple[float, float]]:
    """Each country's ranges, learned from every other country only."""

    scored = [item for item in observations if item.predicted >= MIN_PREDICTED]
    intervals = {}
    for country in sorted({item.country for item in scored}):
        donors = [item for item in scored if item.country != country]
        if not donors:
            continue
        table = ratio_table(donors, min_bin=min_bin)
        for item in scored:
            if item.country == country:
                intervals[(item.country, item.code)] = interval(table, item)
    return intervals


def coverage(
    observations: Sequence[CodeObservation],
    intervals: Mapping[tuple[str, str], tuple[float, float]],
) -> dict[str, Any]:
    """Share of scored codes whose truth lies inside their held-out range."""

    by_country: dict[str, list[bool]] = defaultdict(list)
    below: dict[str, int] = defaultdict(int)
    above: dict[str, int] = defaultdict(int)
    for item in observations:
        key = (item.country, item.code)
        if key not in intervals:
            continue
        low, high = intervals[key]
        by_country[item.country].append(low <= item.truth <= high)
        below[item.country] += item.truth < low
        above[item.country] += item.truth > high
    hits = [hit for values in by_country.values() for hit in values]
    countries = {
        country: {
            "codes": len(values),
            "coverage": sum(values) / len(values),
            "truth_below_range": below[country],
            "truth_above_range": above[country],
        }
        for country, values in sorted(by_country.items())
    }
    return {
        "codes": len(hits),
        "pooled": sum(hits) / len(hits) if hits else 0.0,
        "countries": countries,
    }


def gate(scores: Mapping[str, Any]) -> dict[str, Any]:
    low, high = GATE_POOLED
    pooled = scores["pooled"]
    failing = sorted(
        country
        for country, values in scores["countries"].items()
        if values["coverage"] < GATE_PER_COUNTRY
    )
    return {
        "pooled_range": [low, high],
        "per_country_minimum": GATE_PER_COUNTRY,
        "pooled_passes": low <= pooled <= high,
        "countries_below_minimum": failing,
        "passes": low <= pooled <= high and not failing and bool(scores["countries"]),
    }


def relative_to_level(
    observations: Sequence[CodeObservation], level: Mapping[str, float]
) -> list[CodeObservation]:
    """The same codes with each country's prediction scaled by a known level ratio.

    With the true level (truth over prediction for the whole country), only the
    error in the mix of codes is left: an upper bound on what anchoring the
    estimate's level to official totals could achieve, not an estimator.
    """

    return [
        CodeObservation(
            item.country,
            item.code,
            item.truth,
            item.predicted * level[item.country],
            item.pre_profile_share,
        )
        for item in observations
    ]

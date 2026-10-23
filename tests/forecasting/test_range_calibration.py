"""Measured ranges: learned from other countries only, gated before they ship.

Codes are fictional (`Q` prefix).
"""

from __future__ import annotations

import math

import pytest

from icor.forecasting.range_calibration import (
    CodeObservation,
    coverage,
    gate,
    held_out_intervals,
    ratio_table,
    relative_to_level,
    weighted_quantile,
)


def _observations(country: str, ratios: list[float], *, predicted: float = 2000.0):
    return [
        CodeObservation(country, f"Q{index:03d}", predicted * ratio, predicted, 0.0)
        for index, ratio in enumerate(ratios)
    ]


def test_weighted_quantile_reaches_the_cumulative_weight() -> None:
    values = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0)]

    assert weighted_quantile(values, 0.25) == 1.0
    assert weighted_quantile(values, 0.5) == 2.0
    assert weighted_quantile(values, 0.9) == 4.0
    assert weighted_quantile([(1.0, 3.0), (9.0, 1.0)], 0.7) == 1.0


def test_a_held_out_country_never_learns_from_its_own_codes() -> None:
    """If it did, a country biased far low would cover itself."""

    honest = [*_observations("AA", [1.0] * 60), *_observations("BB", [1.0] * 60)]
    biased = _observations("CC", [0.4] * 60)

    intervals = held_out_intervals([*honest, *biased], min_bin=10)
    scores = coverage([*honest, *biased], intervals)

    assert scores["countries"]["CC"]["coverage"] == 0.0
    assert scores["countries"]["CC"]["truth_below_range"] == 60
    assert gate(scores)["countries_below_minimum"] == ["CC"]
    assert gate(scores)["passes"] is False


def test_every_donor_country_carries_equal_weight() -> None:
    """A large register must not set everyone else's range."""

    large = _observations("AA", [2.0] * 900)
    small = _observations("BB", [0.5] * 100)

    table = ratio_table([*large, *small], min_bin=10)

    low, high = table["pooled"]
    assert math.isclose(low, math.log(0.5))
    assert math.isclose(high, math.log(2.0))


def test_codes_below_the_size_threshold_are_neither_scored_nor_ranged() -> None:
    small = _observations("AA", [1.0] * 5, predicted=999.0)
    large = _observations("BB", [1.0] * 5)

    intervals = held_out_intervals([*small, *large], min_bin=1)

    assert all(country == "BB" for country, _ in intervals)


def test_a_code_with_no_true_cars_counts_as_outside_its_range() -> None:
    donors = _observations("AA", [0.8, 1.0, 1.2] * 20)
    missing = [CodeObservation("BB", "Q900", 0.0, 5000.0, 0.0)]

    intervals = held_out_intervals([*donors, *missing], min_bin=1)
    scores = coverage([*donors, *missing], intervals)

    assert scores["countries"]["BB"]["coverage"] == 0.0


def test_a_calibrated_spread_passes_and_the_gate_needs_every_country() -> None:
    ratios = [math.exp((index - 49.5) / 25) for index in range(100)]
    observations = [
        *_observations("AA", ratios),
        *_observations("BB", ratios),
        *_observations("CC", ratios),
    ]

    scores = coverage(observations, held_out_intervals(observations, min_bin=10))

    assert 0.72 <= scores["pooled"] <= 0.88
    assert gate(scores)["passes"] is True


def test_level_scaling_leaves_only_the_mix_error() -> None:
    observations = _observations("AA", [0.5, 0.5])

    scaled = relative_to_level(observations, {"AA": 0.5})

    assert [item.log_ratio for item in scaled] == pytest.approx([0.0, 0.0])

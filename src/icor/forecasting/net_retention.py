"""Per-market net retention: the cars a market keeps beyond what the survival curve scraps.

The promoted survival curve is measured in the United Kingdom, where few cars
leave the fleet except to the scrapyard. Elsewhere lease and company cars are
exported young (Belgium, Luxembourg) or new cars are registered and re-exported
(Denmark, Germany), so the curve leaves more cars on the road than the national
statistics office counts. Net retention measures that shortfall against
Eurostat's stock by age band (`road_eqs_carage`) and its new registrations
(`road_eqr_carmot`), and only ever lowers the curve:

- per band, the ratio of official cars to the cars the curve leaves from the
  office's own new registrations, read under the office's band convention;
- per band, the largest ratio of the last three clean years, capped at one, and
  one where it lies within three percent of one;
- a curve over age through the band midpoints that reproduces those targets,
  made non-increasing (net retention is cumulative) and capped at one.

Markets whose curve would not have predicted the next three years better than
the uncalibrated curve (fitted to 2022, tested on 2023-2025) are not
calibrated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

#: The bands calibrated: the app serves cars up to twenty years old.
BANDS: tuple[tuple[str, int, int], ...] = (
    ("Y_LT2", 0, 1),
    ("Y2-5", 2, 4),
    ("Y5-10", 5, 9),
    ("Y10-20", 10, 19),
)
TOLERANCE = 0.03
RECENT_YEARS = 3
FIT_ITERATIONS = 200
#: No band may predict worse than the uncalibrated curve by more than this.
BACKTEST_SLACK = 0.02


@dataclass(frozen=True, slots=True)
class MarketYear:
    """One office's clean inputs for one stock year."""

    year: int
    #: Years every band edge moves up (1 when three registration years are "under 2").
    shift: int
    #: Official cars by band; a flagged or missing band is absent.
    official: Mapping[str, int]
    #: Usable new registrations by registration year.
    registrations: Mapping[int, int]
    #: The office's last band counts every car ten years and older.
    open_ended: bool = False


def band_ages(low: int, high: int, shift: int) -> range:
    """Ages a band covers under a convention (`data_coverage._band`)."""

    return range(0 if low == 0 else low + shift, high + shift + 1)


def modelled(
    item: MarketYear,
    low: int,
    high: int,
    share: Callable[[int], float],
    multiplier: Callable[[int], float] = lambda _age: 1.0,
) -> float | None:
    """Cars the curve leaves in a band, or None when a registration year is unusable."""

    total = 0.0
    for age in band_ages(low, high, item.shift):
        registrations = item.registrations.get(item.year - age)
        if registrations is None:
            return None
        total += registrations * share(age) * multiplier(age)
    return total


def band_ratios(item: MarketYear, share: Callable[[int], float]) -> dict[str, float]:
    ratios = {}
    for name, low, high in BANDS:
        official = item.official.get(name)
        if official is None or (item.open_ended and name == BANDS[-1][0]):
            continue
        value = modelled(item, low, high, share)
        if value:
            ratios[name] = official / value
    return ratios


def targets(
    years: Sequence[MarketYear], share: Callable[[int], float], *, until: int
) -> dict[str, float]:
    """Per band: one capped at the largest of its last three ratios, and one within tolerance."""

    by_band: dict[str, list[tuple[int, float]]] = {}
    for item in years:
        if item.year > until:
            continue
        for name, ratio in band_ratios(item, share).items():
            by_band.setdefault(name, []).append((item.year, ratio))
    result = {}
    for name, values in by_band.items():
        recent = sorted(values)[-RECENT_YEARS:]
        target = min(1.0, max(ratio for _year, ratio in recent))
        result[name] = 1.0 if target >= 1 - TOLERANCE else target
    return result


def _pool_non_increasing(values: list[float], weights: list[float]) -> list[float]:
    """Weighted pool-adjacent-violators: the closest non-increasing sequence."""

    blocks: list[list[float]] = []  # [value, weight, length]
    for value, weight in zip(values, weights, strict=True):
        blocks.append([value, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            value_b, weight_b, length_b = blocks.pop()
            value_a, weight_a, length_a = blocks.pop()
            total = weight_a + weight_b
            blocks.append(
                [(value_a * weight_a + value_b * weight_b) / total, total, length_a + length_b]
            )
    pooled: list[float] = []
    for value, _weight, length in blocks:
        pooled.extend([value] * length)
    return pooled


def _interpolate(knots: list[tuple[float, float]], age: int) -> float:
    if age <= knots[0][0]:
        return knots[0][1]
    for (age_a, value_a), (age_b, value_b) in zip(knots, knots[1:], strict=False):
        if age <= age_b:
            return value_a + (value_b - value_a) * (age - age_a) / (age_b - age_a)
    return knots[-1][1]


def fit_curve(
    goal: Mapping[str, float],
    latest: MarketYear,
    share: Callable[[int], float],
    *,
    support: int,
) -> tuple[float, ...] | None:
    """Multipliers for ages 0..support-1, or None when every target is one."""

    if all(value >= 1 for value in goal.values()):
        return None
    bands = [
        (name, band_ages(low, high, latest.shift)) for name, low, high in BANDS if name in goal
    ]
    weights = {
        name: [latest.registrations.get(latest.year - age, 0) * share(age) for age in ages]
        for name, ages in bands
    }
    # A band kept whole sits at its oldest age, so the decline toward a lowered
    # band never reaches into it; a lowered band is fitted through its midpoint.
    knots = [
        [ages[-1] if goal[name] >= 1 else (ages[0] + ages[-1]) / 2, goal[name]]
        for name, ages in bands
    ]
    masses = [max(sum(weights[name]), 1e-9) for name, _ages in bands]
    for _ in range(FIT_ITERATIONS):
        curve = [(age, value) for age, value in knots]
        for knot, (name, ages) in zip(knots, bands, strict=True):
            mass = sum(weights[name])
            if mass <= 0:
                continue
            achieved = (
                sum(
                    weight * _interpolate(curve, age)
                    for weight, age in zip(weights[name], ages, strict=True)
                )
                / mass
            )
            if achieved > 0:
                knot[1] *= goal[name] / achieved
        # Project back onto the constraints at every step: non-increasing, at most one.
        pooled = _pool_non_increasing([knot[1] for knot in knots], masses)
        for knot, value in zip(knots, pooled, strict=True):
            knot[1] = min(value, 1.0)
    capped = [1.0 if knot[1] >= 1 - TOLERANCE else knot[1] for knot in knots]
    if all(value >= 1 for value in capped):
        return None
    curve = [(knot[0], value) for knot, value in zip(knots, capped, strict=True)]
    return tuple(round(_interpolate(curve, age), 4) for age in range(support))


@dataclass(frozen=True, slots=True)
class Backtest:
    #: (year, band, official, uncalibrated, calibrated) for every predicted cell.
    cells: tuple[tuple[int, str, int, float, float], ...]

    def error(self, column: int) -> float:
        return sum(abs(cell[column] / cell[2] - 1) for cell in self.cells) / max(len(self.cells), 1)

    @property
    def passes(self) -> bool:
        if not self.cells or self.error(4) >= self.error(3):
            return False
        return all(
            abs(calibrated / official - 1) <= abs(base / official - 1) + BACKTEST_SLACK
            for _year, _band, official, base, calibrated in self.cells
        )


def backtest(
    years: Sequence[MarketYear],
    share: Callable[[int], float],
    *,
    fit_until: int,
    support: int,
) -> tuple[tuple[float, ...] | None, Backtest]:
    """Fit on years up to ``fit_until`` and predict the bands of every later year."""

    fitted = sorted((item for item in years if item.year <= fit_until), key=lambda item: item.year)
    if not fitted:
        return None, Backtest(())
    curve = fit_curve(targets(fitted, share, until=fit_until), fitted[-1], share, support=support)
    cells = []
    for item in years:
        if item.year <= fit_until:
            continue
        for name, low, high in BANDS:
            official = item.official.get(name)
            if official is None or (item.open_ended and name == BANDS[-1][0]):
                continue
            base = modelled(item, low, high, share)
            if not base:
                continue
            calibrated = (
                base
                if curve is None
                else modelled(item, low, high, share, lambda age: curve[min(age, len(curve) - 1)])
            )
            cells.append((item.year, name, official, base, calibrated or base))
    return curve, Backtest(tuple(cells))

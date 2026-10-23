"""Per-market age anchor: scale a market's cars to its statistics office's count by age.

New-car registrations aged by a survival curve miss what a market's own count
of cars on the road holds: young used imports (the Baltics, Czechia, the
Netherlands), early export (Germany, Denmark) and a scrapping rate unlike the
United Kingdom's, where the curve was measured. Eurostat's passenger cars by
age band (`road_eqs_carage`) is that count for every market. Within each band,
the anchor is the official count over the cars the model leaves from the
market's own registrations, so the model reproduces the band in the year it is
fitted to. Unlike net retention (`icor.forecasting.net_retention`) it may rise
above one and need not fall with age: used imports add cars.

A band is used only when the market's registrations cover every year it
spans; a band reaching before them would be compared with part of its cars.
Ages outside the usable bands take the nearest usable band's anchor. A market
is anchored only when anchors fitted to an earlier year predict every later
year's bands better than none (`backtest`), as net retention requires.

Every function is pure.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from icor.forecasting.net_retention import BANDS, Backtest, MarketYear, modelled

#: An anchor this close to one is one: within the bands' own noise.
TOLERANCE = 0.03


def band_anchors(item: MarketYear, share: Callable[[int], float]) -> dict[str, float]:
    """Official cars over modelled cars, per band the registrations fully cover."""

    anchors = {}
    for name in BANDS:
        official = item.official.get(name)
        if official is None or (item.open_ended and name == BANDS[-1]):
            continue
        value = modelled(item, name, share)
        if value:
            ratio = official / value
            anchors[name] = 1.0 if abs(ratio - 1) < TOLERANCE else ratio
    return anchors


def age_curve(item: MarketYear, anchors: Mapping[str, float], *, support: int) -> tuple[float, ...]:
    """The anchor at every age 0..support-1: its band's, else the nearest usable band's."""

    usable = [(item.ages(name), anchors[name]) for name in BANDS if name in anchors]
    if not usable:
        raise ValueError("no band can be anchored")
    curve = []
    for age in range(support):
        inside = [value for ages, value in usable if age in ages]
        if inside:
            curve.append(inside[0])
        elif age < usable[0][0].start:
            curve.append(usable[0][1])
        else:
            older = [value for ages, value in usable if ages.start <= age]
            curve.append(older[-1])
    return tuple(round(value, 4) for value in curve)


def fit(
    items: Sequence[MarketYear], share: Callable[[int], float], *, until: int, support: int
) -> tuple[float, ...] | None:
    """Anchors from the newest year up to ``until``, or None when every band is one."""

    fitted = sorted((item for item in items if item.year <= until), key=lambda item: item.year)
    for item in reversed(fitted):
        anchors = band_anchors(item, share)
        if anchors:
            if all(value == 1.0 for value in anchors.values()):
                return None
            return age_curve(item, anchors, support=support)
    return None


def backtest(
    items: Sequence[MarketYear],
    share: Callable[[int], float],
    *,
    fit_until: int,
    support: int,
) -> tuple[tuple[float, ...] | None, Backtest]:
    """Fit on years up to ``fit_until`` and predict the bands of every later year."""

    curve = fit(items, share, until=fit_until, support=support)
    cells = []
    for item in items:
        if item.year <= fit_until:
            continue
        for name in BANDS:
            official = item.official.get(name)
            if official is None or (item.open_ended and name == BANDS[-1]):
                continue
            base = modelled(item, name, share)
            if not base:
                continue
            anchored = (
                base
                if curve is None
                else modelled(item, name, share, lambda age: curve[min(age, len(curve) - 1)])
            )
            cells.append((item.year, name, official, base, anchored or base))
    return curve, Backtest(tuple(cells))

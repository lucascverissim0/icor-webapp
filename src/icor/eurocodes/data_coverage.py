"""How much evidence stands behind each Eurocode, and how many of its cars are on the road.

The ranking says how much demand a code carries; this says how far to trust
it. Every fleet cell of the snapshot (one vehicle spelling, market and
registration year) is allocated exactly as the ranking allocates demand, and
its cars are counted under the evidence that produced them: registrations a
publisher reported, a national register's count of the cars still on the
road, an interpolated gap, or a forecast year.

A code's cars are its certain cars (cells only it can explain) plus, reported
apart and never split, the cars it shares with other codes. Its depth tier
reads the evidence behind both, because a shared launch year is still evidence
about the code.

Two blind spots are invisible from inside the fleet and are reported apart:
catalogue codes no reviewed vehicle carries (no data at all), and, per market,
the cars the official national stock counts that the model does not hold
(registrations before the first source year, and used imports, which new-car
registrations never see).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, NamedTuple

from icor.domain.age_bands import (
    AGE_BANDS,
    BAND_LAYOUTS,
    THREE_YEARS,
    TWO_FOUR_YEARS,
    TWO_YEARS,
    UNCLEAR,
    band_layout,
)
from icor.eurocodes.catalog import CatalogueRow
from icor.eurocodes.runtime import EurocodeMap

__all__ = [
    "AGE_BANDS",
    "BAND_LAYOUTS",
    "THREE_YEARS",
    "TWO_FOUR_YEARS",
    "TWO_YEARS",
    "UNCLEAR",
    "band_layout",
]

DOCUMENT_FORMAT = "icor-eurocode-data-coverage"
DOCUMENT_VERSION = 2

#: Evidence behind a fleet cell, strongest first: a national register's count of
#: the cars on the road (stock), reported registrations, an interpolated gap, a forecast.
EVIDENCE_KINDS = ("stock", "registrations", "interpolated", "forecast")
OBSERVED_KINDS = frozenset({"stock", "registrations"})

_STATUSES = ("single", "shared", "outside", "unassignable", "unreviewed")

#: Depth tiers. A code reaches A only with mostly observed cars in several
#: markets: one market's register, however complete, says nothing of the rest.
TIER_A_OBSERVED_SHARE = Decimal("0.8")
TIER_A_MARKETS = 5
TIER_B_OBSERVED_SHARE = Decimal("0.5")

#: Model registrations by name start in 2010 for most markets (EEA monitoring).
FIRST_EEA_YEAR = 2010
#: A code with at least this share of its life before 2010 is tier C at best.
PRE_2010_SHARE_CAP_C = Decimal("0.5")
#: A code launched this many years before the report year or later is new: its
#: cars are mostly the forecast year's, so a low observed share is expected.
NEW_CODE_YEARS = 2

_ZERO = Decimal(0)


class FleetRange(NamedTuple):
    p10: Decimal
    p50: Decimal
    p90: Decimal


@dataclass(frozen=True, slots=True)
class FleetCell:
    """Cars of one spelling, market and registration year, under one kind of evidence."""

    identity: str
    raw_model: str
    geography: str
    cohort_year: int
    evidence: str
    #: Cars on the road at the report year; None for a cohort registered later.
    fleet: FleetRange | None
    #: Cars on the road at 2028 as the snapshot stores them.
    fleet_2028: Decimal
    #: Publishers whose observations this cell rests on (none for a forecast).
    sources: frozenset[str] = frozenset()
    #: The register counts only part of these cars (it masks rare model names),
    #: so they are at least this many.
    lower_bound: bool = False


@dataclass(frozen=True, slots=True)
class OfficialStock:
    """A national statistics office's count of passenger cars on the road."""

    geography: str
    year: int
    total: int
    #: Cars by Eurostat age band in ``band_year``; empty when no year reports them all.
    bands: Mapping[str, int] = field(default_factory=dict)
    #: The publisher's own territory code when it differs (UK, counted for GB).
    publisher_geography: str | None = None
    #: Eurostat status flags of the total's year (``i``, ``p``, ``b``) by age key.
    status: Mapping[str, str] = field(default_factory=dict)
    #: The newest year, up to ``year``, whose age bands are all reported.
    band_year: int | None = None
    #: The total of ``band_year``, which the bands sum to.
    band_total: int | None = None
    band_status: Mapping[str, str] = field(default_factory=dict)
    #: The office publishes no band over twenty years: its "Y10-20" counts every car
    #: ten years and older.
    open_ended: bool = False


@dataclass
class _CodeTally:
    certain: list[Decimal] = field(default_factory=lambda: [_ZERO, _ZERO, _ZERO])
    shared: Decimal = _ZERO
    certain_2028: Decimal = _ZERO
    shared_2028: Decimal = _ZERO
    shared_with: set[str] = field(default_factory=set)
    shared_with_uncoded: bool = False
    evidence: dict[str, Decimal] = field(default_factory=lambda: defaultdict(Decimal))
    markets: dict[str, Decimal] = field(default_factory=lambda: defaultdict(Decimal))
    observed_markets: set[str] = field(default_factory=set)
    observed_years: set[int] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    lower_bound: Decimal = _ZERO
    families: tuple[str, ...] = ()


def _whole(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _share(part: Decimal, whole: Decimal) -> float | None:
    return None if not whole else round(float(part / whole), 4)


def _band(age: int, convention: str | None = TWO_YEARS, *, open_ended: bool = False) -> str | None:
    """The Eurostat band of a cohort ``age`` years old, under an office's convention."""

    if age < 0:
        return None
    for name, low, high in band_layout(convention):
        if age >= low and (high is None or age <= high):
            if open_ended and name == AGE_BANDS[-1][0]:
                return AGE_BANDS[-2][0]
            return name
    return None


@dataclass(frozen=True, slots=True)
class CountryComparison:
    """One market's official stock and what the model and other evidence say of it.

    Every mapping is comparison year -> registration year -> cars (P50).
    ``before_sources`` holds the survivors of the office's own new registrations
    in years the model has no model-level registrations for, and ``unnamed`` the
    survivors of new registrations the model's sources leave without a model
    name, in years they do report. Both are estimates for the country total only:
    they are never assigned to a vehicle or a code.
    """

    stock: OfficialStock
    modelled: Mapping[int, Mapping[int, Decimal]]
    #: The band convention of ``stock.band_year``: "2y", "3y", "2y-4y" or "unclear".
    convention: str | None = None
    before_sources: Mapping[int, Mapping[int, Decimal]] = field(default_factory=dict)
    unnamed: Mapping[int, Mapping[int, Decimal]] = field(default_factory=dict)
    #: Registration years estimated from official new registrations that include used imports.
    imports_in_registrations: frozenset[int] = frozenset()


def _split(gap: Decimal, before: Decimal, unnamed: Decimal) -> dict[str, int]:
    """Cars the official count holds and the model does not, by likely reason.

    The two estimates are capped by the gap they explain, in that order; the
    rest is used imports, definitions and anything else. A negative gap is
    cars the model holds above the official count.
    """

    if gap <= 0:
        return {
            "missing": 0,
            "before_sources": 0,
            "unnamed": 0,
            "other": 0,
            "above_official": _whole(-gap),
        }
    explained_before = min(before, gap)
    explained_unnamed = min(unnamed, gap - explained_before)
    return {
        "missing": _whole(gap),
        "before_sources": _whole(explained_before),
        "unnamed": _whole(explained_unnamed),
        "other": _whole(gap - explained_before - explained_unnamed),
        "above_official": 0,
    }


def _by_band(
    cohorts: Mapping[int, Decimal], year: int, convention: str | None, open_ended: bool
) -> dict[str, Decimal]:
    totals: dict[str, Decimal] = defaultdict(Decimal)
    for cohort, units in cohorts.items():
        name = _band(year - cohort, convention, open_ended=open_ended)
        if name is not None:
            totals[name] += units
    return totals


def _country_gap(comparison: CountryComparison) -> dict[str, Any]:
    stock = comparison.stock
    modelled = comparison.modelled.get(stock.year, {})
    modelled_total = sum(modelled.values(), _ZERO)
    before_total = sum(comparison.before_sources.get(stock.year, {}).values(), _ZERO)
    unnamed_total = sum(comparison.unnamed.get(stock.year, {}).values(), _ZERO)
    entry: dict[str, Any] = {
        "geography": stock.geography,
        "publisher_geography": stock.publisher_geography or stock.geography,
        "year": stock.year,
        "official": stock.total,
        "official_status": stock.status.get("TOTAL"),
        "modelled": _whole(modelled_total),
        "modelled_share": _share(modelled_total, Decimal(stock.total)),
        "estimated": _split(Decimal(stock.total) - modelled_total, before_total, unnamed_total),
        "band_year": stock.band_year,
        "band_official_total": stock.band_total,
        "convention": comparison.convention,
        "open_ended": stock.open_ended,
        "bands": [],
        "under_20": None,
        "registrations_include_used_imports": sorted(comparison.imports_in_registrations),
    }
    band_year = stock.band_year
    if band_year is None:
        return entry
    convention = comparison.convention
    model_bands = _by_band(
        comparison.modelled.get(band_year, {}), band_year, convention, stock.open_ended
    )
    alternative = (
        _by_band(comparison.modelled.get(band_year, {}), band_year, THREE_YEARS, stock.open_ended)
        if convention == UNCLEAR
        else None
    )
    before_bands = _by_band(
        comparison.before_sources.get(band_year, {}), band_year, convention, stock.open_ended
    )
    unnamed_bands = _by_band(
        comparison.unnamed.get(band_year, {}), band_year, convention, stock.open_ended
    )
    for name, _low, _high in AGE_BANDS:
        if stock.open_ended and name == AGE_BANDS[-1][0]:
            continue
        official = stock.bands[name]
        model_units = model_bands.get(name, _ZERO)
        entry["bands"].append(
            {
                "band": name,
                "official": official,
                "status": stock.band_status.get(name),
                "modelled": _whole(model_units),
                "modelled_alternative": (
                    _whole(alternative.get(name, _ZERO)) if alternative is not None else None
                ),
                "estimated": _split(
                    Decimal(official) - model_units,
                    before_bands.get(name, _ZERO),
                    unnamed_bands.get(name, _ZERO),
                ),
            }
        )
    if not stock.open_ended:
        # Cars under twenty years, summed band by band so that an excess in one
        # band does not hide a gap in another.
        younger = entry["bands"][:-1]
        official_under_20 = sum(band["official"] for band in younger)
        modelled_under_20 = sum((model_bands.get(band["band"], _ZERO) for band in younger), _ZERO)
        entry["under_20"] = {
            "official": official_under_20,
            "modelled": _whole(modelled_under_20),
            "modelled_share": _share(modelled_under_20, Decimal(official_under_20)),
            "estimated": {
                key: sum(band["estimated"][key] for band in younger)
                for key in ("missing", "before_sources", "unnamed", "other", "above_official")
            },
        }
    return entry


def depth_tier(
    observed_share: Decimal | None, observed_markets: int, pre_2010_share: Decimal = _ZERO
) -> str:
    """A code's evidence depth, capped by the years most markets cannot see.

    Before 2010 only a few markets report cars by model, so a code's cars from
    those years are missing from every other market rather than unobserved in
    it. Its observed share then overstates what is known: a code with any
    production before 2010 reaches B at most, and one with half its life or
    more before 2010 reaches C at most.
    """

    if observed_share is None:
        return "D"
    if observed_share >= TIER_A_OBSERVED_SHARE and observed_markets >= TIER_A_MARKETS:
        tier = "A"
    elif observed_share >= TIER_B_OBSERVED_SHARE:
        tier = "B"
    else:
        tier = "C"
    if pre_2010_share >= PRE_2010_SHARE_CAP_C:
        return max(tier, "C")
    if pre_2010_share > 0:
        return max(tier, "B")
    return tier


def _catalogue_span(rows: Iterable[CatalogueRow]) -> tuple[int | None, int | None]:
    starts = [year for row in rows for year in (row.start, row.sop) if year]
    ends = [year for row in rows for year in (row.end, row.eop) if year]
    return (min(starts) if starts else None, max(ends) if ends else None)


def build_report(
    eurocodes: EurocodeMap,
    cells: Iterable[FleetCell],
    *,
    year: int,
    snapshot_id: str,
    catalogue_rows: Iterable[CatalogueRow] = (),
    countries: Mapping[str, CountryComparison] | None = None,
    official_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The data-coverage document for one snapshot and report year.

    ``countries`` holds, per market, the official stock and the model's own
    fleet per registration year at the stock's years, so the two can be
    compared band by band, with the estimated reasons for any gap.

    The totals are checked before anything is rounded: a code's certain cars
    must sum to the single-code cars of every market, and each market's cars
    by outcome and by evidence must equal its total. A failed check raises,
    because a report that does not reconcile would be read as true.
    """

    tallies: dict[str, _CodeTally] = defaultdict(_CodeTally)
    market_status: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    market_evidence: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    market_total: dict[str, Decimal] = defaultdict(Decimal)
    market_lower_bound: dict[str, Decimal] = defaultdict(Decimal)
    pre_2010_markets: set[str] = set()

    for cell in cells:
        if cell.evidence not in EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence kind {cell.evidence!r}")
        if cell.fleet is None:
            continue
        units = cell.fleet.p50
        if not units:
            continue
        if cell.cohort_year < FIRST_EEA_YEAR:
            pre_2010_markets.add(cell.geography)
        allocation = eurocodes.allocate(
            cell.identity, cell.cohort_year, cell.geography, raw_model=cell.raw_model
        )
        market_total[cell.geography] += units
        market_status[cell.geography][allocation.status] += units
        market_evidence[cell.geography][cell.evidence] += units
        if cell.lower_bound:
            market_lower_bound[cell.geography] += units
        observed = cell.evidence in OBSERVED_KINDS
        for code in allocation.codes:
            tally = tallies[code]
            if not tally.families:
                tally.families = allocation.families
            if allocation.status == "single":
                tally.certain[0] += cell.fleet.p10
                tally.certain[1] += cell.fleet.p50
                tally.certain[2] += cell.fleet.p90
                tally.certain_2028 += cell.fleet_2028
            else:
                tally.shared += units
                tally.shared_2028 += cell.fleet_2028
                tally.shared_with.update(other for other in allocation.codes if other != code)
                tally.shared_with_uncoded |= allocation.uncoded
            tally.evidence[cell.evidence] += units
            tally.markets[cell.geography] += units
            if cell.lower_bound:
                tally.lower_bound += units
            if observed:
                tally.observed_markets.add(cell.geography)
                tally.observed_years.add(cell.cohort_year)
                tally.sources.update(cell.sources)

    single_total = sum((status["single"] for status in market_status.values()), _ZERO)
    certain_total = sum((tally.certain[1] for tally in tallies.values()), _ZERO)
    if certain_total != single_total:
        raise ValueError("codes' certain cars do not sum to the single-code cars")
    for geography, total in market_total.items():
        if sum(market_status[geography].values(), _ZERO) != total:
            raise ValueError(f"{geography}: cars by outcome do not sum to the market")
        if sum(market_evidence[geography].values(), _ZERO) != total:
            raise ValueError(f"{geography}: cars by evidence do not sum to the market")

    codes = []
    for code, tally in tallies.items():
        summary = eurocodes.describe(code, tally.families) or eurocodes.summary(code)
        if summary is None:
            raise ValueError("an allocated code has no reviewed summary")
        whole = sum(tally.evidence.values(), _ZERO)
        observed = tally.evidence["stock"] + tally.evidence["registrations"]
        observed_share = observed / whole if whole else None
        last = summary.last_year if summary.last_year is not None else year
        life = range(summary.first_year, min(last, year) + 1)
        years_before_2010 = max(0, min(FIRST_EEA_YEAR, last + 1) - summary.first_year)
        pre_2010_share = Decimal(min(years_before_2010, len(life))) / len(life) if life else _ZERO
        flags = []
        if summary.first_year < FIRST_EEA_YEAR:
            flags.append("pre-2010")
        if summary.first_year >= year - NEW_CODE_YEARS:
            flags.append("new")
        if not tally.certain[1]:
            flags.append("shared-only")
        if len(tally.observed_markets) == 1:
            flags.append("one-market")
        if tally.lower_bound:
            flags.append("lower-bound")
        codes.append(
            {
                "code": code,
                "label": summary.label,
                "families": list(summary.families),
                "first_year": summary.first_year,
                "last_year": summary.last_year,
                "drive": summary.drive,
                "fleet": {
                    "certain_p10": _whole(tally.certain[0]),
                    "certain_p50": _whole(tally.certain[1]),
                    "certain_p90": _whole(tally.certain[2]),
                    "shared_p50": _whole(tally.shared),
                    "certain_2028": _whole(tally.certain_2028),
                    "shared_2028": _whole(tally.shared_2028),
                },
                "shared_with": sorted(tally.shared_with),
                "shared_with_uncoded": tally.shared_with_uncoded,
                "evidence": {kind: _whole(tally.evidence[kind]) for kind in EVIDENCE_KINDS},
                "observed_share": _share(observed, whole),
                "lower_bound_p50": _whole(tally.lower_bound),
                "markets": len(tally.markets),
                "observed_markets": sorted(tally.observed_markets),
                "sources": sorted(tally.sources),
                "life_years": len(life),
                "observed_life_years": len(tally.observed_years.intersection(life)),
                "years_before_2010": years_before_2010,
                "tier": depth_tier(observed_share, len(tally.observed_markets), pre_2010_share),
                "flags": flags,
            }
        )

    present = set(tallies)
    reviewed = set(eurocodes.summaries())
    by_code: dict[str, list[CatalogueRow]] = defaultdict(list)
    for row in catalogue_rows:
        by_code[row.code].append(row)
    no_data = []
    for code in sorted(set(by_code) | reviewed):
        if code in present:
            continue
        rows = by_code.get(code, [])
        if rows and not any(row.scope == "passenger" for row in rows):
            continue
        summary = eurocodes.summary(code)
        first, last = (summary.first_year, summary.last_year) if summary else _catalogue_span(rows)
        no_data.append(
            {
                "code": code,
                "label": summary.label if summary else None,
                "makes": sorted({make for row in rows for make in row.makes}),
                "descriptions": sorted({row.description for row in rows})[:3],
                "first_year": first,
                "last_year": last,
                "reviewed": summary is not None,
                "tier": "D",
            }
        )

    markets = [
        {
            "geography": geography,
            "fleet": _whole(total),
            "by_outcome": {
                status: _whole(market_status[geography][status]) for status in _STATUSES
            },
            "by_evidence": {
                kind: _whole(market_evidence[geography][kind]) for kind in EVIDENCE_KINDS
            },
            "has_pre_2010_cohorts": geography in pre_2010_markets,
            "lower_bound": _whole(market_lower_bound[geography]),
        }
        for geography, total in sorted(market_total.items())
    ]

    gap = [_country_gap(item) for _geography, item in sorted((countries or {}).items())]

    tiers: dict[str, dict[str, int]] = {
        tier: {"codes": 0, "certain_p50": 0} for tier in ("A", "B", "C", "D")
    }
    for item in codes:
        tiers[item["tier"]]["codes"] += 1
        tiers[item["tier"]]["certain_p50"] += item["fleet"]["certain_p50"]
    tiers["D"]["codes"] += len(no_data)

    whole_fleet = sum(market_total.values(), _ZERO)
    codes.sort(key=lambda item: (-item["fleet"]["certain_p50"], item["code"]))
    return {
        "format": DOCUMENT_FORMAT,
        "version": DOCUMENT_VERSION,
        "snapshot_id": snapshot_id,
        "catalogue_sha256": eurocodes.catalogue_sha256,
        "decisions_snapshot_id": eurocodes.decisions.metadata.get("snapshot_id"),
        "year": year,
        "thresholds": {
            "tier_a_observed_share": float(TIER_A_OBSERVED_SHARE),
            "tier_a_markets": TIER_A_MARKETS,
            "tier_b_observed_share": float(TIER_B_OBSERVED_SHARE),
            "pre_2010_share_cap_c": float(PRE_2010_SHARE_CAP_C),
        },
        "totals": {
            "fleet": _whole(whole_fleet),
            "by_outcome": {
                status: _whole(sum((m[status] for m in market_status.values()), _ZERO))
                for status in _STATUSES
            },
            "by_evidence": {
                kind: _whole(sum((m[kind] for m in market_evidence.values()), _ZERO))
                for kind in EVIDENCE_KINDS
            },
            "tiers": tiers,
            "no_data": {
                "reviewed": sum(1 for item in no_data if item["reviewed"]),
                "not_reviewed": sum(1 for item in no_data if not item["reviewed"]),
                "first_year_from_2000": sum(
                    1 for item in no_data if (item["first_year"] or 0) >= 2000
                ),
            },
        },
        "codes": codes,
        "no_data": no_data,
        "markets": markets,
        "country_gap": gap,
        "official_source": dict(official_source or {}),
    }

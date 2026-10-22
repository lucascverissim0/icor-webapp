"""Read a snapshot's fleet cells at any year, with the evidence behind each.

The snapshot stores cars on the road at its forecast horizons only. A cohort's
cars at another year follow from the same promoted survival curve the build
used: registrations times the share still on the road at that age, or, for a
cohort a national register counted, the register's count aged from the
register's date. Before any derived number is trusted, every stored 2028 value
is recomputed the same way and must match to the last stored decimal; a
snapshot built with another curve is refused rather than reported on.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from decimal import Decimal

from icor.domain.evidence import STOCK_LOWER_BOUND_REASON
from icor.eurocodes.data_coverage import CountryComparison, FleetCell, FleetRange, OfficialStock
from icor.eurocodes.runtime import identity_key
from icor.evidence.eurostat_new_registrations import NewRegistrations, includes_used_imports
from icor.forecasting.survival import CalibratedCohortSurvivalModel, NetRetainedSurvivalModel
from icor.infrastructure.snapshot_identity import load_identity_index

STORED_HORIZON = 2028

_EVIDENCE = {
    "reconciled-registration-cohort": "registrations",
    "observed-stock-anchored-cohort": "stock",
    "estimated-registration-cohort": "interpolated",
    "forecast-registration-cohort": "forecast",
}
_QUANTUM = Decimal("0.0001")
_SCALE = Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class FleetRead:
    """Fleet cells at the report year, and some markets' cars by cohort at their own years."""

    cells: tuple[FleetCell, ...]
    #: market -> comparison year -> registration year -> cars on the road (P50).
    comparison: dict[str, dict[int, dict[int, Decimal]]]
    #: Cohorts whose stored 2028 value was recomputed and matched.
    verified_cohorts: int
    #: market -> registration year -> the evidence behind the market's cars of that
    #: year: "registrations", "stock" and, for a register that masks names, "lower-bound".
    cohort_evidence: dict[str, dict[int, frozenset[str]]] = field(default_factory=dict)


def _fleet(
    survival: CalibratedCohortSurvivalModel | NetRetainedSurvivalModel,
    stored: Decimal,
    *,
    geography: str,
    cohort_year: int,
    year: int,
    stock_year: int | None,
) -> FleetRange | None:
    """Cars of one cohort at ``year``, as the planning build would have computed them.

    ``stored`` is registrations for a registration cohort and the register's
    count for a register cohort; the latter mirrors
    `GenerationPlanningService._conditional_fleet` forward in time. Before the
    register's own year it is the register's count: aging it back with the
    curve would add cars the register never saw, and the curve, measured in
    the United Kingdom, overstates how many such cars another country scraps.
    """

    age = year - cohort_year
    if age < 0:
        return None
    if stock_year is None:
        interval = survival.interval(stored, age_years=age, geography=geography)
        return FleetRange(interval.p10, interval.p50, interval.p90)
    if year < stock_year:
        return FleetRange(stored, stored, stored)
    at_stock = survival.interval(
        _SCALE, age_years=max(stock_year - cohort_year, 0), geography=geography
    )
    at_year = survival.interval(_SCALE, age_years=age, geography=geography)
    if min(at_stock.p10, at_stock.p50, at_stock.p90) <= 0:
        return None
    return FleetRange(
        *(
            (stored * share / stock_share).quantize(_QUANTUM)
            for share, stock_share in (
                (at_year.p10, at_stock.p10),
                (at_year.p50, at_stock.p50),
                (at_year.p90, at_stock.p90),
            )
        )
    )


def _stock_years(connection: sqlite3.Connection) -> dict[str, int]:
    """Each market's newest register date: the age its counted cars were seen at."""

    return {
        geography: int(str(newest)[:4])
        for geography, newest in connection.execute(
            "SELECT geography, MAX(period_end) FROM observation "
            "WHERE measure = 'active_fleet' AND registration_cohort_year IS NOT NULL "
            "GROUP BY geography"
        )
    }


def _observed_sources(connection: sqlite3.Connection) -> dict[str, frozenset[str]]:
    """Publishers behind each observed 2028 cohort, through its input observations."""

    sources: dict[str, set[str]] = defaultdict(set)
    for cohort_id, source_id in connection.execute(
        """SELECT DISTINCT c.cohort_id, r.source_id
        FROM cohort_estimate c
        JOIN cohort_input i ON i.cohort_id = c.cohort_id
        JOIN observation o ON o.observation_id = i.observation_id
        JOIN source_release r ON r.release_id = o.release_id
        WHERE c.as_of_year = ?
          AND (c.reason_codes LIKE '["reconciled-registration-cohort"%'
            OR c.reason_codes LIKE '["observed-stock-anchored-cohort"%')""",
        (STORED_HORIZON,),
    ):
        sources[cohort_id].add(source_id)
    return {cohort_id: frozenset(ids) for cohort_id, ids in sources.items()}


def read_fleet(
    connection: sqlite3.Connection,
    survival: CalibratedCohortSurvivalModel | NetRetainedSurvivalModel,
    *,
    year: int,
    comparison_years: Mapping[str, Collection[int]],
) -> FleetRead:
    """Every fleet cell at ``year``, and each market's cohorts at its comparison years.

    ``comparison_years`` names, per market, the years an official stock count
    is compared at; markets differ (the United Kingdom stopped reporting to
    Eurostat earlier than the member states, and an office may publish a total
    a year before its age bands).

    Raises when a stored 2028 value cannot be reproduced, when a cohort's
    evidence is not one of the known kinds, or when a register cohort's market
    has no register date.
    """

    requested = [item for years in comparison_years.values() for item in years]
    if max((year, *requested)) > STORED_HORIZON:
        raise ValueError(f"years after {STORED_HORIZON} need registrations forecast past it")
    index = load_identity_index(connection)
    labels = {
        vehicle_id: (make, model)
        for vehicle_id, make, model in connection.execute(
            "SELECT vehicle_id, make, model FROM canonical_vehicle "
            "WHERE vehicle_id IN (SELECT DISTINCT canonical_vehicle_id FROM cohort_estimate)"
        )
    }
    keys = {
        vehicle_id: identity_key(index, vehicle_id, make, model)
        for vehicle_id, (make, model) in labels.items()
    }
    stock_years = _stock_years(connection)
    sources = _observed_sources(connection)

    fleet: dict[tuple[str, str, int, str], list[Decimal]] = {}
    fleet_2028: dict[tuple[str, str, int, str], Decimal] = defaultdict(Decimal)
    cell_sources: dict[tuple[str, str, int, str], set[str]] = defaultdict(set)
    lower_bound: set[tuple[str, str, int, str]] = set()
    comparison: dict[str, dict[int, dict[int, Decimal]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Decimal))
    )
    cohort_evidence: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    verified = 0
    for (
        cohort_id,
        vehicle_id,
        geography,
        cohort_year,
        reason_codes,
        stored,
        p10,
        p50,
        p90,
    ) in connection.execute(
        """SELECT cohort_id, canonical_vehicle_id, geography, registration_cohort_year,
            reason_codes, registrations, active_fleet_p10, active_fleet_p50,
            active_fleet_p90
        FROM cohort_estimate WHERE as_of_year = ?""",
        (STORED_HORIZON,),
    ):
        codes = json.loads(reason_codes)
        first_code = codes[0]
        evidence = _EVIDENCE.get(first_code)
        if evidence is None:
            raise ValueError(f"unknown cohort evidence {first_code!r}")
        stock_year = None
        if evidence == "stock":
            stock_year = stock_years.get(geography)
            if stock_year is None:
                raise ValueError(f"{geography}: register cohort without a register date")
        stored_value = Decimal(stored)
        recomputed = _fleet(
            survival,
            stored_value,
            geography=geography,
            cohort_year=cohort_year,
            year=STORED_HORIZON,
            stock_year=stock_year,
        )
        if recomputed != FleetRange(Decimal(p10), Decimal(p50), Decimal(p90)):
            raise ValueError(
                "a stored 2028 fleet does not match the promoted survival curve; "
                "the snapshot was built with another curve"
            )
        verified += 1
        key = (vehicle_id, geography, cohort_year, evidence)
        fleet_2028[key] += Decimal(p50)
        cell_sources[key].update(sources.get(cohort_id, ()))
        if STOCK_LOWER_BOUND_REASON in codes:
            lower_bound.add(key)
            cohort_evidence[geography][cohort_year].add("lower-bound")
        elif evidence in ("registrations", "stock"):
            cohort_evidence[geography][cohort_year].add(evidence)
        at_year = _fleet(
            survival,
            stored_value,
            geography=geography,
            cohort_year=cohort_year,
            year=year,
            stock_year=stock_year,
        )
        if at_year is not None:
            total = fleet.setdefault(key, [Decimal(0)] * 3)
            for position, value in enumerate(at_year):
                total[position] += value
        for comparison_year in comparison_years.get(geography, ()):
            at_comparison = _fleet(
                survival,
                stored_value,
                geography=geography,
                cohort_year=cohort_year,
                year=comparison_year,
                stock_year=stock_year,
            )
            if at_comparison is not None:
                comparison[geography][comparison_year][cohort_year] += at_comparison.p50

    cells = tuple(
        FleetCell(
            identity=keys[vehicle_id],
            raw_model=labels[vehicle_id][1],
            geography=geography,
            cohort_year=cohort_year,
            evidence=evidence,
            fleet=FleetRange(*fleet[key]) if key in fleet else None,
            fleet_2028=units,
            sources=frozenset(cell_sources[key]),
            lower_bound=key in lower_bound,
        )
        for key, units in sorted(fleet_2028.items())
        for vehicle_id, geography, cohort_year, evidence in (key,)
    )
    return FleetRead(
        cells=cells,
        comparison={
            geography: {item: dict(cohorts) for item, cohorts in years.items()}
            for geography, years in comparison.items()
        },
        verified_cohorts=verified,
        cohort_evidence={
            geography: {cohort: frozenset(kinds) for cohort, kinds in cohorts.items()}
            for geography, cohorts in cohort_evidence.items()
        },
    )


#: The survival curve's support: older cohorts have no share on the road.
_MAX_AGE = 40


def country_comparisons(
    fleet: FleetRead,
    stocks: Mapping[str, OfficialStock],
    registrations: Mapping[str, Mapping[int, NewRegistrations]],
    conventions: Mapping[str, str | None],
    share: Callable[[str, int], Decimal],
) -> dict[str, CountryComparison]:
    """Each market's official stock beside the model, with estimates for the gap.

    For every registration year the office reports new cars for, those cars'
    survivors (``share`` gives the fraction still on the road at an age) are
    compared with what the model holds of that year:

    * no model-level registrations that year (only a register that masks
      names, or nothing): the excess is cars registered before the sources;
    * registrations that year: the excess is cars the sources leave unnamed,
      unless the office's count includes used imports or differs in definition
      (flag ``d``), when it cannot tell unnamed cars from imports;
    * a register's full count that year: nothing, since it counts every car,
      used imports included.
    """

    comparisons = {}
    for geography, stock in stocks.items():
        years = {stock.year, *((stock.band_year,) if stock.band_year is not None else ())}
        modelled = {year: fleet.comparison.get(geography, {}).get(year, {}) for year in years}
        evidence = fleet.cohort_evidence.get(geography, {})
        official = registrations.get(geography, {})
        before: dict[int, dict[int, Decimal]] = {year: {} for year in years}
        unnamed: dict[int, dict[int, Decimal]] = {year: {} for year in years}
        imports = set()
        for cohort, item in official.items():
            kinds = evidence.get(cohort, frozenset())
            if "stock" in kinds:
                continue
            with_imports = includes_used_imports(geography, cohort)
            for year in years:
                age = year - cohort
                if not 0 <= age <= _MAX_AGE:
                    continue
                excess = Decimal(item.count) * share(geography, age) - modelled[year].get(
                    cohort, Decimal(0)
                )
                if excess <= 0:
                    continue
                if "registrations" not in kinds:
                    before[year][cohort] = excess
                    if with_imports:
                        imports.add(cohort)
                elif not with_imports and item.status != "d":
                    unnamed[year][cohort] = excess
        comparisons[geography] = CountryComparison(
            stock,
            modelled,
            convention=conventions.get(geography),
            before_sources=before,
            unnamed=unnamed,
            imports_in_registrations=frozenset(imports),
        )
    return comparisons

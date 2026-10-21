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
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from icor.eurocodes.data_coverage import FleetCell, FleetRange
from icor.eurocodes.runtime import identity_key
from icor.forecasting.survival import CalibratedCohortSurvivalModel
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
    """Fleet cells at the report year, and some markets' cars by cohort at their own year."""

    cells: tuple[FleetCell, ...]
    #: market -> registration year -> cars on the road (P50) at that market's
    #: comparison year.
    comparison: dict[str, dict[int, Decimal]]
    #: Cohorts whose stored 2028 value was recomputed and matched.
    verified_cohorts: int


def _fleet(
    survival: CalibratedCohortSurvivalModel,
    stored: Decimal,
    *,
    cohort_year: int,
    year: int,
    stock_year: int | None,
) -> FleetRange | None:
    """Cars of one cohort at ``year``, as the planning build would have computed them.

    ``stored`` is registrations for a registration cohort and the register's
    count for a register cohort; the latter mirrors
    `GenerationPlanningService._conditional_fleet`.
    """

    age = year - cohort_year
    if age < 0:
        return None
    if stock_year is None:
        interval = survival.interval(stored, age_years=age)
        return FleetRange(interval.p10, interval.p50, interval.p90)
    at_stock = survival.interval(_SCALE, age_years=max(stock_year - cohort_year, 0))
    at_year = survival.interval(_SCALE, age_years=age)
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
    survival: CalibratedCohortSurvivalModel,
    *,
    year: int,
    comparison_years: Mapping[str, int],
) -> FleetRead:
    """Every fleet cell at ``year``, and each market's cohorts at its comparison year.

    ``comparison_years`` names, per market, the year an official stock count
    is compared at; markets differ (the United Kingdom stopped reporting to
    Eurostat earlier than the member states).

    Raises when a stored 2028 value cannot be reproduced, when a cohort's
    evidence is not one of the known kinds, or when a register cohort's market
    has no register date.
    """

    if max((year, *comparison_years.values())) > STORED_HORIZON:
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
    comparison: dict[str, dict[int, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
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
        first_code = json.loads(reason_codes)[0]
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
        at_year = _fleet(
            survival, stored_value, cohort_year=cohort_year, year=year, stock_year=stock_year
        )
        if at_year is not None:
            total = fleet.setdefault(key, [Decimal(0)] * 3)
            for position, value in enumerate(at_year):
                total[position] += value
        comparison_year = comparison_years.get(geography)
        if comparison_year is None:
            continue
        at_comparison = _fleet(
            survival,
            stored_value,
            cohort_year=cohort_year,
            year=comparison_year,
            stock_year=stock_year,
        )
        if at_comparison is not None:
            comparison[geography][cohort_year] += at_comparison.p50

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
        )
        for key, units in sorted(fleet_2028.items())
        for vehicle_id, geography, cohort_year, evidence in (key,)
    )
    return FleetRead(
        cells=cells,
        comparison={geography: dict(years) for geography, years in comparison.items()},
        verified_cohorts=verified,
    )

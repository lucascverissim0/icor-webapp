#!/usr/bin/env python3
"""Compare two snapshots' cohorts and say whether every change is explained.

Counts only. A cohort is matched by its stable id (generation, market, cohort
year, horizon). A change is explained when the newer cohort carries a reason
code that names a new kind of evidence (a register, recovered names, net
retention, an incomplete provisional year's forecast), or when its market gained
or lost a release. Every other change is listed: an unexplained change is a
regression until shown otherwise. Vehicles both snapshots plan are also checked
for refolding (a different real identity).

usage: compare_snapshot_cohorts.py BEFORE.sqlite3 AFTER.sqlite3 [--horizon 2028]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from icor.infrastructure.snapshot_identity import load_identity_index

#: Reason codes that by themselves explain a changed cohort.
EXPLAINING = frozenset(
    {
        "observed-stock-anchored-cohort",
        "observed-stock-lower-bound",
        "registrations-include-model-recovered-by-type-approval",
        "eurostat-net-retention-calibrated",
        "observed-stock-labels-do-not-teach-vocabulary",
    }
)


def _cohorts(connection: sqlite3.Connection, horizon: int) -> dict[str, tuple]:
    return {
        cohort_id: (vehicle_id, geography, year, Decimal(p50), tuple(json.loads(codes)))
        for cohort_id, vehicle_id, geography, year, p50, codes in connection.execute(
            "SELECT cohort_id, canonical_vehicle_id, geography, registration_cohort_year, "
            "active_fleet_p50, reason_codes FROM cohort_estimate WHERE as_of_year = ?",
            (horizon,),
        )
    }


def _releases(connection: sqlite3.Connection) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for geography, release_id in connection.execute(
        "SELECT DISTINCT geography, release_id FROM observation"
    ):
        result[geography].add(release_id)
    return result


def _open(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--horizon", type=int, default=2028)
    args = parser.parse_args()
    before_db, after_db = _open(args.before), _open(args.after)
    try:
        before, after = _cohorts(before_db, args.horizon), _cohorts(after_db, args.horizon)
        releases_before, releases_after = _releases(before_db), _releases(after_db)
        index_before, index_after = load_identity_index(before_db), load_identity_index(after_db)
        shared_vehicles = {
            row[0]
            for row in before_db.execute(
                "SELECT DISTINCT canonical_vehicle_id FROM cohort_estimate"
            )
        } & {
            row[0]
            for row in after_db.execute("SELECT DISTINCT canonical_vehicle_id FROM cohort_estimate")
        }
    finally:
        before_db.close()
        after_db.close()

    changed_sources = {
        geography
        for geography in releases_before.keys() | releases_after.keys()
        if releases_before.get(geography, set()) != releases_after.get(geography, set())
    }
    markets: dict[str, dict] = defaultdict(
        lambda: {
            "before": Decimal(0),
            "after": Decimal(0),
            "added": 0,
            "removed": 0,
            "changed": 0,
            "unexplained": 0,
            "unexplained_delta": Decimal(0),
        }
    )
    unexplained: list[tuple] = []
    for cohort_id in before.keys() | after.keys():
        old, new = before.get(cohort_id), after.get(cohort_id)
        geography = (new or old)[1]
        market = markets[geography]
        market["before"] += old[3] if old else 0
        market["after"] += new[3] if new else 0
        if old is not None and new is not None and old[3] == new[3] and old[4] == new[4]:
            continue
        kind = "added" if old is None else "removed" if new is None else "changed"
        market[kind] += 1
        codes = set((new or old)[4])
        if geography in changed_sources or codes & EXPLAINING:
            continue
        delta = (new[3] if new else 0) - (old[3] if old else 0)
        market["unexplained"] += 1
        market["unexplained_delta"] += delta
        unexplained.append(
            (abs(delta), geography, (new or old)[0], (new or old)[2], kind, (new or old)[4])
        )

    refolded = sorted(
        vehicle_id
        for vehicle_id in shared_vehicles
        if index_before.identity_for(vehicle_id) != index_after.identity_for(vehicle_id)
    )
    result = {
        "horizon": args.horizon,
        "markets_with_changed_sources": sorted(changed_sources),
        "markets": {
            geography: {
                key: (round(float(value)) if isinstance(value, Decimal) else value)
                for key, value in values.items()
            }
            for geography, values in sorted(markets.items())
        },
        "unexplained_cohorts": len(unexplained),
        "unexplained_by_evidence": {
            first: sum(1 for item in unexplained if item[5][0] == first)
            for first in sorted({item[5][0] for item in unexplained})
        },
        "largest_unexplained": [
            {
                "delta": round(float(item[0]), 1),
                "geography": item[1],
                "vehicle": item[2],
                "cohort": item[3],
                "kind": item[4],
                "codes": list(item[5]),
            }
            for item in sorted(unexplained, reverse=True)[:25]
        ],
        "refolded_vehicles": len(refolded),
        "refolded_sample": refolded[:20],
    }
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

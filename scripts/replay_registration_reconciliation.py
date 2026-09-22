"""Replay registration reconciliation against a built snapshot, without rebuilding.

A snapshot rebuild costs about four and a quarter hours. This script scores the
reconciliation rule over the evidence an existing snapshot already contains, so a
change to that rule can be judged in minutes instead. It reads nothing but the
snapshot database, opens it read-only, and writes nothing.

It reports each annual key twice: once under the historical rule, which summed one
winner per dependency group, and once through whatever `RegistrationReconciler`
currently implements. Run it before changing the reconciler and confirm it
reproduces figures already known to be true; a script that cannot reproduce a known
number cannot be trusted to judge a new one.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from icor.forecasting.reconciliation import (
    CoverageCandidate,
    RegistrationCoverageSelector,
    RegistrationInput,
    RegistrationReconciler,
)
from icor.forecasting.survival import load_promoted_survival_model
from icor.infrastructure.snapshot_store import SnapshotStore

# Published national new-car registration totals, used only as a plausibility gate.
# These are EXTERNAL REFERENCE FIGURES, not snapshot evidence, and never enter a
# forecast. GB rows are SMMT annual new car registrations; the DE row is KBA.
_REFERENCE_ACTUALS: dict[tuple[str, int], int] = {
    ("GB", 2010): 2_030_846, ("GB", 2011): 1_941_253, ("GB", 2012): 2_044_609,
    ("GB", 2013): 2_264_737, ("GB", 2014): 2_476_435, ("GB", 2015): 2_633_503,
    ("GB", 2016): 2_692_786, ("GB", 2017): 2_540_617, ("GB", 2018): 2_367_147,
    ("GB", 2019): 2_311_140, ("GB", 2020): 1_631_064, ("GB", 2021): 1_647_181,
    ("GB", 2022): 1_614_063, ("GB", 2023): 1_903_054, ("GB", 2024): 1_952_778,
    ("DE", 2024): 2_817_331,
}

# Total cars licensed, all cohorts including those older than the snapshot's floor.
# A modelled fleet built from 2001+ cohorts alone must stay below this.
_PARC_CEILING: dict[str, int] = {"GB": 34_000_000, "DE": 49_000_000}

_MAX_ACTUAL_RATIO = Decimal("1.2")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay registration reconciliation over a built snapshot."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate", type=Path)
    source.add_argument("--root", type=Path)
    parser.add_argument("--geography", action="append", dest="geographies")
    parser.add_argument("--horizon", type=int, default=2028)
    parser.add_argument("--check-projection", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _database_path(args: argparse.Namespace) -> tuple[str, Path]:
    if args.candidate is not None:
        return "candidate", (args.candidate / "evidence.sqlite3").resolve()
    manifest, repository = SnapshotStore(args.root).open_active_snapshot()
    return manifest.snapshot_id, Path(repository.path).resolve()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _release_priority(source_id: str) -> int:
    """The historical priority table, reproduced so this script survives its move."""

    if source_id.startswith(("kba-", "uk-dft-")):
        return 30
    if source_id == "eea-co2-monitoring":
        return 20
    return 10


def _precedence(item: RegistrationInput) -> tuple[int, str]:
    return (-item.priority, item.input_id)


def _historical_rule(inputs: tuple[RegistrationInput, ...]) -> Decimal:
    """One winner per dependency group, then summed across groups.

    Reimplemented inline rather than imported, so this script keeps reporting the
    old behaviour after the reconciler stops implementing it.
    """

    grouped: dict[str, list[RegistrationInput]] = defaultdict(list)
    for item in inputs:
        grouped[item.dependency_group].append(item)
    total = Decimal(0)
    for dependency_group in sorted(grouped):
        total += min(grouped[dependency_group], key=_precedence).value
    return total


def _annual_inputs(
    connection: sqlite3.Connection, geographies: tuple[str, ...]
) -> dict[tuple[str, str, str, int], dict[str, Decimal]]:
    """Rebuild the planner's per-release annual buckets.

    Mirrors the planner's bucketing exactly: the cohort year falls back to the
    period's year only when the period does not straddle a year boundary, and
    observations that do straddle one are dropped.
    """

    placeholders = ",".join("?" for _ in geographies)
    rows = connection.execute(
        f"""
        SELECT a.selected_generation_id AS generation_id,
               o.canonical_vehicle_id   AS vehicle_id,
               o.geography              AS geography,
               COALESCE(o.registration_cohort_year,
                        CAST(SUBSTR(o.period_end, 1, 4) AS INTEGER)) AS year,
               o.release_id             AS release_id,
               o.value                  AS value
        FROM observation o
        JOIN generation_assignment a ON a.observation_id = o.observation_id
        WHERE o.measure = 'new_registrations'
          AND o.canonical_vehicle_id IS NOT NULL
          AND o.geography IN ({placeholders})
          AND (o.registration_cohort_year IS NOT NULL
               OR SUBSTR(o.period_start, 1, 4) = SUBSTR(o.period_end, 1, 4))
        """,
        geographies,
    )
    buckets: dict[tuple[str, str, str, int], dict[str, Decimal]] = defaultdict(
        lambda: defaultdict(Decimal)
    )
    for row in rows:
        if row["year"] is None:
            continue
        key = (row["generation_id"], row["vehicle_id"], row["geography"], int(row["year"]))
        buckets[key][row["release_id"]] += Decimal(str(row["value"]))
    return buckets


def _releases(connection: sqlite3.Connection) -> dict[str, tuple[str, str, str]]:
    return {
        row["release_id"]: (
            row["dependency_group"],
            row["source_id"],
            row["publication_status"],
        )
        for row in connection.execute(
            """SELECT release_id, dependency_group, source_id, publication_status
            FROM source_release"""
        )
    }


def _coverage(
    buckets: dict[tuple[str, str, str, int], dict[str, Decimal]],
    releases: dict[str, tuple[str, str, str]],
) -> dict[tuple[str, int], str]:
    """The publisher that decomposes each country-year, as the planner picks it."""

    candidates: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    for (_generation, _vehicle, geography, year), by_release in buckets.items():
        for release_id in by_release:
            _group, source_id, status = releases[release_id]
            candidates[geography, year].add((source_id, status))
    selector = RegistrationCoverageSelector()
    return {
        scope: selector.select(
            tuple(
                CoverageCandidate(source_id=source, publication_status=status)
                for source, status in sorted(present)
            )
        ).source_id
        for scope, present in candidates.items()
    }


def _percentiles(values: list[Decimal]) -> dict[str, str]:
    if not values:
        return {"p50": "0", "p90": "0", "p99": "0", "max": "0"}
    ordered = sorted(values)

    def _at(fraction: float) -> Decimal:
        index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1) + 0.5))
        return ordered[index]

    return {
        "p50": str(_at(0.50)), "p90": str(_at(0.90)),
        "p99": str(_at(0.99)), "max": str(ordered[-1]),
    }


def _ratio(value: Decimal, actual: int) -> Decimal:
    return (value / Decimal(actual)).quantize(Decimal("0.0001"))


def _representatives(
    inputs: tuple[RegistrationInput, ...],
) -> dict[str, RegistrationInput]:
    chosen: dict[str, RegistrationInput] = {}
    for item in inputs:
        current = chosen.get(item.dependency_group)
        if current is None or _precedence(item) < _precedence(current):
            chosen[item.dependency_group] = item
    return chosen


def replay(
    connection: sqlite3.Connection, geographies: tuple[str, ...], horizon: int
) -> dict[str, object]:
    releases = _releases(connection)
    buckets = _annual_inputs(connection, geographies)
    coverage = _coverage(buckets, releases)
    reconciler = RegistrationReconciler()
    survival = load_promoted_survival_model()

    totals: dict[tuple[str, int], dict[str, object]] = defaultdict(
        lambda: {
            "historical": Decimal(0), "corrected": Decimal(0), "keys": 0,
            "corrected_keys": 0, "superseded_keys": 0, "corroborated": 0,
            "by_source": defaultdict(Decimal),
        }
    )
    disagreements: dict[str, list[Decimal]] = defaultdict(list)
    fleet: dict[str, dict[str, Decimal]] = defaultdict(
        lambda: {"historical": Decimal(0), "corrected": Decimal(0)}
    )

    for (_generation, _vehicle, geography, year), by_release in buckets.items():
        inputs = tuple(
            RegistrationInput(
                release_id,
                releases[release_id][0],
                value,
                _release_priority(releases[release_id][1]),
            )
            for release_id, value in sorted(by_release.items())
        )
        historical = _historical_rule(inputs)
        age = horizon - year

        # The baseline is accumulated for every key, including keys the coverage
        # rule drops entirely. Skipping those here too would quietly shrink the
        # "before" figure and flatter the fix.
        bucket = totals[geography, year]
        bucket["historical"] += historical
        bucket["keys"] += 1
        for item in inputs:
            bucket["by_source"][releases[item.input_id][1]] += item.value
        if age >= 0:
            fleet[geography]["historical"] += survival.interval(historical, age_years=age).p50

        representatives = _representatives(inputs)
        if len(representatives) > 1:
            bucket["corroborated"] += 1
            values = [item.value for item in representatives.values()]
            largest = max(values)
            if largest > 0:
                gap = (largest - min(values)) / largest
                disagreements[geography].append(gap.quantize(Decimal("0.000001")))

        covered = tuple(
            item
            for item in inputs
            if releases[item.input_id][1] == coverage[geography, year]
        )
        if not covered:
            bucket["superseded_keys"] += 1
            continue
        result = reconciler.reconcile(covered)
        bucket["corrected"] += result.value
        bucket["corrected_keys"] += 1
        if age >= 0:
            fleet[geography]["corrected"] += survival.interval(result.value, age_years=age).p50

    report: dict[str, object] = {"horizon": horizon, "geographies": {}}
    for geography in geographies:
        years = sorted(year for (geo, year) in totals if geo == geography)
        rows = []
        for year in years:
            bucket = totals[geography, year]
            actual = _REFERENCE_ACTUALS.get((geography, year))
            row: dict[str, object] = {
                "year": year,
                "keys": bucket["keys"],
                "corrected_keys": bucket["corrected_keys"],
                "superseded_keys": bucket["superseded_keys"],
                "corroborated_keys": bucket["corroborated"],
                "historical_total": str(bucket["historical"]),
                "corrected_total": str(bucket["corrected"]),
                "by_source": {k: str(v) for k, v in sorted(bucket["by_source"].items())},
            }
            if actual is not None:
                row["reference_actual"] = actual
                row["historical_over_actual"] = str(_ratio(bucket["historical"], actual))
                row["corrected_over_actual"] = str(_ratio(bucket["corrected"], actual))
            rows.append(row)
        report["geographies"][geography] = {
            "years": rows,
            "disagreement_percentiles": _percentiles(disagreements[geography]),
            "modelled_fleet": {
                "historical": str(fleet[geography]["historical"].quantize(Decimal("1"))),
                "corrected": str(fleet[geography]["corrected"].quantize(Decimal("1"))),
                "parc_ceiling": _PARC_CEILING.get(geography),
            },
        }
    return report


def projection_totals(
    connection: sqlite3.Connection, geographies: tuple[str, ...]
) -> dict[str, object]:
    """Totals as the Registrations page reads them, straight off the projection."""

    placeholders = ",".join("?" for _ in geographies)
    rows = connection.execute(
        f"""
        SELECT geography, year, registrations
        FROM registration_family_aggregate
        WHERE geography IN ({placeholders}) AND publication_status = 'final'
        """,
        geographies,
    )
    totals: dict[tuple[str, int], Decimal] = defaultdict(Decimal)
    for row in rows:
        totals[row["geography"], int(row["year"])] += Decimal(str(row["registrations"]))
    out: dict[str, object] = {}
    for geography in geographies:
        years = []
        for (geo, year), value in sorted(totals.items()):
            if geo != geography:
                continue
            entry: dict[str, object] = {"year": year, "projection_total": str(value)}
            actual = _REFERENCE_ACTUALS.get((geography, year))
            if actual is not None:
                entry["reference_actual"] = actual
                entry["projection_over_actual"] = str(_ratio(value, actual))
            years.append(entry)
        out[geography] = years
    return out


def _gate(report: dict[str, object]) -> list[str]:
    failures: list[str] = []
    for geography, payload in report["geographies"].items():
        for row in payload["years"]:
            ratio = row.get("corrected_over_actual")
            if ratio is not None and Decimal(ratio) > _MAX_ACTUAL_RATIO:
                failures.append(
                    f"{geography} {row['year']}: corrected/actual {ratio} "
                    f"exceeds {_MAX_ACTUAL_RATIO}"
                )
        ceiling = payload["modelled_fleet"]["parc_ceiling"]
        corrected = Decimal(payload["modelled_fleet"]["corrected"])
        if ceiling is not None and corrected > Decimal(ceiling):
            failures.append(
                f"{geography}: modelled fleet {corrected} exceeds parc ceiling {ceiling}"
            )
    return failures


def _render(report: dict[str, object], projection: dict[str, object] | None) -> None:
    print(f"horizon {report['horizon']}")
    for geography, payload in report["geographies"].items():
        print(f"\n=== {geography} ===")
        print(
            f"{'year':>6} {'keys':>7} {'kept':>7} {'dropped':>8} {'corrob':>7} "
            f"{'historical':>14} {'corrected':>14} {'hist/act':>9} {'corr/act':>9}"
        )
        for row in payload["years"]:
            print(
                f"{row['year']:>6} {row['keys']:>7} {row['corrected_keys']:>7} "
                f"{row['superseded_keys']:>8} {row['corroborated_keys']:>7} "
                f"{row['historical_total']:>14} {row['corrected_total']:>14} "
                f"{row.get('historical_over_actual', '-'):>9} "
                f"{row.get('corrected_over_actual', '-'):>9}"
            )
        percentiles = payload["disagreement_percentiles"]
        print(
            "  cross-publisher disagreement  "
            f"p50={percentiles['p50']} p90={percentiles['p90']} "
            f"p99={percentiles['p99']} max={percentiles['max']}"
        )
        modelled = payload["modelled_fleet"]
        print(
            f"  modelled fleet at horizon     historical={modelled['historical']} "
            f"corrected={modelled['corrected']} ceiling={modelled['parc_ceiling']}"
        )
        if projection is not None:
            print("  registration projection (page totals):")
            for entry in projection.get(geography, []):
                print(
                    f"    {entry['year']:>6} {entry['projection_total']:>14} "
                    f"{entry.get('projection_over_actual', '-'):>9}"
                )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    geographies = tuple(args.geographies or ("GB",))
    snapshot_id, path = _database_path(args)
    connection = _connect(path)
    try:
        report = replay(connection, geographies, args.horizon)
        projection = (
            projection_totals(connection, geographies) if args.check_projection else None
        )
    finally:
        connection.close()

    report["snapshot_id"] = snapshot_id
    if projection is not None:
        report["projection"] = projection
    failures = _gate(report)
    report["gate_failures"] = failures

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"snapshot {snapshot_id}")
        _render(report, projection)
        if failures:
            print("\nGATE FAILURES")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print("\ngate: pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

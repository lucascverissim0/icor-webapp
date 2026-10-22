#!/usr/bin/env python3
"""Hide each national register and score the estimate a register-less country would get.

Mission 4, Phase 1 (plan `~/.claude/plans/ok-for-the-countries-melodic-pine.md`). The
countries with complete registers (NL, FI, LU, LT, EE, CZ, DK, SI) are the truth: cars on
the road per vehicle and first-registration year at the register's date. Each is estimated
from what a country without a register has: EEA new registrations from 2010, aged with
the snapshot's own survival model, which is the method every register-less country gets
today. GB is not a truth: the survival curve was fitted on it.

Scores (`icor.forecasting.hidden_register`):

- L1, cars per cohort, gate: every cohort within 8% at ages 0-15 and 15% at 16-25;
- L2, make x cohort WAPE for the 20 largest makes, 2010+ cohorts;
- L3, cars per Eurocode at the horizon (demand per code is proportional): WAPE on codes
  with at least 1,000 true units, bias, rank correlation of the top 200 codes and the
  share of large codes inside the P10-P90 interval. Measured on the cohorts the profile
  can see (2010+), on every register cohort (older cars count as missed), and with each
  cohort's level corrected to the truth (an oracle that leaves only the mix error).

Read-only. Writes a JSON report under `.local/private/backtests/`.

usage: backtest_estimated_layer.py [--candidate SNAPSHOT_ID] [--horizon 2028]
       [--geographies NL FI ...] [--first-cohort 2006]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

from icor.eurocodes.runtime import EurocodeMap, identity_key
from icor.evidence.vehicle_identity import fold_make
from icor.forecasting.hidden_register import (
    Cell,
    Interval,
    code_scores,
    cohort_errors,
    l1_gate,
    make_scores,
    per_code,
    project,
    registrations_times_survival,
    rescale_to_cohorts,
)
from icor.forecasting.survival import load_survival_model_for
from icor.infrastructure.snapshot_identity import load_identity_index
from icor.infrastructure.snapshot_store import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]
TRUTHS = ("NL", "FI", "LU", "LT", "EE", "CZ", "DK", "SI")
#: The first year EEA publishes registrations by model: a register-less country's profile.
PROFILE_FIRST_YEAR = 2010
L3_GATES = {"wape_large": 0.30, "bias": 0.05, "spearman_top": 0.85, "coverage": (0.72, 0.88)}


def _ledger(root: Path, candidate: str | None) -> tuple[str, Path]:
    if candidate:
        return candidate, root / "candidates" / candidate / "evidence.sqlite3"
    manifest, ledger = SnapshotStore(root, trust_verified_marker=True).open_active_snapshot()
    return manifest.snapshot_id, ledger.path


def _observations(
    connection: sqlite3.Connection, geographies: tuple[str, ...], measure: str
) -> list[tuple[str, str, str, str, int, str, float]]:
    marks = ",".join("?" * len(geographies))
    return connection.execute(
        f"""SELECT o.geography, o.canonical_vehicle_id, v.make, v.model,
               o.registration_cohort_year, MAX(o.period_end), SUM(o.value)
        FROM observation o JOIN canonical_vehicle v ON v.vehicle_id = o.canonical_vehicle_id
        WHERE o.geography IN ({marks}) AND o.measure = ? AND o.mapping_status != 'rejected'
          AND o.canonical_vehicle_id IS NOT NULL AND o.registration_cohort_year IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5""",
        (*geographies, measure),
    ).fetchall()


def _shares(model: Any, geography: str) -> Callable[[int], Interval]:
    scale = Decimal(1_000_000)
    cache: dict[int, Interval] = {}

    def shares(age: int) -> Interval:
        if age not in cache:
            band = model.interval(scale, age_years=max(age, 0), geography=geography)
            cache[age] = (float(band.p10 / scale), float(band.p50 / scale), float(band.p90 / scale))
        return cache[age]

    return shares


def _l3(
    eurocodes: EurocodeMap,
    geography: str,
    truth: dict[Cell, Interval],
    predicted: dict[Cell, Interval],
) -> dict[str, Any]:
    def allocated(cells: dict[Cell, Interval], quantile: int) -> dict[str, float]:
        rows = []
        for cell, values in cells.items():
            allocation = eurocodes.allocate(
                cell.identity, cell.cohort, geography, raw_model=cell.model
            )
            rows.append((allocation.codes, allocation.status, values[quantile]))
        return per_code(rows)

    return code_scores(
        allocated(truth, 1), allocated(predicted, 0), allocated(predicted, 1),
        allocated(predicted, 2), min_units=1000, top=200,
    )


def _l3_passes(scores: dict[str, Any]) -> bool:
    low, high = L3_GATES["coverage"]
    return (
        scores["wape_large"] <= L3_GATES["wape_large"]
        and abs(scores["bias"]) <= L3_GATES["bias"]
        and scores["spearman_top"] >= L3_GATES["spearman_top"]
        and low <= scores["coverage_large"] <= high
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / ".local" / "evidence")
    parser.add_argument("--candidate")
    parser.add_argument("--eurocode-dir", type=Path, default=ROOT / ".local/private/eurocode")
    parser.add_argument("--out-dir", type=Path, default=ROOT / ".local/private/backtests")
    parser.add_argument("--horizon", type=int, default=2028)
    parser.add_argument("--first-cohort", type=int, default=2006)
    parser.add_argument("--geographies", nargs="*", default=list(TRUTHS))
    args = parser.parse_args()
    geographies = tuple(args.geographies)

    snapshot_id, path = _ledger(args.root, args.candidate)
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    methods = connection.execute("SELECT DISTINCT survival_method FROM cohort_estimate").fetchall()
    if len(methods) != 1:
        raise SystemExit(f"expected one survival method, found {methods}")
    survival = load_survival_model_for(methods[0][0])
    stock = _observations(connection, geographies, "active_fleet")
    registrations = _observations(connection, geographies, "new_registrations")

    # Vehicles the ranking does not hold fold with the same vocabulary but never teach it.
    extras = {(row[1], row[2], row[3]) for row in (*stock, *registrations)}
    baseline = load_identity_index(connection)
    index = load_identity_index(connection, extra_vehicles=extras)
    ranked = [row[0] for row in connection.execute("SELECT vehicle_id FROM canonical_vehicle")]
    refold = sum(
        1
        for vid in ranked
        if (before := baseline.identity_for(vid)) is not None and before != index.identity_for(vid)
    )
    connection.close()
    if refold:
        raise SystemExit(f"refold {refold}: extra vehicles changed how ranked vehicles fold")
    eurocodes = EurocodeMap.from_directory(args.eurocode_dir)

    def cell(vehicle_id: str, make: str, model: str, cohort: int) -> Cell:
        return Cell(
            identity_key(index, vehicle_id, make, model),
            fold_make(make) or make.casefold(),
            model,
            int(cohort),
        )

    truth: dict[str, dict[Cell, float]] = defaultdict(lambda: defaultdict(float))
    dates: dict[str, str] = {}
    for geography, vehicle_id, make, model, cohort, end, units in stock:
        if int(cohort) >= args.first_cohort:
            truth[geography][cell(vehicle_id, make, model, cohort)] += units
        dates[geography] = max(dates.get(geography, ""), end)
    hidden: dict[str, dict[Cell, float]] = defaultdict(lambda: defaultdict(float))
    for geography, vehicle_id, make, model, cohort, _end, units in registrations:
        if int(cohort) >= PROFILE_FIRST_YEAR:
            hidden[geography][cell(vehicle_id, make, model, cohort)] += units

    report: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "survival_method": methods[0][0],
        "estimator": "registrations-x-survival (current fallback)",
        "profile": f"EEA new registrations from {PROFILE_FIRST_YEAR}",
        "horizon": args.horizon,
        "first_cohort": args.first_cohort,
        "refold": refold,
        "countries": {},
    }
    print(f"{snapshot_id}; horizon {args.horizon}; estimator: registrations x survival")
    print("geo  date        2010+ pred/truth  pre-2010 missed  L1   L2 WAPE (mix)   "
          "L3 2010+ WAPE bias rho cover | all-cohorts WAPE bias | mix-only WAPE")
    for geography in geographies:
        year = int(dates[geography][:4])
        shares = _shares(survival, geography)
        true_cells = {c: u for c, u in truth[geography].items() if c.cohort < year}
        predicted = registrations_times_survival(hidden[geography], shares, year=year)
        predicted = {c: v for c, v in predicted.items() if c.cohort < year}

        true_cohort: dict[int, float] = defaultdict(float)
        for c, units in true_cells.items():
            true_cohort[c.cohort] += units
        predicted_cohort: dict[int, float] = defaultdict(float)
        for c, values in predicted.items():
            predicted_cohort[c.cohort] += values[1]
        errors = cohort_errors(true_cohort, predicted_cohort, year=year)
        gate = l1_gate(errors)
        seen = [e for e in errors if e.cohort >= PROFILE_FIRST_YEAR]
        seen_truth = sum(e.truth for e in seen)
        seen_ratio = sum(e.predicted for e in seen) / seen_truth
        missed = sum(e.truth for e in errors if e.cohort < PROFILE_FIRST_YEAR)

        true_make: dict[tuple[str, int], float] = defaultdict(float)
        predicted_make: dict[tuple[str, int], float] = defaultdict(float)
        for c, units in true_cells.items():
            if c.cohort >= PROFILE_FIRST_YEAR:
                true_make[(c.make, c.cohort)] += units
        for c, values in predicted.items():
            predicted_make[(c.make, c.cohort)] += values[1]
        makes = make_scores(true_make, predicted_make, top=20)

        def at_horizon(
            cells: dict[Cell, Interval],
            shares: Callable[[int], Interval] = shares,
            year: int = year,
        ) -> dict[Cell, Interval]:
            return project(cells, shares, year=year, horizon=args.horizon)

        truth_all = at_horizon({c: (u, u, u) for c, u in true_cells.items()})
        truth_seen = {c: v for c, v in truth_all.items() if c.cohort >= PROFILE_FIRST_YEAR}
        predicted_h = at_horizon(predicted)
        oracle = at_horizon(
            rescale_to_cohorts(
                predicted,
                {k: v for k, v in true_cohort.items() if k >= PROFILE_FIRST_YEAR},
            )
        )
        l3_seen = _l3(eurocodes, geography, truth_seen, predicted_h)
        l3_all = _l3(eurocodes, geography, truth_all, predicted_h)
        l3_mix = _l3(eurocodes, geography, truth_seen, oracle)
        for scores in (l3_seen, l3_all, l3_mix):
            scores["passes"] = _l3_passes(scores)

        report["countries"][geography] = {
            "register_date": dates[geography],
            "seen_cohorts_predicted_over_truth": seen_ratio,
            "missed_before_profile": missed,
            "cohorts": [
                {"cohort": e.cohort, "age": e.age, "truth": e.truth,
                 "predicted": e.predicted, "error": e.error}
                for e in errors
            ],
            "l1": gate,
            "l2": makes,
            "l3_seen_cohorts": l3_seen,
            "l3_all_cohorts": l3_all,
            "l3_mix_only_oracle": l3_mix,
        }
        print(
            f"{geography}   {dates[geography]}  {seen_ratio:6.3f}            "
            f"{missed:10,.0f}     {'pass' if gate['passes'] else 'FAIL'} "
            f"{makes['wape']:.3f} ({makes['mix_wape']:.3f})   "
            f"{l3_seen['wape_large']:.3f} {l3_seen['bias']:+.3f} "
            f"{l3_seen['spearman_top']:.3f} {l3_seen['coverage_large']:.2f} | "
            f"{l3_all['wape_large']:.3f} {l3_all['bias']:+.3f} | {l3_mix['wape_large']:.3f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir.resolve() / f"hidden-register-{snapshot_id}-h{args.horizon}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

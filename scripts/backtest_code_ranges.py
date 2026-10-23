#!/usr/bin/env python3
"""Measure how wrong each Eurocode's estimate is, and whether a measured 80% range holds.

Mission 4 Part B step 2, harness v3. The eight complete national registers (NL, FI,
LU, LT, EE, CZ, DK, SI) are hidden one at a time and estimated the way a country
without a register is: its EEA new registrations from 2010 times the app's
survival model for that market, at the register's own date (ageing truth and
estimate to a later horizon by the same curve adds nothing). The market's own
Eurostat inputs (net retention, age anchor) are used, as the app would use them
without the register: they are band totals every market publishes, and the
per-code truth stays hidden. Each code is counted as the ranking counts it, on
the units it certainly carries. The estimate is scored three ways:

- ``raw``: the app's estimate itself;
- ``anchored``: the bare curve's cohorts scaled to the country's official cars
  by age band at the register's own date (Eurostat ``road_eqs_carage``); what
  remains is the error in the mix of codes;
- ``oracle_level``: the country's true level, an upper bound for any anchor.

A range for a code is learned from the other countries' codes only (leave one
country out), in bins fixed before this was run (predicted size x share of the
code's life before 2010), each country weighing one. The gate: on codes
predicted at 1,000 cars or more, P10-P90 must hold the truth for 72-88% pooled
and at least 60% in every country. The gate decides on ``raw``, the estimate the
app shows; ``anchored`` passing says what would make a range hold, not that the
app's figures have one. Also measured, never gated:

- baseline (iii), the other registers' mix of codes scaled to the estimate's total;
- the same gate with the hidden profile degraded (whole vehicles lost,
  publisher mismatch per vehicle, first-word model names), ranges still learned
  undegraded;
- Spain, one-sided: DGT's unmasked stock is a lower bound per code, so a code
  whose lower bound exceeds its P90 is certainly outside its range;
- the stock kind: GB's 2019 stock aged to 2025 with the curve against GB's 2025
  stock, in-sample (the curve was fitted on GB);
- the forecast kind: each series forecast from 2023 with the production
  forecaster, per code and market, for 2024 (step 1) and 2025 (step 2).

Known leakage, reported beside the results: the identity vocabulary was built
with the registers' stock in the snapshot; the fixed constants were chosen after
seeing aggregate (not per-code) errors from harness v1; the registration series
are today's revised values, not what was published at the time.

Read-only. Writes per-code records under ``.local/private/backtests/`` and a
status document, pass or fail, under ``.local/private/eurocode/calibration/``.
Prints counts and rates only, never a code.

    uv run python scripts/backtest_code_ranges.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icor.domain.age_bands import band_layout  # noqa: E402
from icor.eurocodes.runtime import EurocodeMap, identity_key  # noqa: E402
from icor.evidence import (  # noqa: E402
    eurostat_age_conventions,
    eurostat_car_stock,
    eurostat_new_registrations,
)
from icor.evidence.vehicle_identity import fold_make  # noqa: E402
from icor.forecasting.hidden_register import (  # noqa: E402
    Cell,
    Interval,
    band_level_factors,
    coarse_names,
    donor_mix,
    drop_cells,
    per_code_certain,
    perturb_cells,
    pre_profile_share,
    project,
    registrations_times_survival,
)
from icor.forecasting.range_calibration import (  # noqa: E402
    MIN_PREDICTED,
    QUANTILES,
    CodeObservation,
    bin_of,
    coverage,
    gate,
    held_out_intervals,
    interval,
    ratio_table,
    relative_to_level,
    weighted_quantile,
)
from icor.forecasting.registration_forecast import RegistrationForecaster  # noqa: E402
from icor.forecasting.survival import (  # noqa: E402
    load_age_anchored_survival_model,
    load_survival_model_for,
)
from icor.infrastructure.snapshot_identity import load_identity_index  # noqa: E402
from scripts.backtest_estimated_layer import (  # noqa: E402
    PROFILE_FIRST_YEAR,
    TRUTHS,
    _ledger,
    _observations,
)
from scripts.backtest_market_totals import _forecast  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EUROSTAT = ROOT / ".local" / "external-data" / "eurostat"
DGT = ROOT / ".local" / "evidence" / "downloads" / "dgt-es-fleet.csv"
FORMAT = "icor-range-calibration"
VERSION = 1


def _survival_shares(model: Any, geography: str | None) -> Callable[[int], Interval]:
    """The model's surviving share by age in a market; None is the bare curve."""

    scale = Decimal(1_000_000)
    cache: dict[int, Interval] = {}

    def shares(age: int) -> Interval:
        if age not in cache:
            band = model.interval(scale, age_years=max(age, 0), geography=geography)
            cache[age] = (float(band.p10 / scale), float(band.p50 / scale), float(band.p90 / scale))
        return cache[age]

    return shares


def _table(table: Mapping[str, Any]) -> dict[str, Any]:
    """P10/P90 as ratios of truth to estimate (a log of zero would not be JSON)."""

    def ratios(value: tuple[float, float]) -> list[float]:
        return [math.exp(bound) for bound in value]

    return {
        "pooled": ratios(table["pooled"]),
        "bins": {f"{key[0]}:{key[1]}": ratios(value) for key, value in table["bins"].items()},
    }


def _support(shares: Callable[[int], Interval]) -> int:
    """The oldest age the curve is calibrated for."""

    age = 0
    while True:
        try:
            shares(age + 1)
        except ValueError:
            return age
        age += 1


def _official(geographies: set[str]) -> tuple[dict, dict, dict, dict]:
    stock_path = max(EUROSTAT.glob(f"{eurostat_car_stock.DATASET}-*.json"))
    registrations_path = max(EUROSTAT.glob(f"{eurostat_new_registrations.DATASET}-*.json"))
    document = json.loads(stock_path.read_text(encoding="utf-8"))
    registrations = eurostat_new_registrations.parse(
        json.loads(registrations_path.read_text(encoding="utf-8")), geographies
    )
    return (
        document,
        registrations,
        eurostat_car_stock.band_by_year(document, geographies, "Y_LT2"),
        eurostat_car_stock.band_by_year(document, geographies, "Y2-5"),
    )


def _stock_for(official: tuple[dict, dict, dict, dict], geography: str, year: int) -> Any:
    """The office's newest count at or before ``year``, with its age convention."""

    document, registrations, youngest, second = official
    for candidate in range(year, year - 4, -1):
        stock = eurostat_car_stock.parse(document, {geography}, year=candidate).get(geography)
        if stock is not None and stock.band_year is not None:
            break
    else:
        stock = eurostat_car_stock.parse(document, {geography}).get(geography)
    if stock is None or stock.band_year is None:
        return None, None
    convention = eurostat_age_conventions.for_stock(
        stock, youngest.get(geography, {}), registrations.get(geography, {}), second.get(geography)
    )
    return stock, convention.convention if convention else None


def _all_registrations(connection: sqlite3.Connection, geography: str) -> dict[int, float]:
    """Every new car the publisher counted per cohort, mapped to a vehicle or not."""

    rows = connection.execute(
        """SELECT registration_cohort_year, SUM(CAST(value AS REAL)) FROM observation
        WHERE geography = ? AND measure = 'new_registrations' AND mapping_status != 'rejected'
          AND registration_cohort_year >= ? GROUP BY 1""",
        (geography, PROFILE_FIRST_YEAR),
    )
    return {int(cohort): float(units) for cohort, units in rows}


def _anchor(
    official: tuple[dict, dict, dict, dict],
    geography: str,
    year: int,
    registrations: Mapping[int, float],
    shares: Callable[[int], Interval],
) -> dict[str, Any]:
    stock, convention = _stock_for(official, geography, year)
    if stock is None or convention in (None, eurostat_age_conventions.UNCLEAR):
        # An unclear convention cannot say which cohorts a band holds.
        return {"available": False, "factor": {}, "convention": convention}
    band_year = stock.band_year
    predicted = {
        cohort: units * shares(band_year - cohort)[1]
        for cohort, units in registrations.items()
        if cohort <= band_year
    }
    predicted.update({cohort: 0.0 for cohort in registrations if cohort > band_year})
    result = band_level_factors(
        stock.bands,
        band_layout(convention),
        predicted,
        band_year=band_year,
        profile_first_year=PROFILE_FIRST_YEAR,
    )
    return {
        "available": bool(result["factor"]),
        "band_year": band_year,
        "convention": convention,
        "bands": result["bands"],
        "factor": result["factor"],
    }


def _allocate(
    eurocodes: EurocodeMap, geography: str, cells: Mapping[Cell, float]
) -> tuple[dict[str, float], dict[str, float]]:
    rows = []
    for cell, units in cells.items():
        allocation = eurocodes.allocate(cell.identity, cell.cohort, geography, raw_model=cell.model)
        rows.append((allocation.codes, allocation.status, units))
    return per_code_certain(rows)


def _quantile(cells: Mapping[Cell, Interval], position: int) -> dict[Cell, float]:
    return {cell: values[position] for cell, values in cells.items()}


def _records(
    geography: str,
    truth: Mapping[str, float],
    predicted: Mapping[str, float],
    life: Callable[[str], float],
) -> list[CodeObservation]:
    return [
        CodeObservation(geography, code, truth.get(code, 0.0), units, life(code))
        for code, units in predicted.items()
        if units > 0
    ]


def _wape(truth: Mapping[str, float], predicted: Mapping[str, float]) -> float | None:
    large = [code for code, units in truth.items() if units >= MIN_PREDICTED]
    denominator = sum(truth[code] for code in large)
    if not denominator:
        return None
    return sum(abs(predicted.get(code, 0.0) - truth[code]) for code in large) / denominator


def _gate_block(observations: list[CodeObservation]) -> dict[str, Any]:
    intervals = held_out_intervals(observations)
    scores = coverage(observations, intervals)
    return {"coverage": scores, "gate": gate(scores)}


def _log_ratio_quantiles(observations: Iterable[CodeObservation]) -> dict[str, Any]:
    scored = [item for item in observations if item.predicted >= MIN_PREDICTED]
    if not scored:
        return {"codes": 0}
    values = [(item.log_ratio, 1.0) for item in scored]
    return {
        "codes": len(scored),
        "ratio_p10": math.exp(weighted_quantile(values, 0.10)),
        "ratio_p50": math.exp(weighted_quantile(values, 0.50)),
        "ratio_p90": math.exp(weighted_quantile(values, 0.90)),
    }


def _degraded_coverage(
    clean: list[CodeObservation], degraded: Mapping[str, list[CodeObservation]]
) -> dict[str, Any]:
    """Each country's degraded codes against ranges learned from the others' clean ones."""

    scored = [item for item in clean if item.predicted >= MIN_PREDICTED]
    intervals = {}
    for country, items in degraded.items():
        table = ratio_table([item for item in scored if item.country != country])
        for item in items:
            if item.predicted >= MIN_PREDICTED:
                intervals[(country, item.code)] = interval(table, item)
    flat = [item for items in degraded.values() for item in items]
    scores = coverage(flat, intervals)
    return {"coverage": scores, "gate": gate(scores)}


def _uk_stock(
    connection: sqlite3.Connection, index: Any, eurocodes: EurocodeMap, shares: Any
) -> dict[str, Any]:
    rows = connection.execute(
        """SELECT substr(o.period_end, 1, 4), o.canonical_vehicle_id, v.make, v.model,
               o.registration_cohort_year, SUM(CAST(o.value AS REAL))
        FROM observation o JOIN canonical_vehicle v ON v.vehicle_id = o.canonical_vehicle_id
        WHERE o.geography = 'UK' AND o.measure = 'active_fleet'
          AND o.period_end IN ('2019-12-31', '2025-12-31')
          AND o.registration_cohort_year IS NOT NULL AND o.registration_cohort_year <= 2019
          AND o.mapping_status != 'rejected'
        GROUP BY 1, 2, 3, 4, 5"""
    ).fetchall()
    oldest = 2025 - _support(shares)
    vintages: dict[str, dict[Cell, float]] = defaultdict(lambda: defaultdict(float))
    for vintage, vehicle_id, make, model, cohort, units in rows:
        if int(cohort) < oldest:
            continue
        cell = Cell(identity_key(index, vehicle_id, make, model), make, model, int(cohort))
        vintages[vintage][cell] += units
    aged = project(
        {cell: (units, units, units) for cell, units in vintages["2019"].items()},
        shares,
        year=2019,
        horizon=2025,
    )
    banded = {}
    for cell, units in vintages["2019"].items():
        now, later = shares(2019 - cell.cohort), shares(2025 - cell.cohort)
        banded[cell] = tuple(units * after / now[1] if now[1] else 0.0 for after in later)
    truth, _ = _allocate(eurocodes, "GB", vintages["2025"])
    low, _ = _allocate(eurocodes, "GB", _quantile(banded, 0))
    mid, _ = _allocate(eurocodes, "GB", _quantile(aged, 1))
    high, _ = _allocate(eurocodes, "GB", _quantile(banded, 2))
    scored = [code for code, units in mid.items() if units >= MIN_PREDICTED]
    inside = [
        code for code in scored if low.get(code, 0) <= truth.get(code, 0) <= high.get(code, 0)
    ]
    records = _records("GB", truth, mid, lambda _code: 0.0)
    return {
        "vintages": [2019, 2025],
        "cohorts": [oldest, 2019],
        "leakage": "the survival curve was fitted on GB stock; in-sample",
        "codes_scored": len(scored),
        "wape_large": _wape(truth, mid),
        "bias": sum(mid.values()) / sum(truth.values()) - 1 if truth else None,
        "survival_band_coverage": len(inside) / len(scored) if scored else None,
        "log_ratio": _log_ratio_quantiles(records),
    }


def _forecast_kind(
    connection: sqlite3.Connection, index: Any, eurocodes: EurocodeMap
) -> dict[str, Any]:
    series: dict[tuple[str, str, str], dict[int, Decimal]] = defaultdict(dict)
    for generation, vehicle, geography, year, units in connection.execute(
        """SELECT generation_id, canonical_vehicle_id, geography,
               registration_cohort_year, registrations
        FROM cohort_estimate WHERE as_of_year = 2031
        AND reason_codes LIKE '%"reconciled-registration-cohort"%'"""
    ):
        series[generation, vehicle, geography][int(year)] = Decimal(units)
    labels = dict(
        (vehicle_id, (make, model))
        for vehicle_id, make, model in connection.execute(
            "SELECT vehicle_id, make, model FROM canonical_vehicle"
        )
    )
    forecaster = RegistrationForecaster()
    truth: dict[tuple[str, int], dict[Cell, float]] = defaultdict(lambda: defaultdict(float))
    predicted: dict[tuple[str, int], dict[Cell, float]] = defaultdict(lambda: defaultdict(float))
    for (_generation, vehicle, geography), history in series.items():
        make, model = labels[vehicle]
        identity = identity_key(index, vehicle, make, model)
        values = _forecast(forecaster, history, 2023, 2025)
        for year in (2024, 2025):
            cell = Cell(identity, make, model, year)
            if year in history:
                truth[geography, year][cell] += float(history[year])
            if values is not None:
                predicted[geography, year][cell] += float(values.get(year, 0))
    result: dict[str, Any] = {"origin": 2023, "steps": {}}
    for step, year in ((1, 2024), (2, 2025)):
        observations = []
        for geography in sorted({key[0] for key in truth}):
            true_codes, _ = _allocate(eurocodes, geography, truth[geography, year])
            predicted_codes, _ = _allocate(eurocodes, geography, predicted[geography, year])
            observations += _records(geography, true_codes, predicted_codes, lambda _c: 0.0)
        result["steps"][str(step)] = {
            "year": year,
            **_gate_block(observations),
            "log_ratio": _log_ratio_quantiles(observations),
        }
    return result


def _spain(
    connection: sqlite3.Connection,
    index: Any,
    dgt_ids: Mapping[tuple[str, str], str],
    eurocodes: EurocodeMap,
    shares: Callable[[int], Interval],
    table: Mapping[str, Any],
    life: Callable[[str], float],
) -> dict[str, Any]:
    """The app's ES estimate and applied ranges against DGT's lower bound per code."""

    year = 2025
    lower: dict[Cell, float] = defaultdict(float)
    masked = total = 0.0
    with DGT.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            cohort, units = int(row["cohort"]), float(row["vehicles"])
            if not PROFILE_FIRST_YEAR <= cohort < year:
                continue
            total += units
            if "¡" in row["marca"] or "¡" in row["modelo"]:
                masked += units
                continue
            vehicle_id = dgt_ids[row["marca"], row["modelo"]]
            identity = identity_key(index, vehicle_id, row["marca"], row["modelo"])
            lower[Cell(identity, row["marca"], row["modelo"], cohort)] += units
    registrations: dict[Cell, float] = defaultdict(float)
    for _geo, vehicle_id, make, model, cohort, _end, units in _observations(
        connection, ("ES",), "new_registrations"
    ):
        if PROFILE_FIRST_YEAR <= int(cohort) < year:
            identity = identity_key(index, vehicle_id, make, model)
            registrations[Cell(identity, make, model, int(cohort))] += units
    estimate = registrations_times_survival(registrations, shares, year=year)
    mid = {cell: values[1] for cell, values in estimate.items()}
    truth, _ = _allocate(eurocodes, "ES", lower)
    predicted, _ = _allocate(eurocodes, "ES", mid)
    scored = [
        item for item in _records("ES", truth, predicted, life) if item.predicted >= MIN_PREDICTED
    ]
    above = [item for item in scored if item.truth > interval(table, item)[1]]
    return {
        "truth": "DGT parque 2025, unmasked 2010-2024 cohorts: a lower bound per code",
        "masked_share_2010_2024": masked / total if total else None,
        "codes_scored": len(scored),
        "lower_bound_above_p90": len(above),
        "share_certainly_above_range": len(above) / len(scored) if scored else None,
        "expected_if_calibrated_at_most": 0.10,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT / ".local" / "evidence")
    parser.add_argument("--eurocode-dir", type=Path, default=ROOT / ".local/private/eurocode")
    parser.add_argument("--out-dir", type=Path, default=ROOT / ".local/private/backtests")
    parser.add_argument(
        "--age-anchor",
        type=Path,
        help="measure the snapshot's survival model with this age anchor artifact instead "
        "(a what-if: no range status is written)",
    )
    arguments = parser.parse_args(argv)

    snapshot_id, path = _ledger(arguments.root, None)
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    methods = connection.execute("SELECT DISTINCT survival_method FROM cohort_estimate").fetchall()
    if len(methods) != 1:
        raise SystemExit(f"expected one survival method, found {len(methods)}")
    survival_model = load_survival_model_for(methods[0][0])
    if arguments.age_anchor is not None:
        survival_model = load_age_anchored_survival_model(
            arguments.age_anchor, unanchored=survival_model
        )
    shares = _survival_shares(survival_model, None)
    app_shares = {
        geography: _survival_shares(survival_model, geography) for geography in (*TRUTHS, "ES")
    }
    stock = _observations(connection, TRUTHS, "active_fleet")
    registrations = _observations(connection, (*TRUTHS, "ES"), "new_registrations")

    dgt_ids: dict[tuple[str, str], str] = {}
    with DGT.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["marca"], row["modelo"])
            dgt_ids.setdefault(key, f"dgt-es|{row['marca']}|{row['modelo']}")
    extras = {(row[1], row[2], row[3]) for row in (*stock, *registrations)}
    extras |= {(vehicle_id, make, model) for (make, model), vehicle_id in dgt_ids.items()}
    baseline = load_identity_index(connection)
    index = load_identity_index(connection, extra_vehicles=extras)
    refold = sum(
        1
        for (vehicle_id,) in connection.execute("SELECT vehicle_id FROM canonical_vehicle")
        if (before := baseline.identity_for(vehicle_id)) is not None
        and before != index.identity_for(vehicle_id)
    )
    if refold:
        raise SystemExit(f"refold {refold}: extra vehicles changed how ranked vehicles fold")
    eurocodes = EurocodeMap.from_directory(arguments.eurocode_dir)
    official = _official({*TRUTHS, "ES"})
    summaries = eurocodes.summaries()

    def cell(vehicle_id: str, make: str, model: str, cohort: int) -> Cell:
        return Cell(
            identity_key(index, vehicle_id, make, model),
            fold_make(make) or make.casefold(),
            model,
            int(cohort),
        )

    truth_cells: dict[str, dict[Cell, float]] = defaultdict(lambda: defaultdict(float))
    dates: dict[str, str] = {}
    for geography, vehicle_id, make, model, cohort, end, units in stock:
        truth_cells[geography][cell(vehicle_id, make, model, cohort)] += units
        dates[geography] = max(dates.get(geography, ""), end)
    hidden: dict[str, dict[Cell, float]] = defaultdict(lambda: defaultdict(float))
    for geography, vehicle_id, make, model, cohort, _end, units in registrations:
        if geography in TRUTHS and int(cohort) >= PROFILE_FIRST_YEAR:
            hidden[geography][cell(vehicle_id, make, model, cohort)] += units

    variants: dict[str, list[CodeObservation]] = {"raw": [], "anchored": []}
    degraded: dict[str, dict[str, list[CodeObservation]]] = {
        "lost_vehicles_17pct": {},
        "publisher_mismatch_5pct": {},
        "first_word_model_names": {},
    }
    countries: dict[str, Any] = {}
    true_codes_by_country: dict[str, dict[str, float]] = {}
    totals: dict[str, dict[str, float]] = {}
    current_range: dict[str, Any] = {}
    for geography in TRUTHS:
        year = int(dates[geography][:4])

        def life(code: str, year: int = year) -> float:
            summary = summaries.get(code)
            if summary is None:
                return 0.0
            return pre_profile_share(
                summary.first_year,
                summary.last_year,
                profile_first_year=PROFILE_FIRST_YEAR,
                year=year,
            )

        true_cells = {c: u for c, u in truth_cells[geography].items() if c.cohort < year}
        anchor = _anchor(
            official, geography, year, _all_registrations(connection, geography), shares
        )

        def estimate(
            profile: Mapping[Cell, float],
            anchor: dict = anchor,
            year: int = year,
            geography: str = geography,
        ) -> tuple[dict[Cell, Interval], dict[Cell, float]]:
            cells = registrations_times_survival(profile, app_shares[geography], year=year)
            cells = {c: v for c, v in cells.items() if c.cohort < year}
            bare = registrations_times_survival(profile, shares, year=year)
            anchored = {
                c: v[1] * anchor["factor"].get(c.cohort, 1.0)
                for c, v in bare.items()
                if c.cohort < year
            }
            return cells, anchored

        cells, anchored_cells = estimate(hidden[geography])
        truth, truth_other = _allocate(eurocodes, geography, true_cells)
        raw, raw_other = _allocate(eurocodes, geography, _quantile(cells, 1))
        low, _ = _allocate(eurocodes, geography, _quantile(cells, 0))
        high, _ = _allocate(eurocodes, geography, _quantile(cells, 2))
        anchored, _ = _allocate(eurocodes, geography, anchored_cells)
        variants["raw"] += _records(geography, truth, raw, life)
        variants["anchored"] += _records(geography, truth, anchored, life)
        true_codes_by_country[geography] = truth
        totals[geography] = {
            "truth_all": sum(true_cells.values()),
            "estimate_all": sum(v[1] for v in cells.values()),
            "anchored_all": sum(anchored_cells.values()),
        }
        scored = [code for code, units in raw.items() if units >= MIN_PREDICTED]
        inside = [
            code for code in scored if low.get(code, 0) <= truth.get(code, 0) <= high.get(code, 0)
        ]
        current_range[geography] = len(inside) / len(scored) if scored else None
        missed = {
            code: units
            for code, units in truth.items()
            if units >= MIN_PREDICTED and raw.get(code, 0.0) < MIN_PREDICTED
        }
        for name, profile in (
            ("lost_vehicles_17pct", drop_cells(hidden[geography], 0.17, seed=geography)),
            ("publisher_mismatch_5pct", perturb_cells(hidden[geography], 0.05, seed=geography)),
            ("first_word_model_names", coarse_names(hidden[geography])),
        ):
            degraded_cells, _anchored = estimate(profile)
            codes, _ = _allocate(eurocodes, geography, _quantile(degraded_cells, 1))
            degraded[name][geography] = _records(geography, truth, codes, life)
        countries[geography] = {
            "register_date": dates[geography],
            "anchor": {key: value for key, value in anchor.items() if key != "factor"},
            "codes_with_truth": len(truth),
            "truth_certain": sum(truth.values()),
            "truth_shared_or_unassigned": sum(truth_other.values()),
            "estimate_certain": sum(raw.values()),
            "estimate_shared_or_unassigned": sum(raw_other.values()),
            "large_codes_missed": len(missed),
            "large_codes_missed_units": sum(missed.values()),
            "wape_large": {
                "raw": _wape(truth, raw),
                "anchored": _wape(truth, anchored),
            },
            "survival_band_coverage": current_range[geography],
            **totals[geography],
        }

    shares_by_country = {
        geography: {code: units / totals[geography]["truth_all"] for code, units in codes.items()}
        for geography, codes in true_codes_by_country.items()
    }
    for geography in TRUTHS:
        mix = donor_mix(shares_by_country, geography)
        truth = true_codes_by_country[geography]
        for name, total in (
            ("donor_mix_raw_total", totals[geography]["estimate_all"]),
            ("donor_mix_anchored_total", totals[geography]["anchored_all"]),
        ):
            countries[geography]["wape_large"][name] = _wape(
                truth, {code: share * total for code, share in mix.items()}
            )

    levels = {}
    for geography in TRUTHS:
        scored = [
            item
            for item in variants["raw"]
            if item.country == geography and item.predicted >= MIN_PREDICTED
        ]
        levels[geography] = sum(item.truth for item in scored) / sum(
            item.predicted for item in scored
        )
    variants["oracle_level"] = relative_to_level(variants["raw"], levels)

    results = {name: _gate_block(items) for name, items in variants.items()}
    for name in results:
        results[name]["log_ratio"] = _log_ratio_quantiles(variants[name])
    degradation = {name: _degraded_coverage(variants["raw"], by) for name, by in degraded.items()}
    applied_table = ratio_table(
        [item for item in variants["raw"] if item.predicted >= MIN_PREDICTED]
    )
    spain = _spain(
        connection,
        index,
        dgt_ids,
        eurocodes,
        app_shares["ES"],
        applied_table,
        lambda code: (
            pre_profile_share(
                summaries[code].first_year,
                summaries[code].last_year,
                profile_first_year=PROFILE_FIRST_YEAR,
                year=2025,
            )
            if code in summaries
            else 0.0
        ),
    )
    uk = _uk_stock(connection, index, eurocodes, shares)
    forecast = _forecast_kind(connection, index, eurocodes)
    connection.close()

    # The gate decides on the estimate the app shows. Anchoring is measured as the
    # change that would make a range hold; it is not what the app computes.
    applied = "raw"
    passed = results[applied]["gate"]["passes"]
    report = {
        "snapshot_id": snapshot_id,
        "survival_method": survival_model.method,
        "snapshot_survival_method": methods[0][0],
        "estimator": "EEA new registrations from 2010 x survival, at each register's date",
        "truths": list(TRUTHS),
        "min_predicted": MIN_PREDICTED,
        "quantiles": list(QUANTILES),
        "countries": countries,
        "fleet_kind": results,
        "degradation": degradation,
        "spain_one_sided": spain,
        "stock_kind_gb": uk,
        "forecast_kind": forecast,
        "baselines_not_run": {
            "ii_own_2010_mix_back_projected": (
                "needs model lineages the snapshot does not hold; not implemented"
            ),
            "it_mit_2019": "make x date only, so it cannot score a code; not downloaded",
            "pl_cepik": "crawl unfinished",
        },
        "leakage": [
            "identity vocabulary built with the registers' stock in the snapshot",
            "bins and gate constants fixed after harness v1's aggregate errors, before any per-code"
            " result",
            "registration series are today's revised values, not as published",
            "GB stock-kind check is in-sample for the survival curve",
        ],
    }
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    out = arguments.out_dir.resolve() / f"code-ranges-{snapshot_id}.json"
    out.write_text(
        json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    records = arguments.out_dir.resolve() / f"code-ranges-{snapshot_id}-records.json"
    records.write_text(
        json.dumps(
            {
                name: [
                    [item.country, item.code, item.truth, item.predicted, bin_of(item)]
                    for item in items
                ]
                for name, items in variants.items()
            }
        )
        + "\n",
        encoding="utf-8",
    )
    status = {
        "format": FORMAT,
        "version": VERSION,
        "snapshot_id": snapshot_id,
        "catalogue_sha256": eurocodes.catalogue_sha256,
        "decisions_snapshot_id": eurocodes.decisions.metadata.get("snapshot_id"),
        "method": "leave-one-register-out quantiles of truth over estimate, per code",
        "applies_to": "registrations x survival, as the app estimates a register-less market",
        "status": "passed" if passed else "failed",
        "gate": results[applied]["gate"],
        "pooled_coverage": results[applied]["coverage"]["pooled"],
        "per_country_coverage": {
            country: values["coverage"]
            for country, values in results[applied]["coverage"]["countries"].items()
        },
        "table": _table(ratio_table([i for i in variants[applied] if i.predicted >= MIN_PREDICTED]))
        if passed
        else None,
        "alternatives": {
            name: {
                "passes": results[name]["gate"]["passes"],
                "pooled_coverage": results[name]["coverage"]["pooled"],
                "countries_below_minimum": results[name]["gate"]["countries_below_minimum"],
                "estimator_in_app": False,
            }
            for name in ("anchored", "oracle_level")
        },
    }
    status["alternatives"]["anchored"]["countries_without_anchor"] = sorted(
        geography for geography, values in countries.items() if not values["anchor"]["available"]
    )
    if arguments.age_anchor is None:
        calibration = arguments.eurocode_dir.resolve() / "calibration"
        calibration.mkdir(exist_ok=True)
        (calibration / f"range-calibration-{snapshot_id}.json").write_text(
            json.dumps(status, indent=1, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print(f"{snapshot_id}: per-code fleet at each register's date; codes predicted >= 1,000")
    for name, block in results.items():
        values = block["coverage"]
        per = " ".join(
            f"{c} {'-' if v['coverage'] is None else format(v['coverage'], '.2f')}"
            for c, v in values["countries"].items()
        )
        print(
            f"{name:13s} pooled {values['pooled']:.3f} on {values['codes']} codes | {per} | "
            f"gate {'PASS' if block['gate']['passes'] else 'FAIL'}"
        )
    for name, block in degradation.items():
        print(f"degraded {name}: pooled {block['coverage']['pooled']:.3f}")
    print(f"spain: {spain['share_certainly_above_range']} certainly above P90")
    print(f"gb stock: wape {uk['wape_large']} band coverage {uk['survival_band_coverage']}")
    for step, block in forecast["steps"].items():
        print(f"forecast step {step}: pooled {block['coverage']['pooled']:.3f}")
    state = status["status"] if arguments.age_anchor is None else "not written (what-if)"
    print(survival_model.method)
    print(f"written: {out.name}, {records.name}; app range status {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

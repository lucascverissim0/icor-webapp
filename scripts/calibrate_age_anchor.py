#!/usr/bin/env python3
"""Calibrate per-market age anchors on Eurostat and write the reviewed artifact.

Reads the same pinned `road_eqs_carage` and `road_eqr_carmot` copies as
`calibrate_net_retention.py`, with the same clean market years (known age
convention, unflagged bands, registrations that hold no used imports), and fits
`icor.forecasting.age_anchor` on the net-retained survival model the snapshot
build uses. A market is kept only when its anchors fitted to 2022 predict
2023-2025 better than none. GB is excluded: the curve was measured there, and
Eurostat's UK counts another territory. Prints counts only.

usage: calibrate_age_anchor.py [--stock FILE] [--registrations FILE] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icor.evidence import eurostat_car_stock, eurostat_new_registrations  # noqa: E402
from icor.forecasting import age_anchor  # noqa: E402
from icor.forecasting.survival import (  # noqa: E402
    AGE_ANCHOR_PATH,
    NET_RETENTION_PATH,
    PROMOTED_CURVE_PATH,
    load_net_retained_survival_model,
)
from scripts.calibrate_net_retention import (  # noqa: E402
    BACKTEST_UNTIL,
    EXCLUDED,
    FIT_UNTIL,
    _newest,
    _sha256,
    market_years,
)

METHOD = "eurostat-age-anchor-v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stock", type=Path)
    parser.add_argument("--registrations", type=Path)
    parser.add_argument("--output", type=Path, default=AGE_ANCHOR_PATH)
    args = parser.parse_args(argv)
    stock_path = args.stock or _newest(eurostat_car_stock.DATASET)
    registrations_path = args.registrations or _newest(eurostat_new_registrations.DATASET)
    stock_document = json.loads(stock_path.read_text(encoding="utf-8"))
    registration_document = json.loads(registrations_path.read_text(encoding="utf-8"))

    model = load_net_retained_survival_model()
    scale = Decimal(1_000_000)
    support = len(model.band.p50.shares)

    markets: dict[str, dict] = {}
    rejected: dict[str, dict] = {}
    for geography, items in market_years(stock_document, registration_document).items():
        shares = [
            float(model.interval(scale, age_years=age, geography=geography).p50 / scale)
            for age in range(support)
        ]

        def share(age: int, shares: list[float] = shares) -> float:
            return shares[age]

        curve = age_anchor.fit(items, share, until=FIT_UNTIL, support=support)
        if curve is None:
            continue
        fitted_year = max(
            item.year
            for item in items
            if item.year <= FIT_UNTIL and age_anchor.band_anchors(item, share)
        )
        fitted = next(item for item in items if item.year == fitted_year)
        _curve, test = age_anchor.backtest(items, share, fit_until=BACKTEST_UNTIL, support=support)
        entry = {
            "fitted_year": fitted_year,
            "convention": fitted.convention,
            "bands": {
                name: round(value, 4)
                for name, value in age_anchor.band_anchors(fitted, share).items()
            },
            "backtest": {
                "fit_until": BACKTEST_UNTIL,
                "cells": len(test.cells),
                "unanchored_error": round(test.error(3), 4),
                "anchored_error": round(test.error(4), 4),
            },
        }
        if test.passes:
            markets[geography] = {"anchors": [f"{value:.4f}" for value in curve], **entry}
        else:
            rejected[geography] = entry
    document = {
        "method": METHOD,
        "base_method": model.method,
        "excluded_markets": sorted(EXCLUDED),
        "tolerance": age_anchor.TOLERANCE,
        "fit_until": FIT_UNTIL,
        "inputs": {
            "stock": {"file": stock_path.name, "sha256": _sha256(stock_path)},
            "registrations": {
                "file": registrations_path.name,
                "sha256": _sha256(registrations_path),
            },
            "base_curve": {
                "file": PROMOTED_CURVE_PATH.name,
                "sha256": _sha256(PROMOTED_CURVE_PATH),
            },
            "net_retention": {
                "file": NET_RETENTION_PATH.name,
                "sha256": _sha256(NET_RETENTION_PATH),
            },
        },
        "markets": markets,
        "rejected_markets": rejected,
    }
    args.output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for geography, entry in sorted({**markets, **rejected}.items()):
        test = entry["backtest"]
        state = "anchored" if geography in markets else "rejected"
        print(
            f"{geography} {state:<8} {entry['fitted_year']} {entry['bands']} backtest "
            f"{test['unanchored_error']:.3f} -> {test['anchored_error']:.3f} "
            f"({test['cells']} cells)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

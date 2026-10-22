#!/usr/bin/env python3
"""Write the committed Eurostat new-car totals that planning checks provisional years against.

Reads the pinned `road_eqr_carmot` copy (fetched by `scripts/report_eurocode_data.py
--fetch-eurostat`) and keeps every market's national totals from 2010, the first year
of model-level registrations. Counts only.

usage: pin_eurostat_registration_totals.py [--registrations FILE] [--output FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from icor.evidence import eurostat_new_registrations

ROOT = Path(__file__).resolve().parents[1]
EUROSTAT = ROOT / ".local" / "external-data" / "eurostat"
FIRST_YEAR = 2010


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registrations", type=Path)
    parser.add_argument("--output", type=Path, default=eurostat_new_registrations.TOTALS_PATH)
    args = parser.parse_args()
    path = (
        args.registrations
        or sorted(EUROSTAT.glob(f"{eurostat_new_registrations.DATASET}-*.json"))[-1]
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    publishers = document["dimension"]["geo"]["category"]["index"]
    geographies = {
        {"EL": "GR", "UK": "GB"}.get(code, code) for code in publishers if len(code) == 2
    }
    totals = {
        geography: {
            str(year): item.count
            for year, item in sorted(years.items())
            if year >= FIRST_YEAR and item.status != "d"
        }
        for geography, years in sorted(
            eurostat_new_registrations.parse(document, geographies).items()
        )
    }
    output = {
        "dataset": eurostat_new_registrations.DATASET,
        "licence": eurostat_new_registrations.LICENCE,
        "input": {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "totals": {geography: years for geography, years in totals.items() if years},
    }
    args.output.write_text(json.dumps(output, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

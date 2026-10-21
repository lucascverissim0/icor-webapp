#!/usr/bin/env python3
"""Crawl Poland's registered passenger cars from the CEPiK API, then aggregate them.

``crawl`` is resumable: the slice plan is fixed by the date stored in the state
directory on the first run, and every finished slice is skipped on the next one.
``aggregate`` refuses to write until every planned slice is finished.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

from icor.evidence.cepik_fleet_acquisition import (
    EARLIEST_WINDOW_YEAR,
    PASSENGER_KINDS,
    Slice,
    crawl,
    plan_slices,
    write_aggregate,
)

#: Only the dictionary's main spelling is crawled. In the 2022 bulk files the other
#: spellings are almost all deregistered: none of Lubuskie's 473,210 registered
#: passenger cars and 192 of Mazowieckie's 2,687,152 (0.007%). Filtering on a rare
#: spelling also makes the API time out (HTTP 504), as it scans the whole window.
DEFAULT_KINDS = PASSENGER_KINDS[:1]
#: The model takes register cohorts from 2006 on; older windows (about a third of
#: Poland's registered cars) would double the crawl for cohorts it does not use.
DEFAULT_FROM_YEAR = 2006


def _plan(state_dir: Path) -> tuple[Slice, ...]:
    plan = state_dir / "plan.json"
    if not plan.exists():
        state_dir.mkdir(parents=True, exist_ok=True)
        document = {
            "plan_date": date.today().isoformat(),
            "kinds": list(DEFAULT_KINDS),
            "from_year": DEFAULT_FROM_YEAR,
        }
        plan.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
    document = json.loads(plan.read_text(encoding="utf-8"))
    return plan_slices(
        date.fromisoformat(document["plan_date"]),
        document["kinds"],
        from_year=document.get("from_year", EARLIEST_WINDOW_YEAR),
    )


def _log(message: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    crawl_parser = commands.add_parser("crawl")
    crawl_parser.add_argument("--state-dir", required=True, type=Path)
    crawl_parser.add_argument("--workers", type=int, default=2)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--state-dir", required=True, type=Path)
    aggregate_parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()

    slices = _plan(args.state_dir)
    slice_dir = args.state_dir / "slices"
    if args.command == "crawl":
        try:
            crawl(slices, slice_dir, workers=args.workers, log=_log)
        except RuntimeError as error:
            _log(str(error))
            return 1
        _log("crawl complete")
        return 0
    result = write_aggregate(slice_dir, slices, args.destination)
    payload = asdict(result)
    payload["path"] = str(payload["path"])
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

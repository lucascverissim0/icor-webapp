#!/usr/bin/env python3
"""Acquire Denmark's registered passenger cars as a counts-only aggregate and profile.

The Motorstyrelsen DMR extract (about 7.1 GB zipped, 129 GB of XML) is streamed from its
anonymous FTP server and never stored; run it detached and follow its log:

    nohup uv run python scripts/acquire_dk_fleet.py \\
        --destination .local/pinned-artifacts/dk-registered-cars-{date}.csv \\
        > .local/evidence/downloads/dk/acquire.log 2>&1 &

``{date}`` in ``--destination`` becomes the YYYYMMDD of the snapshot that was read.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from icor.evidence.dk_fleet_acquisition import acquire_fleet


def _log(message: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="the aggregate CSV; {date} becomes the snapshot's YYYYMMDD",
    )
    parser.add_argument(
        "--name",
        help="read this ESStatistikListeModtag-YYYYMMDD-HHMMSS.zip instead of the newest",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=180.0,
        help="seconds between progress lines",
    )
    args = parser.parse_args()
    result = acquire_fleet(args.destination, name=args.name, log=_log, interval=args.log_interval)
    payload = asdict(result)
    payload["path"] = str(payload["path"])
    payload["profile_path"] = str(payload["profile_path"])
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

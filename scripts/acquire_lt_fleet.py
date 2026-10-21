#!/usr/bin/env python3
"""Acquire Lithuania's registered passenger-car aggregate from Regitra's open data.

The ZIP needs random access, so it is downloaded to ``--archive`` (default
``.local/evidence/downloads/lt/``), aggregated and then deleted. ``--destination`` may
contain ``{date}``, which becomes the snapshot date (the CSV member's date stamp) as
YYYYMMDD, for example ``.local/pinned-artifacts/lt-registered-cars-{date}.csv``. The
variant that also counts cars not in traffic and the counts-only profile are written
beside it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from icor.evidence.lt_fleet_acquisition import (
    acquire_fleet,
    destination_for,
    download_archive,
    snapshot_date,
)

DEFAULT_ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / ".local"
    / "evidence"
    / "downloads"
    / "lt"
    / "Atviri_TP_parko_duomenys.zip"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="the aggregate CSV; {date} is replaced by the snapshot date as YYYYMMDD",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="where the ZIP is downloaded; it is deleted after aggregation",
    )
    args = parser.parse_args()
    archive: Path = args.archive
    download = download_archive(archive)
    try:
        destination = destination_for(args.destination, snapshot_date(archive))
        result = acquire_fleet(destination, archive, download=download)
    finally:
        if archive.exists():
            archive.unlink()
    payload = asdict(result)
    payload["path"] = str(payload["path"])
    payload["profile_path"] = str(payload["profile_path"])
    payload["variant"]["path"] = str(payload["variant"]["path"])
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

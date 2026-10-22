#!/usr/bin/env python3
"""Acquire the checksum-reporting Slovenian registered passenger-car aggregate.

The source ZIP holds VINs and owner data, so a ZIP downloaded here is deleted once it
has been aggregated; one supplied with ``--archive`` is left for its owner to delete.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from icor.evidence.si_fleet_acquisition import acquire_fleet, download_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--model-head-destination",
        type=Path,
        help="also write an aggregate whose model is the text before the first slash",
    )
    parser.add_argument(
        "--model-head",
        action="store_true",
        help=(
            "write the model-head aggregate, the release ICOR loads, to --destination and "
            "the full commercial labels beside it"
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="a previously downloaded pinned ZIP; it is kept, while a downloaded one is deleted",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path(".local/evidence/downloads/si"),
        help="where the ZIP is downloaded when --archive is not given",
    )
    args = parser.parse_args()
    destination, head_destination = args.destination, args.model_head_destination
    if args.model_head:
        head_destination = args.destination
        destination = args.destination.with_name(f"{args.destination.stem}-full-labels.csv")
    archive = args.archive
    downloaded = archive is None
    last_modified: dict[str, str] = {}
    if downloaded:
        download = download_archive(args.download_dir / "opsi-si-vehicles-20230630.zip")
        archive = download.path
        last_modified = {
            "last_modified": download.last_modified or "",
            "retrieved_utc": download.retrieved_utc,
        }
        last_modified = {key: value for key, value in last_modified.items() if value}
    try:
        result = acquire_fleet(
            destination,
            archive,
            model_head_destination=head_destination,
            **last_modified,
        )
    finally:
        if downloaded and archive.exists():
            archive.unlink()
    print(json.dumps(asdict(result), sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

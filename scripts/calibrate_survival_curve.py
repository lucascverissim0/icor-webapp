#!/usr/bin/env python3
"""Produce the versioned licensed-stock survival curve the application serves.

The production model does not recalibrate during a snapshot build. A curve is
calibrated here, written to a provenance-carrying artifact, reviewed, and
committed, so the curve a snapshot used is a reviewable fact rather than a side
effect of whatever data happened to be on disk that day.

The UK registrations artifact is an official DfT release pinned by URL, byte
count and SHA-256. It is read only to anchor age one; it is deliberately not
loaded into the evidence set, because GB registrations are already there and a
UK-wide copy of the same register would double count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from icor.forecasting.survival_calibration import calibrate_licensed_stock_band
from icor.infrastructure.snapshot_store import SnapshotStore

_DEFAULT_OUTPUT = Path("src/icor/forecasting/survival_curves/uk_licensed_stock.json")
_PINNED_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "6a54d2eca6586e258d371d71/df_VEH0160_UK.csv"
)
_PINNED_BYTES = 10_092_936
_PINNED_SHA256 = "f5390dfb66087b4299fffa2fe77c32fe35cf2dcbdfb0db70d38c6d4926f7abce"
_TERMS_URL = "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".local/evidence"))
    parser.add_argument("--uk-registrations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser


def _verify_pinned(artifact: Path) -> None:
    digest = hashlib.sha256()
    size = 0
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    if size != _PINNED_BYTES or digest.hexdigest() != _PINNED_SHA256:
        raise SystemExit(
            "UK registrations artifact does not match the pinned release; refusing "
            "to calibrate from an unverified file"
        )


def main() -> int:
    args = _parser().parse_args()
    _verify_pinned(args.uk_registrations)

    # The sibling benchmark owns the pinned parsers for both inputs; reuse them
    # rather than maintaining a second copy that could drift.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import scripts.benchmark_survival_calibration as benchmark

    manifest, repository = SnapshotStore(args.root).open_active_snapshot()
    registrations = benchmark._load_registrations(args.uk_registrations)
    stocks = benchmark._load_stocks(repository.path)
    band = calibrate_licensed_stock_band(registrations, stocks)

    payload = {
        "method": "uk-dft-licensed-stock-band-v1",
        "calibrated_geography": "UK",
        "calibrated_support_age": band.p50.calibrated_support_age,
        "anchor_cohort_count": band.p50.anchor_cohort_count,
        "band_cohort_counts": [list(item) for item in band.cohort_counts],
        "shares": {
            "p10": [str(value) for value in band.p10.shares],
            "p50": [str(value) for value in band.p50.shares],
            "p90": [str(value) for value in band.p90.shares],
        },
        "calibrated_at": datetime.now(UTC).isoformat(),
        "stock_snapshot_id": manifest.snapshot_id,
        "registration_source": {
            "publisher": "UK Department for Transport / DVLA",
            "url": _PINNED_URL,
            "artifact_bytes": _PINNED_BYTES,
            "sha256": _PINNED_SHA256,
            "terms_url": _TERMS_URL,
            "permitted_local_use": (
                "Reuse permitted with Crown copyright attribution under OGL v3.0."
            ),
        },
        "limitations": (
            "UK licensed stock is an administrative proxy affected by imports, "
            "exports, relicensing, scrappage and record revisions. The curve is "
            "measured on one country and aggregated across makes and models; every "
            "other geography receives it as an explicit transfer."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "output": str(args.output),
        "support_age": len(band.p50.shares) - 1,
        "age_one_share": str(band.p50.shares[1]),
        "age_ten_share": str(band.p50.shares[min(10, len(band.p50.shares) - 1)]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

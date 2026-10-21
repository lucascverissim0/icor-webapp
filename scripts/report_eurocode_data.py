#!/usr/bin/env python3
"""Report, per Eurocode, the evidence behind it and how many of its cars are on the road.

The report names real Eurocodes, which are ICOR's private data, so it is written
only under the git-ignored ``.local/private/eurocode/reports`` and the script
refuses any other output directory. It prints counts only.

    uv run python scripts/report_eurocode_data.py --fetch-eurostat

reads the active snapshot read-only and writes, for that snapshot,
``data-coverage-<snapshot>.json`` (also served by the local app at
``/api/v1/eurocodes/data-coverage``), ``.csv`` and a self-contained ``.html``.

``--fetch-eurostat`` downloads Eurostat's passenger cars by age once into
``.local/external-data/eurostat`` with its sha256; later runs reuse the newest
file. Without any file, the country comparison is left empty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import ssl
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import certifi

from icor.eurocodes.catalog import from_document
from icor.eurocodes.data_coverage import EVIDENCE_KINDS, build_report
from icor.eurocodes.runtime import EurocodeMap
from icor.evidence import eurostat_car_stock
from icor.forecasting.survival import load_promoted_survival_model
from icor.infrastructure.snapshot_data_coverage import read_fleet
from icor.infrastructure.snapshot_eurocodes import measure
from icor.infrastructure.snapshot_store import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".local" / "private" / "eurocode"
EUROSTAT_DIR = ROOT / ".local" / "external-data" / "eurostat"


def _within_private(path: Path) -> bool:
    try:
        path.resolve().relative_to(PRIVATE.resolve())
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_eurostat() -> Path:
    EUROSTAT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    target = EUROSTAT_DIR / f"{eurostat_car_stock.DATASET}-{stamp}.json"
    partial = target.with_suffix(".part")
    context = ssl.create_default_context(cafile=certifi.where())
    with (
        urllib.request.urlopen(eurostat_car_stock.URL, timeout=120, context=context) as response,
        partial.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle)
    json.loads(partial.read_text(encoding="utf-8"))
    partial.replace(target)
    target.with_suffix(".sha256").write_text(f"{_sha256(target)}  {target.name}\n")
    return target


def _latest_eurostat() -> Path | None:
    files = sorted(EUROSTAT_DIR.glob(f"{eurostat_car_stock.DATASET}-*.json"))
    return files[-1] if files else None


def _write_csv(report: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "code",
                "label",
                "tier",
                "first_year",
                "last_year",
                "drive",
                "cars_certain_p50",
                "cars_certain_p10",
                "cars_certain_p90",
                "cars_shared_p50",
                "cars_certain_2028",
                "observed_share",
                *(f"cars_{kind}" for kind in EVIDENCE_KINDS),
                "markets",
                "observed_markets",
                "sources",
                "life_years",
                "observed_life_years",
                "years_before_2010",
                "flags",
                "shared_with",
            ]
        )
        for item in report["codes"]:
            fleet = item["fleet"]
            writer.writerow(
                [
                    item["code"],
                    item["label"],
                    item["tier"],
                    item["first_year"],
                    item["last_year"] or "",
                    item["drive"],
                    fleet["certain_p50"],
                    fleet["certain_p10"],
                    fleet["certain_p90"],
                    fleet["shared_p50"],
                    fleet["certain_2028"],
                    item["observed_share"] if item["observed_share"] is not None else "",
                    *(item["evidence"][kind] for kind in EVIDENCE_KINDS),
                    item["markets"],
                    " ".join(item["observed_markets"]),
                    " ".join(item["sources"]),
                    item["life_years"],
                    item["observed_life_years"],
                    item["years_before_2010"],
                    " ".join(item["flags"]),
                    " ".join(item["shared_with"]),
                ]
            )
        for item in report["no_data"]:
            writer.writerow(
                [
                    item["code"],
                    item["label"] or " | ".join(item["descriptions"]),
                    "D",
                    item["first_year"] or "",
                    item["last_year"] or "",
                    "",
                    0,
                    0,
                    0,
                    0,
                    0,
                    "",
                    *(0 for _kind in EVIDENCE_KINDS),
                    0,
                    "",
                    "",
                    0,
                    0,
                    0,
                    "reviewed-no-fleet" if item["reviewed"] else "not-reviewed",
                    "",
                ]
            )


TEMPLATE = Path(__file__).with_suffix(".html")


def _write_html(report: dict, path: Path) -> None:
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(
        TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.getenv("ICOR_EVIDENCE_ACTIVE_ROOT", ROOT / ".local" / "evidence")),
    )
    parser.add_argument("--eurocode-dir", type=Path, default=PRIVATE)
    parser.add_argument("--out-dir", type=Path, default=PRIVATE / "reports")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--eurostat", type=Path, help="road_eqs_carage JSON-stat file")
    parser.add_argument("--fetch-eurostat", action="store_true")
    arguments = parser.parse_args(argv)

    if not _within_private(arguments.out_dir):
        print(f"Refusing to write outside {PRIVATE.relative_to(ROOT)}.", file=sys.stderr)
        return 2
    eurocodes = EurocodeMap.from_directory(arguments.eurocode_dir)
    catalogue = from_document(
        json.loads((arguments.eurocode_dir / "catalog.json").read_text(encoding="utf-8"))
    )
    manifest, ledger = SnapshotStore(
        arguments.root, trust_verified_marker=True
    ).open_active_snapshot()

    eurostat_path = arguments.eurostat
    if eurostat_path is None and arguments.fetch_eurostat and _latest_eurostat() is None:
        eurostat_path = _fetch_eurostat()
    eurostat_path = eurostat_path or _latest_eurostat()

    connection = sqlite3.connect(f"{ledger.path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        geographies = {
            row[0] for row in connection.execute("SELECT DISTINCT geography FROM cohort_estimate")
        }
        official = {}
        official_source = {}
        comparison_years: dict[str, int] = {}
        if eurostat_path is not None:
            document = json.loads(eurostat_path.read_text(encoding="utf-8"))
            official = eurostat_car_stock.parse(document, geographies)
            comparison_years = {geography: stock.year for geography, stock in official.items()}
            official_source = {
                "dataset": eurostat_car_stock.DATASET,
                "url": eurostat_car_stock.URL,
                "licence": eurostat_car_stock.LICENCE,
                "file": eurostat_path.name,
                "sha256": _sha256(eurostat_path),
                "updated": document.get("updated"),
                "missing_markets": sorted(geographies - set(official)),
            }
        fleet = read_fleet(
            connection,
            load_promoted_survival_model(),
            year=arguments.year,
            comparison_years=comparison_years,
        )
        status = measure(eurocodes, connection, manifest.snapshot_id)
    finally:
        connection.close()

    report = build_report(
        eurocodes,
        fleet.cells,
        year=arguments.year,
        snapshot_id=manifest.snapshot_id,
        catalogue_rows=catalogue.rows,
        official_stock=official,
        official_fleet=fleet.comparison,
        official_source=official_source,
    )
    report["verified_cohorts"] = fleet.verified_cohorts
    report["demand_coverage"] = [
        {
            "horizon": item.horizon,
            "total_units": item.total_units,
            "single_units": item.single_units,
            "shared_units": item.shared_units,
            "outside_units": item.outside_units,
            "unassignable_units": item.unassignable_units,
            "unreviewed_units": item.unreviewed_units,
        }
        for item in status.coverage
    ]

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    stem = arguments.out_dir / f"data-coverage-{manifest.snapshot_id}"
    stem.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    _write_csv(report, stem.with_suffix(".csv"))
    _write_html(report, stem.with_suffix(".html"))

    totals = report["totals"]
    print(
        f"snapshot {manifest.snapshot_id}; "
        f"2028 cohorts recomputed exactly: {fleet.verified_cohorts:,}"
    )
    print(f"cars on the road at end of {arguments.year} in the model: {totals['fleet']:,}")
    print("by evidence:", {kind: f"{units:,}" for kind, units in totals["by_evidence"].items()})
    print("by outcome:", {status: f"{units:,}" for status, units in totals["by_outcome"].items()})
    for tier, values in totals["tiers"].items():
        print(f"tier {tier}: {values['codes']:,} codes, {values['certain_p50']:,} certain cars")
    if report["country_gap"]:
        official_total = sum(item["official"] for item in report["country_gap"])
        modelled_total = sum(item["modelled"] for item in report["country_gap"])
        print(
            f"official stock (years {sorted(set(comparison_years.values()))}) "
            f"in {len(report['country_gap'])} markets: "
            f"{official_total:,}; model: {modelled_total:,}"
        )
    print(f"written: {stem.relative_to(ROOT)}.{{json,csv,html}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Report, per Eurocode, the evidence behind it and how many of its cars are on the road.

The report names real Eurocodes, which are ICOR's private data, so it is written
only under the git-ignored ``.local/private/eurocode/reports`` and the script
refuses any other output directory. It prints counts only.

    uv run python scripts/report_eurocode_data.py --fetch-eurostat

reads the active snapshot read-only and writes, for that snapshot,
``data-coverage-<snapshot>.json`` (also served by the local app at
``/api/v1/eurocodes/data-coverage``), ``.csv`` and a self-contained ``.html``.

``--fetch-eurostat`` downloads Eurostat's passenger cars by age and new
passenger-car registrations once into ``.local/external-data/eurostat`` with
their sha256; later runs reuse the newest files. Without the stock file the
country comparison is left empty; without the registrations it has no estimates.

``--candidate <snapshot-id>`` reports on a built, not yet promoted candidate
under ``<root>/candidates`` instead of the active snapshot.
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
from decimal import Decimal
from pathlib import Path

import certifi

from icor.eurocodes.catalog import from_document
from icor.eurocodes.data_coverage import EVIDENCE_KINDS, build_report
from icor.eurocodes.runtime import EurocodeMap
from icor.evidence import eurostat_age_conventions, eurostat_car_stock, eurostat_new_registrations
from icor.evidence.release_manifests import load_snapshot_manifest
from icor.forecasting.survival import load_survival_model_for
from icor.infrastructure.snapshot_data_coverage import country_comparisons, read_fleet
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


def _fetch_eurostat(dataset: str, url: str) -> Path:
    EUROSTAT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    target = EUROSTAT_DIR / f"{dataset}-{stamp}.json"
    partial = target.with_suffix(".part")
    context = ssl.create_default_context(cafile=certifi.where())
    with (
        urllib.request.urlopen(url, timeout=120, context=context) as response,
        partial.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle)
    json.loads(partial.read_text(encoding="utf-8"))
    partial.replace(target)
    target.with_suffix(".sha256").write_text(f"{_sha256(target)}  {target.name}\n")
    return target


def _latest_eurostat(dataset: str) -> Path | None:
    files = sorted(EUROSTAT_DIR.glob(f"{dataset}-*.json"))
    return files[-1] if files else None


def _eurostat_file(dataset: str, url: str, given: Path | None, fetch: bool) -> Path | None:
    if given is not None:
        return given
    if fetch and _latest_eurostat(dataset) is None:
        return _fetch_eurostat(dataset, url)
    return _latest_eurostat(dataset)


def _source(module, path: Path, document: dict) -> dict:
    return {
        "dataset": module.DATASET,
        "url": module.URL,
        "licence": module.LICENCE,
        "file": path.name,
        "sha256": _sha256(path),
        "updated": document.get("updated"),
    }


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
    parser.add_argument(
        "--eurostat-registrations", type=Path, help="road_eqr_carmot JSON-stat file"
    )
    parser.add_argument("--fetch-eurostat", action="store_true")
    parser.add_argument("--candidate", help="a candidate snapshot id under <root>/candidates")
    arguments = parser.parse_args(argv)

    if not _within_private(arguments.out_dir):
        print(f"Refusing to write outside {PRIVATE.relative_to(ROOT)}.", file=sys.stderr)
        return 2
    eurocodes = EurocodeMap.from_directory(arguments.eurocode_dir)
    catalogue = from_document(
        json.loads((arguments.eurocode_dir / "catalog.json").read_text(encoding="utf-8"))
    )
    if arguments.candidate:
        directory = arguments.root / "candidates" / arguments.candidate
        manifest = load_snapshot_manifest(directory / "snapshot.json")
        if manifest.snapshot_id != arguments.candidate:
            print("The candidate's manifest names another snapshot.", file=sys.stderr)
            return 2
        ledger_path = directory / "evidence.sqlite3"
    else:
        manifest, ledger = SnapshotStore(
            arguments.root, trust_verified_marker=True
        ).open_active_snapshot()
        ledger_path = ledger.path

    eurostat_path = _eurostat_file(
        eurostat_car_stock.DATASET,
        eurostat_car_stock.URL,
        arguments.eurostat,
        arguments.fetch_eurostat,
    )
    registrations_path = _eurostat_file(
        eurostat_new_registrations.DATASET,
        eurostat_new_registrations.URL,
        arguments.eurostat_registrations,
        arguments.fetch_eurostat,
    )

    connection = sqlite3.connect(f"{ledger_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        methods = {
            row[0]
            for row in connection.execute("SELECT DISTINCT survival_method FROM cohort_estimate")
        }
        if len(methods) != 1:
            print(f"The snapshot records {len(methods)} survival methods.", file=sys.stderr)
            return 2
        survival = load_survival_model_for(methods.pop())
        geographies = {
            row[0] for row in connection.execute("SELECT DISTINCT geography FROM cohort_estimate")
        }
        official = {}
        official_source: dict = {}
        youngest: dict[str, dict[int, int]] = {}
        comparison_years: dict[str, set[int]] = {}
        if eurostat_path is not None:
            document = json.loads(eurostat_path.read_text(encoding="utf-8"))
            official = eurostat_car_stock.parse(document, geographies)
            youngest = eurostat_car_stock.band_by_year(document, geographies, "Y_LT2")
            comparison_years = {
                geography: {stock.year, *((stock.band_year,) if stock.band_year else ())}
                for geography, stock in official.items()
            }
            official_source = {
                **_source(eurostat_car_stock, eurostat_path, document),
                "missing_markets": sorted(geographies - set(official)),
            }
        fleet = read_fleet(
            connection, survival, year=arguments.year, comparison_years=comparison_years
        )
        status = measure(eurocodes, connection, manifest.snapshot_id)
    finally:
        connection.close()

    registrations: dict = {}
    if registrations_path is not None:
        document = json.loads(registrations_path.read_text(encoding="utf-8"))
        registrations = eurostat_new_registrations.parse(document, geographies)
        official_source["registrations"] = _source(
            eurostat_new_registrations, registrations_path, document
        )
    conventions = {}
    for geography, stock in official.items():
        convention = eurostat_age_conventions.for_stock(
            stock, youngest.get(geography, {}), registrations.get(geography, {})
        )
        conventions[geography] = convention.convention if convention else None
    scale = Decimal(1_000_000)

    def share(geography: str, age: int) -> Decimal:
        return survival.interval(scale, age_years=age, geography=geography).p50 / scale

    countries = country_comparisons(fleet, official, registrations, conventions, share)

    report = build_report(
        eurocodes,
        fleet.cells,
        year=arguments.year,
        snapshot_id=manifest.snapshot_id,
        catalogue_rows=catalogue.rows,
        countries=countries,
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
    prefix = "data-coverage-candidate-" if arguments.candidate else "data-coverage-"
    stem = arguments.out_dir / f"{prefix}{manifest.snapshot_id}"
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
        years = sorted({item["year"] for item in report["country_gap"]})
        print(
            f"official stock (years {years}) in {len(report['country_gap'])} markets: "
            f"{official_total:,}; model: {modelled_total:,}"
        )
        for item in report["country_gap"]:
            under = item["under_20"]
            print(
                f"  {item['geography']} {item['year']}: official {item['official']:,}, "
                f"model {item['modelled']:,} ({item['modelled_share']:.1%}); "
                f"estimated {item['estimated']}"
                + (
                    f"; under 20 y ({item['band_year']}, {item['convention']}): "
                    f"official {under['official']:,}, model {under['modelled']:,} "
                    f"({under['modelled_share']:.1%})"
                    if under
                    else ""
                )
            )
    print(f"written: {stem.relative_to(ROOT)}.{{json,csv,html}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

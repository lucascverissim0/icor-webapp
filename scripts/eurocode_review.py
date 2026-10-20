#!/usr/bin/env python3
"""Prepare, validate, compare and merge the Eurocode mapping review.

The review links every vehicle identity that carries forecast demand to ICOR's
Eurocodes. Two reviewers work each batch blind; their outputs are validated
against the catalogue and compared year by year, weighted by demand, so every
disagreement is adjudicated rather than averaged away.

All inputs and outputs are ICOR-private and live under the git-ignored
``.local/private/eurocode``; the script refuses to write anywhere else and
prints counts and identity names only, never a code.

    uv run python scripts/eurocode_review.py prepare
    uv run python scripts/eurocode_review.py validate <reviewer.json>
    uv run python scripts/eurocode_review.py compare <a.json> <b.json>
    uv run python scripts/eurocode_review.py merge <decided.json>...
    uv run python scripts/eurocode_review.py coverage
    uv run python scripts/eurocode_review.py preview <reviewer.json> [family...]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from icor.eurocodes.allocation import (
    DECISIONS_FORMAT,
    DECISIONS_VERSION,
    EurocodeDecisions,
    body_hint,
    drive_side,
)
from icor.eurocodes.catalog import from_document
from icor.infrastructure.snapshot_identity import load_identity_index

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".local" / "private" / "eurocode"
REVIEW = PRIVATE / "review"
IDENTITY_RULES = ROOT / "data" / "vehicle_identity.json"
HORIZON = 2028
COVERAGE_TARGET = 0.99
MAX_BATCH_IDENTITIES = 70
SMALL_MAKE_IDENTITIES = 12
EXCLUDED_SCOPES = frozenset({"truck", "bus"})
RAW_PREFIX = "raw"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within_private(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(PRIVATE.resolve())
    return resolved


def _write_json(path: Path, document: object) -> None:
    _within_private(path).parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )


def _active_snapshot() -> tuple[str, Path]:
    active = json.loads((ROOT / ".local" / "evidence" / "active.json").read_text())
    snapshot_id = active["snapshot_id"]
    return (
        snapshot_id,
        ROOT / ".local" / "evidence" / "snapshots" / snapshot_id / "evidence.sqlite3",
    )


def identity_key(index: Any, vehicle_id: str, make: str, model: str) -> str:
    """The key a vehicle's registrations are reviewed and allocated under."""

    key = index.identity_for(vehicle_id)
    if key is not None:
        return f"{key[0]}|{key[1]}"
    return f"{RAW_PREFIX}|{' '.join(make.split()).casefold()}|{' '.join(model.split()).casefold()}"


def build_demand(database: Path) -> dict[str, Any]:
    """Horizon demand per identity, registration year and geography, read-only."""

    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro&immutable=1", uri=True)
    try:
        index = load_identity_index(connection)
        labels = {
            vehicle_id: (make, model)
            for vehicle_id, make, model in connection.execute(
                "SELECT vehicle_id, make, model FROM canonical_vehicle"
            )
        }
        rows = connection.execute(
            """SELECT o.canonical_vehicle_id, o.geography, a.registration_cohort_year,
                SUM(a.base_units)
            FROM opportunity_cohort_attribution a
            JOIN opportunity_estimate o ON o.opportunity_id = a.opportunity_id
            WHERE o.horizon_year = ?
            GROUP BY 1, 2, 3""",
            (HORIZON,),
        ).fetchall()
    finally:
        connection.close()
    identities: dict[str, dict[str, Any]] = {}
    for vehicle_id, geography, year, units in rows:
        if not units:
            continue
        make, model = labels[vehicle_id]
        key = identity_key(index, vehicle_id, make, model)
        entry = identities.setdefault(
            key, {"units": 0, "cells": defaultdict(int), "spellings": defaultdict(int)}
        )
        entry["units"] += int(units)
        hint = body_hint(model) or ""
        entry["cells"][f"{geography}|{year}|{hint}"] += int(units)
        entry["spellings"][f"{make} / {model}"] += int(units)
    for entry in identities.values():
        entry["cells"] = dict(entry["cells"])
        entry["spellings"] = dict(entry["spellings"])
    return identities


def _make_of(key: str) -> str:
    parts = key.split("|")
    return parts[1] if parts[0] == RAW_PREFIX else parts[0]


def _groups(path: Path) -> dict[str, str]:
    """Junk or corporate make spellings -> the make whose catalogue rows apply."""

    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8"))["groups"])


def _splits(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Large makes split into batches by model family, so no family spans two."""

    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        make: [(name, pattern) for name, pattern in rules]
        for make, rules in document.get("splits", {}).items()
    }


def _catalogue_lines(catalogue: Any, makes: set[str]) -> list[str]:
    lines = []
    for code, rows in catalogue.codes().items():
        relevant = [
            row for row in rows if set(row.makes) & makes and row.scope not in EXCLUDED_SCOPES
        ]
        if not relevant:
            continue
        starts = [row.start for row in relevant if row.start is not None]
        ends = [row.end for row in relevant if row.end is not None]
        if starts and max(starts) < 1985 and (not ends or max(ends) < 1988):
            continue
        for row in sorted(relevant, key=lambda item: item.row):
            notes = ",".join(row.flags)
            lines.append(
                f"r{row.row} | {code} | {row.brand} | {row.description} | "
                f"sop={row.sop or ''} eop={row.eop or ''} | {row.scope}"
                + (f" | {notes}" if notes else "")
            )
    return lines


def _identity_lines(key: str, entry: Mapping[str, Any], total: float) -> list[str]:
    by_year: dict[int, int] = defaultdict(int)
    rhd = 0
    bodies: dict[str, int] = defaultdict(int)
    for cell, units in entry["cells"].items():
        geography, year, hint = cell.split("|")
        by_year[int(year)] += units
        if drive_side(geography) == "RHD":
            rhd += units
        if hint:
            bodies[hint] += units
    units = entry["units"]
    years = " ".join(
        f"{year}:{value}"
        for year, value in sorted(by_year.items())
        if value >= max(1, 0.005 * units)
    )
    spellings = sorted(entry["spellings"].items(), key=lambda item: -item[1])[:8]
    lines = [
        f"### {key}",
        f"- demand {units} units ({units / total:.3%} of the market); RHD share {rhd / units:.0%}",
        f"- by registration year: {years}",
        "- top raw spellings: " + "; ".join(f"{label} ({value})" for label, value in spellings),
    ]
    if bodies:
        lines.append(
            "- units whose raw label names a body: "
            + ", ".join(f"{body} {value}" for body, value in sorted(bodies.items()))
        )
    return lines


def prepare(arguments: argparse.Namespace) -> int:
    snapshot_id, database = _active_snapshot()
    catalogue_path = PRIVATE / "catalog.json"
    catalogue = from_document(json.loads(catalogue_path.read_text(encoding="utf-8")))
    demand = build_demand(database)
    total = sum(entry["units"] for entry in demand.values())
    ranked = sorted(demand.items(), key=lambda item: -item[1]["units"])
    selected: list[str] = []
    covered = 0
    for key, entry in ranked:
        if covered / total >= arguments.target:
            break
        selected.append(key)
        covered += entry["units"]

    groups = _groups(REVIEW / "make_groups.json")
    by_make: dict[str, list[str]] = defaultdict(list)
    for key in selected:
        make = _make_of(key)
        by_make[groups.get(make, make)].append(key)
    batches: list[tuple[str, list[str], set[str]]] = []
    small: list[str] = []
    splits = _splits(REVIEW / "make_groups.json")
    for make, keys in sorted(by_make.items(), key=lambda item: -len(item[1])):
        catalogue_makes = set(make.split("+"))
        if len(keys) < SMALL_MAKE_IDENTITIES:
            small.append(make)
            continue
        keys.sort(key=lambda key: key.split("|")[-1])
        parts: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            model = key.split("|")[-1]
            name = next(
                (name for name, pattern in splits.get(make, ()) if re.match(pattern, model)),
                make,
            )
            parts[name].append(key)
        for name, chunk in parts.items():
            batches.append((name.replace(" ", "_"), chunk, catalogue_makes))
    bundle: list[str] = []
    bundle_keys: list[str] = []
    for make in sorted(small):
        bundle.append(make)
        bundle_keys.extend(by_make[make])
        if len(bundle_keys) >= MAX_BATCH_IDENTITIES // 2:
            makes = {part for name in bundle for part in name.split("+")}
            batches.append((f"mixed-{len(batches) + 1:02d}", bundle_keys, makes))
            bundle, bundle_keys = [], []
    if bundle_keys:
        makes = {part for name in bundle for part in name.split("+")}
        batches.append((f"mixed-{len(batches) + 1:02d}", bundle_keys, makes))

    batch_dir = REVIEW / "batches"
    _within_private(batch_dir).mkdir(parents=True, exist_ok=True)
    index = []
    for batch_id, keys, makes in batches:
        lines = [
            f"# Eurocode review batch {batch_id}",
            "",
            f"Catalogue makes: {', '.join(sorted(makes))}",
            f"Snapshot: {snapshot_id}; demand = horizon {HORIZON} base units.",
            "",
            "## Catalogue rows (row | code | brand | description | sop/eop | scope | parser flags)",
            "",
            *_catalogue_lines(catalogue, makes),
            "",
            f"## Identities to decide ({len(keys)})",
            "",
        ]
        for key in keys:
            lines.extend(_identity_lines(key, demand[key], total))
            lines.append("")
        (batch_dir / f"{batch_id}.md").write_text("\n".join(lines), encoding="utf-8")
        index.append(
            {
                "batch": batch_id,
                "makes": sorted(makes),
                "identities": keys,
                "units": sum(demand[key]["units"] for key in keys),
            }
        )
    _write_json(
        REVIEW / "demand.json",
        {"snapshot_id": snapshot_id, "horizon": HORIZON, "total": total, "identities": demand},
    )
    _write_json(
        REVIEW / "index.json",
        {
            "snapshot_id": snapshot_id,
            "catalogue_sha256": catalogue.source_sha256,
            "identity_rules_sha256": _sha256(IDENTITY_RULES),
            "target": arguments.target,
            "selected_share": covered / total,
            "batches": index,
        },
    )
    print(
        json.dumps(
            {
                "identities": len(selected),
                "share": round(covered / total, 5),
                "batches": [(item["batch"], len(item["identities"])) for item in index],
            }
        )
    )
    return 0


def _catalogue_rows_by_code() -> dict[str, list[int]]:
    catalogue = from_document(json.loads((PRIVATE / "catalog.json").read_text(encoding="utf-8")))
    return {code: [row.row for row in rows] for code, rows in catalogue.codes().items()}


def _load_reviewer(path: Path, rows_by_code: Mapping[str, Iterable[int]]) -> EurocodeDecisions:
    document = json.loads(path.read_text(encoding="utf-8"))
    document = {
        "format": DECISIONS_FORMAT,
        "version": DECISIONS_VERSION,
        "families": document.get("families", {}),
        "identities": document.get("identities", {}),
    }
    return EurocodeDecisions.from_document(document, catalogue_codes=rows_by_code)


def _batch(batch_id: str) -> dict[str, Any]:
    index = json.loads((REVIEW / "index.json").read_text(encoding="utf-8"))
    for item in index["batches"]:
        if item["batch"] == batch_id:
            return item
    raise SystemExit(f"unknown batch {batch_id}")


def validate(arguments: argparse.Namespace) -> int:
    rows_by_code = _catalogue_rows_by_code()
    path = Path(arguments.output)
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        decisions = _load_reviewer(path, rows_by_code)
    except (KeyError, TypeError, ValueError) as error:
        print(f"INVALID: {error}")
        return 1
    batch = _batch(raw.get("batch", ""))
    expected = set(batch["identities"])
    decided = set(decisions.identities)
    problems = []
    if expected - decided:
        problems.append(f"undecided identities: {sorted(expected - decided)}")
    if decided - expected:
        problems.append(f"identities outside the batch: {sorted(decided - expected)}")
    catalogue = from_document(json.loads((PRIVATE / "catalog.json").read_text(encoding="utf-8")))
    makes_by_code = {
        code: {make for row in rows for make in row.makes}
        for code, rows in catalogue.codes().items()
    }
    batch_makes = set(batch["makes"])
    for key, family in decisions.families.items():
        for fact in family.codes:
            if not makes_by_code[fact.code] & batch_makes:
                problems.append(f"{key}: a code whose catalogue make is outside the batch")
    used = {decision.family for decision in decisions.identities.values() if decision.family}
    unused = sorted(set(decisions.families) - used)
    if unused:
        problems.append(f"families no identity uses: {unused}")
    demand = json.loads((REVIEW / "demand.json").read_text(encoding="utf-8"))["identities"]
    outside: dict[str, dict[str, int]] = {}
    for key in sorted(expected & decided):
        years: dict[str, int] = defaultdict(int)
        for geography, year, hint, units in _cells(demand[key]):
            if decisions.allocate(key, year, geography, hint).status == "outside":
                years[f"{drive_side(geography)} {year}"] += units
        if years:
            outside[key] = dict(sorted(years.items()))
    print(
        json.dumps(
            {
                "batch": raw.get("batch"),
                "families": len(decisions.families),
                "identities": len(decided),
                "problems": problems,
                "demand_outside_every_code": outside,
            },
            indent=1,
        )
    )
    return 1 if problems else 0


def _cells(entry: Mapping[str, Any]) -> Iterable[tuple[str, int, str | None, int]]:
    for cell, units in entry["cells"].items():
        geography, year, hint = cell.split("|")
        yield geography, int(year), hint or None, units


def _decision(decisions: EurocodeDecisions, key: str) -> dict[str, Any] | None:
    decision = decisions.identities.get(key)
    return None if decision is None else dataclasses.asdict(decision)


def compare(arguments: argparse.Namespace) -> int:
    rows_by_code = _catalogue_rows_by_code()
    first = _load_reviewer(Path(arguments.first), rows_by_code)
    second = _load_reviewer(Path(arguments.second), rows_by_code)
    batch_id = json.loads(Path(arguments.first).read_text())["batch"]
    batch = _batch(batch_id)
    demand = json.loads((REVIEW / "demand.json").read_text(encoding="utf-8"))["identities"]
    report = []
    agreed_units = 0
    total_units = 0
    for key in batch["identities"]:
        entry = demand[key]
        differences: dict[tuple[int, str, str], dict[str, Any]] = {}
        agreed = 0
        for geography, year, hint, units in _cells(entry):
            a = first.allocate(key, year, geography, hint)
            b = second.allocate(key, year, geography, hint)
            same = (a.codes, a.status) == (b.codes, b.status)
            if same:
                agreed += units
                continue
            slot = differences.setdefault(
                (year, drive_side(geography), hint or ""),
                {"a": [a.status, list(a.codes)], "b": [b.status, list(b.codes)], "units": 0},
            )
            slot["units"] += units
        total_units += entry["units"]
        agreed_units += agreed
        if differences:
            report.append(
                {
                    "identity": key,
                    "units": entry["units"],
                    "agreement": round(agreed / entry["units"], 4),
                    "a": _decision(first, key),
                    "b": _decision(second, key),
                    "differences": [
                        {"year": year, "drive": drive, "body": body, **value}
                        for (year, drive, body), value in sorted(differences.items())
                    ],
                }
            )
    report.sort(key=lambda item: -item["units"] * (1 - item["agreement"]))
    out = REVIEW / "compare" / f"{batch_id}.json"
    _write_json(
        out,
        {
            "batch": batch_id,
            "agreement": round(agreed_units / total_units, 4) if total_units else 1.0,
            "disagreements": report,
        },
    )
    print(
        json.dumps(
            {
                "batch": batch_id,
                "unit_agreement": round(agreed_units / total_units, 4) if total_units else 1.0,
                "identities_with_differences": len(report),
                "report": str(out.relative_to(ROOT)),
            }
        )
    )
    return 0


def _runs(values: list[tuple[int, str]]) -> str:
    """`2010:A 2011:A 2012:A+B` -> `2010-2011 A | 2012 A+B`."""

    parts: list[str] = []
    start = previous = None
    label = ""
    for year, current in values + [(10_000, "\0")]:
        if current != label or (previous is not None and year != previous + 1):
            if start is not None and label:
                span = str(start) if start == previous else f"{start}-{previous}"
                parts.append(f"{span} {label}")
            start, label = year, current
        previous = year
    return " | ".join(parts)


def timeline_lines(decisions: EurocodeDecisions, family_key: str) -> list[str]:
    """Every registration year's code set for a family, per drive side and body."""

    family = decisions.families[family_key]
    timeline = decisions.timeline(family_key)
    first = min(fact.first_year for fact in family.codes)
    years = range(first, HORIZON + 1)
    bodies = sorted({body for fact in family.codes for body in fact.bodies} - {"all"})
    lines = [f"family {family_key} ({family.label})"]
    for drive in ("LHD", "RHD"):
        for body in [None, *bodies]:
            values = [(year, "+".join(timeline.codes(year, drive, body)) or "-") for year in years]
            lines.append(f"  {drive} {body or 'unnamed body'}: {_runs(values)}")
    return lines


def preview(arguments: argparse.Namespace) -> int:
    decisions = _load_reviewer(Path(arguments.output), _catalogue_rows_by_code())
    for key in sorted(decisions.families):
        if arguments.family and not any(part in key for part in arguments.family):
            continue
        print("\n".join(timeline_lines(decisions, key)))
    return 0


def merge(arguments: argparse.Namespace) -> int:
    rows_by_code = _catalogue_rows_by_code()
    index = json.loads((REVIEW / "index.json").read_text(encoding="utf-8"))
    families: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for name in arguments.decided:
        document = json.loads(Path(name).read_text(encoding="utf-8"))
        _load_reviewer(Path(name), rows_by_code)
        for key, family in document["families"].items():
            if key in families and families[key] != family:
                raise SystemExit(f"family {key} is defined differently in two batches")
            families[key] = family
        for key, decision in document["identities"].items():
            if key in identities and identities[key] != decision:
                raise SystemExit(f"identity {key} is decided differently in two batches")
            identities[key] = decision
    decisions = {
        "format": DECISIONS_FORMAT,
        "version": DECISIONS_VERSION,
        "metadata": {
            "snapshot_id": index["snapshot_id"],
            "catalogue_sha256": index["catalogue_sha256"],
            "identity_rules_sha256": index["identity_rules_sha256"],
            "review": "two blind reviewers per batch, disagreements adjudicated",
        },
        "families": families,
        "identities": identities,
    }
    EurocodeDecisions.from_document(decisions, catalogue_codes=rows_by_code)
    _write_json(PRIVATE / "decisions.json", decisions)
    print(json.dumps({"families": len(families), "identities": len(identities)}))
    return 0


def coverage(arguments: argparse.Namespace) -> int:
    rows_by_code = _catalogue_rows_by_code()
    decisions = EurocodeDecisions.from_document(
        json.loads((PRIVATE / "decisions.json").read_text(encoding="utf-8")),
        catalogue_codes=rows_by_code,
    )
    demand = json.loads((REVIEW / "demand.json").read_text(encoding="utf-8"))
    statuses: dict[str, int] = defaultdict(int)
    for key, entry in demand["identities"].items():
        for geography, year, hint, units in _cells(entry):
            statuses[decisions.allocate(key, year, geography, hint).status] += units
    total = demand["total"]
    report = {status: round(units / total, 5) for status, units in sorted(statuses.items())}
    reviewed = 1 - statuses.get("unreviewed", 0) / total
    print(json.dumps({"reviewed_share": round(reviewed, 5), "by_status": report}, indent=1))
    return 0 if reviewed >= COVERAGE_TARGET else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--target", type=float, default=COVERAGE_TARGET)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("output")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("first")
    compare_parser.add_argument("second")
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("decided", nargs="+")
    commands.add_parser("coverage")
    preview_parser = commands.add_parser("preview")
    preview_parser.add_argument("output")
    preview_parser.add_argument("family", nargs="*", help="only families whose key contains this")
    arguments = parser.parse_args(argv)
    handler = {
        "prepare": prepare,
        "validate": validate,
        "compare": compare,
        "merge": merge,
        "coverage": coverage,
        "preview": preview,
    }[arguments.command]
    return handler(arguments)


if __name__ == "__main__":
    sys.exit(main())

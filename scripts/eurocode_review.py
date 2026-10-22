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

A later snapshot adds identities without undoing the review: ``--round <name>``
(before the command) keeps that round's batches, demand, index and outputs under
``review/rounds/<name>``, ``prepare --database <ledger> --only-undecided`` selects
only identities ``decisions.json`` does not decide, counting the decided ones
toward the target, and ``merge`` in a round adds to the existing decisions.
"""

from __future__ import annotations

import argparse
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
from icor.eurocodes.runtime import RAW_IDENTITY_PREFIX, identity_key
from icor.evidence.release_manifests import load_snapshot_manifest
from icor.infrastructure.snapshot_identity import load_identity_index

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".local" / "private" / "eurocode"
BASE_REVIEW = PRIVATE / "review"
#: The current round's directory; ``--round`` moves it under ``BASE_REVIEW/rounds``.
REVIEW = BASE_REVIEW
IDENTITY_RULES = ROOT / "data" / "vehicle_identity.json"
HORIZON = 2028
COVERAGE_TARGET = 0.99
MAX_BATCH_IDENTITIES = 70
SMALL_MAKE_IDENTITIES = 12


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
    collected: dict[str, dict[tuple[str, str], dict[tuple[str, int], int]]] = {}
    for vehicle_id, geography, year, units in rows:
        if not units:
            continue
        make, model = labels[vehicle_id]
        key = identity_key(index, vehicle_id, make, model)
        cells = collected.setdefault(key, {}).setdefault((make, model), defaultdict(int))
        cells[(geography, int(year))] += int(units)
    identities: dict[str, dict[str, Any]] = {}
    for key, by_spelling in collected.items():
        # Spellings are kept with their own cells: a reviewed variant claims the
        # raw labels its pattern matches, so demand must stay attributable to them.
        ranked = sorted(by_spelling.items(), key=lambda item: (-sum(item[1].values()), item[0]))
        spellings = [[make, model, sum(cells.values())] for (make, model), cells in ranked]
        identities[key] = {
            "units": sum(spelling[2] for spelling in spellings),
            "spellings": spellings,
            "cells": {
                f"{geography}|{year}|{position}": units
                for position, (_, cells) in enumerate(ranked)
                for (geography, year), units in sorted(cells.items())
            },
        }
    return identities


def _make_of(key: str) -> str:
    parts = key.split("|")
    return parts[1] if parts[0] == RAW_IDENTITY_PREFIX else parts[0]


def _extends(base: str, sibling: str) -> bool:
    """Whether a model name is the base name plus a qualifier (``c4 picasso``, ``ioniq5``)."""

    if not sibling.startswith(base) or len(sibling) <= len(base):
        return False
    following = sibling[len(base)]
    return following in " -._/" or base[-1].isdigit() != following.isdigit()


def lumped_labels(demand: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Per identity, markets whose labels for it probably include a sibling model.

    Some publishers label a whole model range with the bare model name. Where a
    qualified sibling (``yaris cross`` next to ``yaris``) holds a material share of
    the pair's demand elsewhere but almost none in one market, that market's bare
    labels very likely include the sibling.
    """

    by_make: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in demand:
        prefix, _, model = key.rpartition("|")
        by_make[prefix].append((model, key))
    findings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for models in by_make.values():
        for base, base_key in models:
            for sibling, sibling_key in models:
                if not _extends(base, sibling) or demand[sibling_key]["units"] < 200:
                    continue
                sibling_units: dict[str, int] = defaultdict(int)
                years: set[int] = set()
                for geography, year, _, units in _cells(demand[sibling_key]):
                    sibling_units[geography] += units
                    years.add(year)
                base_units: dict[str, int] = defaultdict(int)
                for geography, year, _, units in _cells(demand[base_key]):
                    if year in years:
                        base_units[geography] += units
                for geography, units in base_units.items():
                    if units < 50:
                        continue
                    other_sibling = sum(v for g, v in sibling_units.items() if g != geography)
                    other_base = sum(v for g, v in base_units.items() if g != geography)
                    if not other_sibling + other_base:
                        continue
                    elsewhere = other_sibling / (other_sibling + other_base)
                    here = sibling_units.get(geography, 0) / (
                        sibling_units.get(geography, 0) + units
                    )
                    if elsewhere >= 0.10 and here < elsewhere / 5:
                        findings[base_key].append(
                            {
                                "geography": geography,
                                "units": units,
                                "sibling": sibling_key,
                                "here": round(here, 3),
                                "elsewhere": round(elsewhere, 3),
                            }
                        )
    for items in findings.values():
        items.sort(key=lambda item: -item["units"] * item["elsewhere"])
    return dict(findings)


def _main_market(entry: Mapping[str, Any], position: int, units: int) -> str:
    markets: dict[str, int] = defaultdict(int)
    for cell, value in entry["cells"].items():
        geography, _, index = cell.split("|")
        if int(index) == position:
            markets[geography] += value
    geography, value = max(markets.items(), key=lambda item: item[1])
    return f"; {geography} {value / units:.0%}" if value >= 0.5 * units else ""


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


#: Codes whose rows all ended before this year never cover reviewed demand.
OLDEST_RELEVANT_END = 1980


def _catalogue_lines(catalogue: Any, makes: set[str]) -> list[str]:
    lines = []
    for code, rows in catalogue.codes().items():
        relevant = [row for row in rows if set(row.makes) & makes]
        if not relevant:
            continue
        # Only codes whose every row ended long before any reviewed demand are left
        # out. An open-ended row is still current, and a code whose rows end a few
        # years before production did (the Renault 4's 1986) can still be the right one.
        ends = [row.end for row in relevant]
        if all(end is not None for end in ends) and max(ends) < OLDEST_RELEVANT_END:
            continue
        for row in sorted(relevant, key=lambda item: item.row):
            notes = ",".join(row.flags)
            lines.append(
                f"r{row.row} | {code} | {row.brand} | {row.description} | "
                f"sop={row.sop or ''} eop={row.eop or ''} | {row.scope}"
                + (f" | {notes}" if notes else "")
            )
    return lines


def _identity_lines(
    key: str, entry: Mapping[str, Any], total: float, lumped: Iterable[Mapping[str, Any]] = ()
) -> list[str]:
    by_year: dict[int, int] = defaultdict(int)
    rhd = 0
    bodies: dict[str, int] = defaultdict(int)
    for geography, year, model, units in _cells(entry):
        by_year[year] += units
        if drive_side(geography) == "RHD":
            rhd += units
        hint = body_hint(model)
        if hint:
            bodies[hint] += units
    units = entry["units"]
    years = " ".join(
        f"{year}:{value}"
        for year, value in sorted(by_year.items())
        if value >= max(1, 0.005 * units)
    )
    # The top spellings, plus any other with a material share: a reviewer can only
    # route a lumped model to its own family if its spellings are visible.
    spellings = [
        (f"{make} / {model}", f"{value}{_main_market(entry, position, value)}")
        for position, (make, model, value) in enumerate(entry["spellings"])
        if position < 8 or (value >= 0.02 * entry["units"] and position < 25)
    ]
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
    for item in list(lumped)[:6]:
        lines.append(
            f"- labels in {item['geography']} ({item['units']} units in the sibling's years) "
            f"probably include `{item['sibling']}`: {item['here']:.0%} of the pair there "
            f"vs {item['elsewhere']:.0%} elsewhere"
        )
    return lines


def _snapshot_of(database: Path) -> str:
    """The snapshot id the manifest beside a ledger names."""

    manifest = load_snapshot_manifest(database.parent / "snapshot.json")
    return manifest.snapshot_id


def prepare(arguments: argparse.Namespace) -> int:
    if arguments.database:
        database = Path(arguments.database)
        snapshot_id = _snapshot_of(database)
    else:
        snapshot_id, database = _active_snapshot()
    catalogue_path = PRIVATE / "catalog.json"
    catalogue = from_document(json.loads(catalogue_path.read_text(encoding="utf-8")))
    demand = build_demand(database)
    decided: set[str] = set()
    if arguments.only_undecided:
        if REVIEW == BASE_REVIEW:
            raise SystemExit("--only-undecided needs --round, so the first review stays intact")
        existing = json.loads((PRIVATE / "decisions.json").read_text(encoding="utf-8"))
        decided = set(existing["identities"])
    total = sum(entry["units"] for entry in demand.values())
    ranked = sorted(demand.items(), key=lambda item: -item[1]["units"])
    selected: list[str] = []
    covered = sum(entry["units"] for key, entry in ranked if key in decided)
    for key, entry in ranked:
        if covered / total >= arguments.target:
            break
        if key in decided:
            continue
        selected.append(key)
        covered += entry["units"]

    groups = _groups(BASE_REVIEW / "make_groups.json")
    by_make: dict[str, list[str]] = defaultdict(list)
    for key in selected:
        make = _make_of(key)
        by_make[groups.get(make, make)].append(key)
    batches: list[tuple[str, list[str], set[str]]] = []
    small: list[str] = []
    splits = _splits(BASE_REVIEW / "make_groups.json")
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
    lumped = lumped_labels(demand)
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
            lines.extend(_identity_lines(key, demand[key], total, lumped.get(key, ())))
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
            "decided_before": len(decided),
            "batches": index,
        },
    )
    print(
        json.dumps(
            {
                "identities": len(selected),
                "decided_before": len(decided),
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
    used = {
        family
        for decision in decisions.identities.values()
        for family in (*decision.families, *(f for v in decision.variants for f in v.families))
    }
    unused = sorted(set(decisions.families) - used)
    if unused:
        problems.append(f"families no identity uses: {unused}")
    demand = json.loads((REVIEW / "demand.json").read_text(encoding="utf-8"))["identities"]
    outside: dict[str, dict[str, int]] = {}
    for key in sorted(expected & decided):
        years: dict[str, int] = defaultdict(int)
        for geography, year, model, units in _cells(demand[key]):
            if decisions.allocate(key, year, geography, raw_model=model).status == "outside":
                years[f"{drive_side(geography)} {year}"] += units
        if years:
            outside[key] = dict(sorted(years.items()))
    variant_units: dict[str, dict[str, int]] = {}
    for key in sorted(expected & decided):
        decision = decisions.identities[key]
        if not decision.variants:
            continue
        counts: dict[str, int] = defaultdict(int)
        for geography, year, model, units in _cells(demand[key]):
            hit = next(
                (item for item in decision.variants if item.matches(model, geography, year)), None
            )
            label = "(no variant)"
            if hit is not None:
                label = hit.pattern.pattern
                if hit.geographies:
                    label += " @" + ",".join(sorted(hit.geographies))
                if hit.years:
                    label += f" in {hit.years[0]}-{hit.years[1]}"
            counts[label] += units
        variant_units[key] = dict(counts)
    lumped_report = _lumped_report(decisions, demand, expected & decided)
    print(
        json.dumps(
            {
                "batch": raw.get("batch"),
                "families": len(decisions.families),
                "identities": len(decided),
                "problems": problems,
                "demand_outside_every_code": outside,
                "units_per_variant": variant_units,
                "lumped_labels": lumped_report,
            },
            indent=1,
        )
    )
    return 1 if problems else 0


def _lumped_report(
    decisions: EurocodeDecisions, demand: Mapping[str, Any], keys: Iterable[str]
) -> list[dict[str, Any]]:
    """Markets whose bare labels probably include a sibling the decisions do not cover.

    For each detected market, the units of the identity whose codes there miss a
    code the sibling carries in the same year (a false certainty unless the label
    is in fact specific). A sibling decided elsewhere cannot be checked here.
    """

    findings = lumped_labels(demand)
    report = []
    for key in sorted(keys):
        for item in findings.get(key, ()):
            sibling = item["sibling"]
            row = {"identity": key, **item}
            if sibling not in decisions.identities:
                report.append({**row, "uncovered_units": None, "note": "sibling not in this file"})
                continue
            sibling_label = demand[sibling]["spellings"][0][1]
            sibling_years = {year for _, year, _, _ in _cells(demand[sibling])}
            uncovered = 0
            for geography, year, model, units in _cells(demand[key]):
                if geography != item["geography"] or year not in sibling_years:
                    continue
                own = decisions.allocate(key, year, geography, raw_model=model)
                other = decisions.allocate(sibling, year, geography, raw_model=sibling_label)
                if other.codes and not set(other.codes) <= set(own.codes):
                    uncovered += units
            if uncovered:
                report.append({**row, "uncovered_units": uncovered})
    return report


def _cells(entry: Mapping[str, Any]) -> Iterable[tuple[str, int, str, int]]:
    """Each demand cell as (geography, registration year, raw model label, units)."""

    spellings = entry["spellings"]
    for cell, units in entry["cells"].items():
        geography, year, position = cell.split("|")
        yield geography, int(year), spellings[int(position)][1], units


def _decision(decisions: EurocodeDecisions, key: str) -> dict[str, Any] | None:
    decision = decisions.identities.get(key)
    if decision is None:
        return None
    return {
        "status": decision.status,
        "family": list(decision.families),
        "body": decision.body,
        "variants": [
            {
                "match": variant.pattern.pattern,
                "status": variant.status,
                "family": list(variant.families),
                "body": variant.body,
            }
            for variant in decision.variants
        ],
    }


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
        for geography, year, model, units in _cells(entry):
            a = first.allocate(key, year, geography, raw_model=model)
            b = second.allocate(key, year, geography, raw_model=model)
            if (a.codes, a.status, a.family, a.uncoded) == (b.codes, b.status, b.family, b.uncoded):
                agreed += units
                continue
            slot = differences.setdefault(
                (year, drive_side(geography), body_hint(model) or "", a, b),
                {
                    "a": [a.status, [*a.codes, *("?" * a.uncoded)]],
                    "b": [b.status, [*b.codes, *("?" * b.uncoded)]],
                    "units": 0,
                },
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
                        for (year, drive, body, _, _), value in sorted(
                            differences.items(), key=lambda item: item[0][:3]
                        )
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
    """Every registration year's code set for a family, per drive side and body.

    ``?`` marks a year in which a registration may be a vehicle without a code.
    """

    family = decisions.families[family_key]
    timeline = decisions.timeline(family_key)
    first = min(fact.first_year for fact in family.codes)
    years = range(first, HORIZON + 1)
    bodies = sorted({body for fact in family.codes for body in fact.bodies} - {"all"})
    lines = [f"family {family_key} ({family.label})"]
    for drive in ("LHD", "RHD"):
        for body in [None, *bodies]:
            values = []
            for year in years:
                codes = timeline.codes(year, drive, body)
                if codes and timeline.shared_with_uncoded(year, drive, codes):
                    codes = (*codes, "?")
                values.append((year, "+".join(codes) or "-"))
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
    previous: dict[str, Any] | None = None
    if REVIEW != BASE_REVIEW:
        # A later round adds to what was decided; it never silently redefines it.
        previous = json.loads((PRIVATE / "decisions.json").read_text(encoding="utf-8"))
        families.update(previous["families"])
        identities.update(previous["identities"])
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
            **(
                {"rounds": [*previous["metadata"].get("rounds", []), REVIEW.name]}
                if previous is not None
                else {}
            ),
        },
        "families": families,
        "identities": identities,
    }
    EurocodeDecisions.from_document(decisions, catalogue_codes=rows_by_code)
    if previous is not None:
        if previous["metadata"]["catalogue_sha256"] != index["catalogue_sha256"]:
            raise SystemExit("the round was prepared against another catalogue")
        _write_json(PRIVATE / f"decisions.before-{REVIEW.name}.json", previous)
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
    uncoded = 0
    for key, entry in demand["identities"].items():
        for geography, year, model, units in _cells(entry):
            allocation = decisions.allocate(key, year, geography, raw_model=model)
            statuses[allocation.status] += units
            uncoded += units if allocation.uncoded else 0
    total = demand["total"]
    report = {status: round(units / total, 5) for status, units in sorted(statuses.items())}
    reviewed = 1 - statuses.get("unreviewed", 0) / total
    print(
        json.dumps(
            {
                "reviewed_share": round(reviewed, 5),
                "by_status": report,
                "shared_with_uncoded": round(uncoded / total, 5),
            },
            indent=1,
        )
    )
    return 0 if reviewed >= COVERAGE_TARGET else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", help="work in review/rounds/<name> instead of review/")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--target", type=float, default=COVERAGE_TARGET)
    prepare_parser.add_argument("--database", help="a candidate's evidence.sqlite3")
    prepare_parser.add_argument("--only-undecided", action="store_true")
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
    global REVIEW
    REVIEW = BASE_REVIEW
    if arguments.round:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", arguments.round):
            raise SystemExit("a round name is lower-case letters, digits and hyphens")
        REVIEW = BASE_REVIEW / "rounds" / arguments.round
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

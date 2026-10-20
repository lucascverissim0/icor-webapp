#!/usr/bin/env python3
"""Parse ICOR's Eurocode catalogue workbook into its private JSON document.

The workbook is ICOR's proprietary data and the repository is public, so input
and output must both live under the git-ignored ``.local/private/eurocode``; the
script refuses any other output directory. It prints counts only, never a code
or a catalogue description.

    uv run python scripts/ingest_eurocode_catalog.py

writes ``catalog.json`` (deterministic: the same workbook gives the same bytes)
and ``anomalies.csv`` (every rejected row and every flagged row, for review).
The workbook's sha256 must match the private ``manifest.json`` when one exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from icor.eurocodes.catalog import file_sha256, read_workbook, to_document

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".local" / "private" / "eurocode"


def _within_private(path: Path) -> bool:
    try:
        path.resolve().relative_to(PRIVATE.resolve())
    except ValueError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--aliases", type=Path, default=PRIVATE / "make_aliases.json")
    parser.add_argument("--out-dir", type=Path, default=PRIVATE)
    arguments = parser.parse_args(argv)

    manifest_path = PRIVATE / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    )
    workbook = arguments.workbook or (PRIVATE / manifest["artifact"] if manifest else None)
    if workbook is None or not workbook.exists():
        print("Eurocode workbook not found; pass --workbook.", file=sys.stderr)
        return 2
    if not _within_private(arguments.out_dir):
        print(f"Refusing to write outside {PRIVATE.relative_to(ROOT)}.", file=sys.stderr)
        return 2
    sha256 = file_sha256(workbook)
    if manifest is not None and manifest.get("sha256") != sha256:
        print("Workbook sha256 differs from the private manifest.", file=sys.stderr)
        return 2

    aliases = json.loads(arguments.aliases.read_text(encoding="utf-8"))["aliases"]
    catalogue = read_workbook(workbook, make_aliases=aliases)
    document = to_document(catalogue)
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    (arguments.out_dir / "catalog.json").write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with (arguments.out_dir / "anomalies.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "status", "detail", "code", "brand", "description", "sop", "eop"])
        for rejected in catalogue.rejected:
            code, brand, description, eop, sop = (rejected.values + ("",) * 5)[:5]
            writer.writerow(
                [rejected.row, "rejected", rejected.reason, code, brand, description, sop, eop]
            )
        for row in catalogue.rows:
            if row.flags:
                writer.writerow(
                    [
                        row.row,
                        "flagged",
                        ";".join(row.flags),
                        row.code,
                        row.brand,
                        row.description,
                        row.sop,
                        row.eop,
                    ]
                )

    flags = Counter(flag for row in catalogue.rows for flag in row.flags)
    summary = {
        "source_rows": catalogue.source_rows,
        "parsed": len(catalogue.rows),
        "rejected": dict(sorted(Counter(row.reason for row in catalogue.rejected).items())),
        "codes": len(catalogue.codes()),
        "scopes": dict(sorted(Counter(row.scope for row in catalogue.rows).items())),
        "drive": dict(sorted(Counter(str(row.drive) for row in catalogue.rows).items())),
        "flags": dict(sorted(flags.items())),
        "sha256_matches_manifest": manifest is not None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if len(catalogue.rows) + len(catalogue.rejected) == catalogue.source_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

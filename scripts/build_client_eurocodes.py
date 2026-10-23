#!/usr/bin/env python3
"""Write the Eurocode map the client release image carries.

The private folder holds the source workbook, review batches and reports that
name real codes; none of it belongs in an image. This writes only what the
running app loads, into the git-ignored ``.local/client-eurocode``:

- ``catalog.json``, trimmed to codes, row numbers and the source hash
  (``runtime_document``), so no catalogue description, make or rejected source
  row is copied;
- ``decisions.json``, with every family, and every reviewed identity the client
  snapshot forecasts, re-serialised from the fields the loader reads
  (``icor.eurocodes.client_map``) so a note or rationale a reviewer added never
  ships. An identity the snapshot does not forecast carries no units; kept, it
  would only show clients a "no longer exist" warning about review work.

It refuses to write a map that does not serve exactly what the private one
does, and, given the client evidence root, one that was not reviewed for the
snapshot that root serves, and it refuses a destination that is, holds or lies
inside the private folder, because it replaces the destination. It prints
counts only.

    uv run python scripts/build_client_eurocodes.py --client-root .local/client-evidence
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from icor.application.client_snapshot import reviewed_snapshot_id
from icor.eurocodes.allocation import IdentityDecision
from icor.eurocodes.catalog import from_document, runtime_document
from icor.eurocodes.client_map import client_decisions
from icor.eurocodes.runtime import EurocodeMap
from icor.infrastructure.snapshot_eurocodes import ranked_identities
from icor.infrastructure.snapshot_store import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".local" / "private"
SOURCE = PRIVATE / "eurocode"
DESTINATION = ROOT / ".local" / "client-eurocode"
FILES = ("catalog.json", "decisions.json")


class ClientEurocodeError(RuntimeError):
    """A client map that must not be packaged."""


def _check_destination(source: Path, destination: Path) -> None:
    """Refuse a destination the build would replace that holds the private data.

    The destination is deleted to make room for the map, so one that is, holds or
    lies inside the source, or the private folder, would destroy review work.
    """

    source, destination, private = source.resolve(), destination.resolve(), PRIVATE.resolve()
    for folder, name in ((source, "source"), (private, "private")):
        if destination == folder:
            reason = f"is the {name} folder"
        elif folder in destination.parents:
            reason = f"is inside the {name} folder"
        elif destination in folder.parents:
            reason = f"contains the {name} folder"
        else:
            continue
        raise SystemExit(f"client Eurocodes not written: the destination {reason}")


def _without_rationale(decision: IdentityDecision) -> IdentityDecision:
    """The decision as the client map carries it: its reviewers' explanations are not shipped."""

    variants = tuple(replace(variant, why="") for variant in decision.variants)
    return replace(decision, why="", variants=variants)


def _ranked(client_root: Path, private: EurocodeMap) -> frozenset[str]:
    manifest, ledger = SnapshotStore(client_root).open_active_snapshot()
    if reviewed_snapshot_id(manifest, ledger.path) != private.decisions.metadata.get("snapshot_id"):
        raise ClientEurocodeError(
            "the Eurocode decisions were reviewed for another snapshot than the client serves"
        )
    connection = sqlite3.connect(f"{Path(ledger.path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return ranked_identities(connection)
    finally:
        connection.close()


def build(
    source: Path,
    destination: Path,
    *,
    client_root: Path | None = None,
    ranked: frozenset[str] | None = None,
) -> dict[str, object]:
    _check_destination(source, destination)
    private = EurocodeMap.from_directory(source)
    if client_root is not None:
        ranked = _ranked(client_root, private)
    document = client_decisions(
        json.loads((source / "decisions.json").read_text(encoding="utf-8")), identities=ranked
    )
    catalogue = from_document(json.loads((source / "catalog.json").read_text(encoding="utf-8")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".client-eurocode-"))
    try:
        (staged / "catalog.json").write_text(
            json.dumps(runtime_document(catalogue), sort_keys=True) + "\n", encoding="utf-8"
        )
        (staged / "decisions.json").write_text(
            json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        packaged = EurocodeMap.from_directory(staged)
        if packaged.summaries() != private.summaries():
            raise ClientEurocodeError("the trimmed map does not describe the same codes")
        kept = packaged.decisions.identities
        if any(
            kept[key] != _without_rationale(private.decisions.identities[key]) for key in kept
        ) or (
            ranked is not None
            and {key for key in private.decisions.identities if key in ranked} != kept.keys()
        ):
            raise ClientEurocodeError("the trimmed map does not review the ranked identities alike")
        # Swap rather than delete first: if the new map cannot be put in place, the
        # previous one is put back.
        retired = staged.with_name(f"{staged.name}-retired")
        if destination.exists():
            destination.rename(retired)
        try:
            staged.rename(destination)
        except BaseException:
            if retired.exists():
                retired.rename(destination)
            raise
        if retired.exists():
            shutil.rmtree(retired)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return {
        "codes": len(packaged.summaries()),
        "reviewed_identities": len(packaged.decisions.identities),
        "unranked_identities_left_out": len(private.decisions.identities)
        - len(packaged.decisions.identities),
        "files": sorted(path.name for path in destination.iterdir()),
        "reviewed_for_client_snapshot": client_root is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument(
        "--client-root",
        type=Path,
        help="the client evidence root whose active snapshot the map must have been reviewed for",
    )
    arguments = parser.parse_args(argv)
    try:
        result = build(arguments.source, arguments.destination, client_root=arguments.client_root)
    except (ClientEurocodeError, OSError, ValueError, KeyError) as error:
        print(f"client Eurocodes not written: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

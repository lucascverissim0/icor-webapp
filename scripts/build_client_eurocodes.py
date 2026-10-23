#!/usr/bin/env python3
"""Write the Eurocode map the client release image carries.

The private folder holds the source workbook, review batches and reports that
name real codes; none of it belongs in an image. This writes only what the
running app loads, into the git-ignored ``.local/client-eurocode``:

- ``catalog.json``, trimmed to codes, row numbers and the source hash
  (``runtime_document``), so no catalogue description, make or rejected source
  row is copied;
- ``decisions.json``, byte for byte, because allocation reads all of it.

It refuses to write a map that does not serve exactly what the private one
does, and, given the client evidence root, one that was not reviewed for the
snapshot that root serves. It prints counts only.

    uv run python scripts/build_client_eurocodes.py --client-root .local/client-evidence
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from icor.application.client_snapshot import reviewed_snapshot_id
from icor.eurocodes.catalog import from_document, runtime_document
from icor.eurocodes.runtime import EurocodeMap
from icor.infrastructure.snapshot_store import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".local" / "private" / "eurocode"
DESTINATION = ROOT / ".local" / "client-eurocode"
FILES = ("catalog.json", "decisions.json")


class ClientEurocodeError(RuntimeError):
    """A client map that must not be packaged."""


def build(source: Path, destination: Path, *, client_root: Path | None = None) -> dict[str, object]:
    private = EurocodeMap.from_directory(source)
    if client_root is not None:
        manifest, ledger = SnapshotStore(client_root).open_active_snapshot()
        reviewed_for = private.decisions.metadata.get("snapshot_id")
        if reviewed_snapshot_id(manifest, ledger.path) != reviewed_for:
            raise ClientEurocodeError(
                "the Eurocode decisions were reviewed for another snapshot than the client serves"
            )
    catalogue = from_document(json.loads((source / "catalog.json").read_text(encoding="utf-8")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".client-eurocode-"))
    try:
        (staged / "catalog.json").write_text(
            json.dumps(runtime_document(catalogue), sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copyfile(source / "decisions.json", staged / "decisions.json")
        packaged = EurocodeMap.from_directory(staged)
        if packaged.summaries() != private.summaries():
            raise ClientEurocodeError("the trimmed map does not describe the same codes")
        if packaged.decisions.identities.keys() != private.decisions.identities.keys():
            raise ClientEurocodeError("the trimmed map does not review the same identities")
        if destination.exists():
            shutil.rmtree(destination)
        staged.rename(destination)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return {
        "codes": len(packaged.summaries()),
        "reviewed_identities": len(packaged.decisions.identities),
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

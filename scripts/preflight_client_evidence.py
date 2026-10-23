"""Prove the baked evidence root will pass the image build, before uploading it.

`Dockerfile:58` copies `.local/client-evidence` to `/srv/icor/evidence` and then
asserts, at build time, that the snapshot there opens and is client-scoped. That
assertion is the right place for it — a bad bake becomes a red build rather than
a crash-looping machine — but it is an expensive place to *discover* a mistake:
the build context has to be archived and uploaded to a remote builder first.

This reproduces the same checks locally in about a minute, so the shape of the
root is settled before any of that. Run it after promoting the derived client
snapshot into its own root, and before `fly deploy`.

    uv run python scripts/preflight_client_evidence.py .local/client-evidence

Exit 0 and one JSON object means the root is ready. Anything else names what the
image build would have failed on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from icor.domain.snapshots import CLIENT_RELEASE_SCOPE  # noqa: E402
from icor.infrastructure.snapshot_store import SnapshotStore  # noqa: E402

#: What `_verify_snapshot_directory` allows a snapshot directory to contain. A
#: stray file there fails the build, so it is checked here rather than remotely.
SNAPSHOT_FILES = frozenset({"evidence.sqlite3", "snapshot.json", "validation.json"})


def _failures(root: Path) -> list[str]:
    problems: list[str] = []
    if not root.is_dir():
        return [f"{root} does not exist; Dockerfile:58 copies it unconditionally"]
    pointer = root / "active.json"
    if not pointer.is_file():
        problems.append(
            f"{pointer} is missing: this path must be a whole evidence root, "
            "not a snapshot directory"
        )
    if (root / "candidates").exists():
        problems.append(
            "candidates/ is present and would put a second copy of a "
            "two-gigabyte database into the build context"
        )
    if (root / "releases").exists():
        problems.append("releases/ is present and is not needed in an image")
    if (root / "verified.json").exists():
        problems.append(
            "verified.json is already present; the image writes it, and a marker "
            "naming another snapshot makes a trusted open refuse"
        )
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        problems.append(f"{snapshots} is missing")
        return problems
    for directory in sorted(p for p in snapshots.iterdir() if p.is_dir()):
        extra = {p.name for p in directory.iterdir()} - SNAPSHOT_FILES
        if extra:
            problems.append(
                f"{directory.name} holds files the store refuses: {', '.join(sorted(extra))}"
            )
    return problems


def _eurocode_failures(directory: Path, manifest: object, database: Path) -> list[str]:
    """The image's build-time Eurocode assertions, plus what the folder may hold."""

    from icor.application.client_snapshot import reviewed_snapshot_id
    from icor.eurocodes.runtime import EurocodeMap
    from icor.evidence.vehicle_identity import identity_rules_sha256

    if not directory.is_dir():
        return ["the client Eurocode map is missing; run scripts/build_client_eurocodes.py"]
    names = sorted(path.name for path in directory.iterdir())
    if names != ["catalog.json", "decisions.json"]:
        return [f"the client Eurocode folder must hold exactly the two map files, not {names}"]
    catalogue = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))
    if any(row.get("description") or row.get("makes") for row in catalogue.get("rows", ())):
        return ["the client catalogue is not trimmed; it would carry catalogue text"]
    try:
        metadata = EurocodeMap.from_directory(directory).decisions.metadata
    except (OSError, ValueError, KeyError) as error:
        return [f"the client Eurocode map does not load ({type(error).__name__})"]
    problems = []
    if metadata.get("snapshot_id") != reviewed_snapshot_id(manifest, database):  # type: ignore[arg-type]
        problems.append("the Eurocode decisions were reviewed for another snapshot")
    if metadata.get("identity_rules_sha256") != identity_rules_sha256():
        problems.append("the Eurocode decisions were reviewed under other identity rules")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="the evidence root the image bakes")
    parser.add_argument(
        "--eurocode-dir",
        type=Path,
        default=REPO / ".local" / "client-eurocode",
        help="the trimmed Eurocode map the image bakes beside it",
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()

    problems = _failures(root)
    if problems:
        print(json.dumps({"ready": False, "problems": problems}, indent=2))
        return 1

    # The same call the build-time assertion makes, and the same comparison.
    manifest, ledger = SnapshotStore(root).open_active_snapshot()
    database = Path(ledger.path)
    report = {
        "ready": manifest.scope == CLIENT_RELEASE_SCOPE,
        "root": str(root),
        "snapshot_id": manifest.snapshot_id,
        "scope": manifest.scope,
        "expected_scope": CLIENT_RELEASE_SCOPE,
        "database_bytes": database.stat().st_size,
        "observation_count": manifest.observation_count,
    }
    problems = []
    if not report["ready"]:
        problems.append(
            f"scope is {manifest.scope!r}; the image asserts "
            f"{CLIENT_RELEASE_SCOPE!r} and would fail the build"
        )
    problems += _eurocode_failures(arguments.eurocode_dir.resolve(), manifest, database)
    report["eurocode_dir"] = str(arguments.eurocode_dir.resolve())
    if problems:
        report["ready"] = False
        report["problems"] = problems
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

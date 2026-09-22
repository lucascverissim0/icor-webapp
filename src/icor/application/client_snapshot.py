"""Derive a client-release snapshot from a finished full snapshot.

The client preview exposes Opportunities, Model search and evidence freshness
only. Tracing those routes shows they read about a fifth of the database: the
raw observation corpus, its identity and assignment lineage and the label-level
registration projection are never touched. On the promoted snapshot that is
7.3 GB of 9.25 GB, and it is also most of the cost of verifying the database at
start-up.

This derives the smaller artifact from the finished one rather than building it
again. A second canonical build would repeat the four-and-a-quarter-hour replay
for a result whose rows are already known to be identical; copy, prune and
vacuum takes minutes. Nothing is hand-edited: the manifest and validation report
are produced by the same writer and the same validator as any other snapshot, so
the integrity checks apply to it unchanged.

Rows are removed, never tables. The validator requires the tables to exist, and
an absent table would also hide the difference between "this snapshot has no
observations" and "this snapshot was never given any".
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

from icor.domain.snapshots import CLIENT_RELEASE_SCOPE, SnapshotManifest, SnapshotStatus
from icor.evidence.serialization import canonical_json_bytes, sha256_file
from icor.evidence.validation import SnapshotValidator
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

# Emptied in this order so a row is never orphaned from the row it references
# while the transaction is open.
PRUNED_TABLES: tuple[str, ...] = (
    "cohort_input",
    "generation_assignment",
    "identity_mapping",
    "registration_label_aggregate",
    "observation",
)

# Read by at least one client-release route, so they must survive with rows.
REQUIRED_TABLES: tuple[str, ...] = (
    "opportunity_estimate",
    "opportunity_input",
    "opportunity_cohort_attribution",
    "cohort_estimate",
    "canonical_vehicle",
    "generation_entry",
    "registration_family_aggregate",
    "source_release",
    "completeness_record",
)


class ClientSnapshotError(RuntimeError):
    """The client-release snapshot cannot be derived safely."""


def prune_to_client_scope(database_path: Path) -> dict[str, int]:
    """Empty the tables no client-release route reads. Returns rows removed."""

    removed: dict[str, int] = {}
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        with connection:
            for table in PRUNED_TABLES:
                before = connection.execute(
                    f"SELECT COUNT(*) FROM {table}"  # noqa: S608 - fixed table names
                ).fetchone()[0]
                connection.execute(f"DELETE FROM {table}")  # noqa: S608
                removed[table] = int(before)
        connection.execute("VACUUM")
    finally:
        connection.close()
    return removed


def assert_client_scope_is_servable(database_path: Path) -> None:
    """Refuse an artifact that pruned something a client route still needs."""

    connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
    try:
        present = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing = [table for table in (*PRUNED_TABLES, *REQUIRED_TABLES) if table not in present]
        if missing:
            raise ClientSnapshotError(
                f"client snapshot is missing tables: {', '.join(sorted(missing))}"
            )
        empty = [
            table
            for table in REQUIRED_TABLES
            if table not in {"completeness_record", "source_release"}
            and connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0  # noqa: S608
        ]
        if empty:
            raise ClientSnapshotError(
                f"client snapshot served tables are empty: {', '.join(sorted(empty))}"
            )
    finally:
        connection.close()


def derive_client_snapshot(
    source: Path,
    destination: Path,
    *,
    validator: SnapshotValidator | None = None,
) -> SnapshotManifest:
    """Copy a built snapshot, prune it, and re-issue its manifest canonically.

    The identity is derived exactly as the builder derives one, from the release
    artifact hashes recorded in the snapshot itself and the client scope, because
    promotion recomputes it from the same table and would otherwise reject the
    artifact.
    """

    source_manifest_path = source / "snapshot.json"
    source_database = source / "evidence.sqlite3"
    if not source_manifest_path.is_file() or not source_database.is_file():
        raise ClientSnapshotError("source snapshot is incomplete")
    if destination.exists():
        raise ClientSnapshotError("client snapshot destination already exists")

    from icor.evidence.release_manifests import load_snapshot_manifest

    manifest = load_snapshot_manifest(source_manifest_path)
    if manifest.scope == CLIENT_RELEASE_SCOPE:
        raise ClientSnapshotError("source snapshot is already client scoped")

    destination.mkdir(parents=True)
    database_path = destination / "evidence.sqlite3"
    shutil.copy2(source_database, database_path)
    database_path.chmod(0o644)
    prune_to_client_scope(database_path)
    assert_client_scope_is_servable(database_path)

    repository = SQLiteEvidenceRepository(database_path)

    from icor.application.snapshot_build import snapshot_id_for

    # Promotion recomputes identity from `source_release` in the database, so
    # the hashes must come from exactly there and nowhere else.
    releases_by_id = {item.release_id: item for item in repository.list_releases()}
    snapshot_id = snapshot_id_for(
        build_as_of=manifest.built_at,
        deterministic_seed=manifest.deterministic_seed,
        versions=manifest.versions,
        release_artifact_hashes=tuple(
            (release_id, releases_by_id[release_id].sha256)
            for release_id in manifest.release_ids
        ),
        scope=CLIENT_RELEASE_SCOPE,
    )
    derived = replace(
        manifest,
        snapshot_id=snapshot_id,
        status=SnapshotStatus.CANDIDATE,
        scope=CLIENT_RELEASE_SCOPE,
        database_sha256=sha256_file(database_path),
        observation_count=len(repository.list_observations()),
        published_value_count=len(repository.list_published_values()),
        warnings=(),
    )
    report = (validator or SnapshotValidator()).validate(repository, derived)
    warnings = tuple(
        finding.code for finding in report.findings if finding.severity.name == "WARNING"
    )
    if warnings:
        derived = replace(derived, warnings=warnings)
        report = (validator or SnapshotValidator()).validate(repository, derived)
    (destination / "snapshot.json").write_bytes(canonical_json_bytes(derived))
    (destination / "validation.json").write_bytes(canonical_json_bytes(report))
    return derived

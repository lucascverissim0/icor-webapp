from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from icor.application.client_snapshot import (
    PRUNED_TABLES,
    REQUIRED_TABLES,
    ClientSnapshotError,
    assert_client_scope_is_servable,
    prune_to_client_scope,
)
from icor.application.snapshot_build import snapshot_id_for
from icor.domain.snapshots import (
    CLIENT_RELEASE_SCOPE,
    FULL_SCOPE,
    SnapshotManifest,
    SnapshotStatus,
    SnapshotVersions,
)
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

BUILD_AS_OF = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _versions() -> SnapshotVersions:
    return SnapshotVersions(
        source_registry="sources-v1",
        identity_registry="identity-v1",
        reconciliation_method="single-coverage-corroboration-v2",
        confidence_method="confidence-v1",
        estimation_method="interpolation-v1",
        survival_method="survival-v1",
        hazard_method="hazard-v1",
        forecast_method="forecast-v1",
    )


def _identity(scope: str) -> str:
    return snapshot_id_for(
        build_as_of=BUILD_AS_OF,
        deterministic_seed=20260922,
        versions=_versions(),
        release_artifact_hashes=(("eea-2024", "a" * 64),),
        scope=scope,
    )


def test_adding_the_scope_field_did_not_change_any_existing_snapshot_id() -> None:
    """The regression guard on identity.

    Every promoted snapshot was minted before `scope` existed. If a full-scope
    build started hashing the new key, every one of them would fail promotion
    verification.
    """

    without_scope = snapshot_id_for(
        build_as_of=BUILD_AS_OF,
        deterministic_seed=20260922,
        versions=_versions(),
        release_artifact_hashes=(("eea-2024", "a" * 64),),
    )

    assert _identity(FULL_SCOPE) == without_scope


def test_a_client_scoped_snapshot_takes_a_distinct_identity() -> None:
    """Same inputs, different contents, so it must not claim the same id."""

    assert _identity(CLIENT_RELEASE_SCOPE) != _identity(FULL_SCOPE)


def test_a_manifest_written_before_scope_existed_still_loads_as_full(
    tmp_path: Path,
) -> None:
    from icor.evidence.release_manifests import load_snapshot_manifest
    from icor.evidence.serialization import canonical_json_bytes

    manifest = SnapshotManifest(
        snapshot_id="snapshot-legacy-v1",
        status=SnapshotStatus.ACTIVE,
        built_at=BUILD_AS_OF,
        deterministic_seed=20260922,
        release_ids=("eea-2024",),
        versions=_versions(),
        database_sha256="b" * 64,
        observation_count=1,
        published_value_count=0,
        warnings=(),
    )
    payload = canonical_json_bytes(manifest).decode("utf-8")
    assert '"scope"' in payload
    legacy = payload.replace(f'"scope":"{FULL_SCOPE}",', "").replace(
        f',"scope":"{FULL_SCOPE}"', ""
    )
    path = tmp_path / "snapshot.json"
    path.write_text(legacy, encoding="utf-8")

    loaded = load_snapshot_manifest(path)

    assert loaded.scope == FULL_SCOPE


def test_an_unsupported_scope_is_refused() -> None:
    with pytest.raises(ValueError, match="scope is unsupported"):
        SnapshotManifest(
            snapshot_id="snapshot-bad-scope",
            status=SnapshotStatus.CANDIDATE,
            built_at=BUILD_AS_OF,
            deterministic_seed=1,
            release_ids=("eea-2024",),
            versions=_versions(),
            database_sha256="c" * 64,
            observation_count=0,
            published_value_count=0,
            warnings=(),
            scope="whatever-i-like",
        )


def _schema(tmp_path: Path) -> Path:
    path = tmp_path / "evidence.sqlite3"
    SQLiteEvidenceRepository(path, writable=True)
    return path


def test_pruning_empties_exactly_the_declared_tables(tmp_path: Path) -> None:
    path = _schema(tmp_path)

    removed = prune_to_client_scope(path)

    assert set(removed) == set(PRUNED_TABLES)


def test_pruning_keeps_every_table_present(tmp_path: Path) -> None:
    """The validator errors on an absent table, so rows go and tables stay."""

    path = _schema(tmp_path)
    prune_to_client_scope(path)

    with sqlite3.connect(path) as connection:
        present = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    for table in (*PRUNED_TABLES, *REQUIRED_TABLES):
        assert table in present


def test_a_pruned_snapshot_missing_served_rows_is_refused(tmp_path: Path) -> None:
    """An empty artifact must not be mistaken for a servable one."""

    path = _schema(tmp_path)
    prune_to_client_scope(path)

    with pytest.raises(ClientSnapshotError, match="served tables are empty"):
        assert_client_scope_is_servable(path)

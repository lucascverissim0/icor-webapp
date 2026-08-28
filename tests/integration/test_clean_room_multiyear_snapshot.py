from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from icor.application.snapshot_build import snapshot_id_for
from icor.domain.snapshots import SnapshotVersions


def test_snapshot_identity_changes_for_time_seed_method_or_artifact() -> None:
    from datetime import UTC, datetime

    built_at = datetime(2026, 8, 28, 10, 45, tzinfo=UTC)
    versions = SnapshotVersions(*("v1",) * 10)
    artifacts = (("release-a", "a" * 64),)
    baseline = snapshot_id_for(
        build_as_of=built_at,
        deterministic_seed=20260827,
        versions=versions,
        release_artifact_hashes=artifacts,
    )

    assert baseline != snapshot_id_for(
        build_as_of=built_at + timedelta(seconds=1),
        deterministic_seed=20260827,
        versions=versions,
        release_artifact_hashes=artifacts,
    )
    assert baseline != snapshot_id_for(
        build_as_of=built_at,
        deterministic_seed=20260828,
        versions=versions,
        release_artifact_hashes=artifacts,
    )
    assert baseline != snapshot_id_for(
        build_as_of=built_at,
        deterministic_seed=20260827,
        versions=replace(versions, forecast_method="v2"),
        release_artifact_hashes=artifacts,
    )
    assert baseline != snapshot_id_for(
        build_as_of=built_at,
        deterministic_seed=20260827,
        versions=versions,
        release_artifact_hashes=(("release-a", "b" * 64),),
    )

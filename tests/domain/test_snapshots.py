from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from icor.domain.snapshots import SnapshotManifest, SnapshotStatus, SnapshotVersions


def make_versions(**overrides: object) -> SnapshotVersions:
    values: dict[str, object] = {
        "source_registry": "source-registry-v1",
        "identity_registry": "identity-registry-v1",
        "reconciliation_method": "reconciliation-v1",
        "confidence_method": "confidence-v1",
        "estimation_method": "estimation-v1",
        "survival_method": "survival-v1",
        "hazard_method": "hazard-v1",
        "forecast_method": "forecast-v1",
    }
    values.update(overrides)
    return SnapshotVersions(**values)  # type: ignore[arg-type]


def make_snapshot(**overrides: object) -> SnapshotManifest:
    values: dict[str, object] = {
        "snapshot_id": "snapshot-20260826-abc123",
        "status": SnapshotStatus.CANDIDATE,
        "built_at": datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        "deterministic_seed": 42,
        "release_ids": ("eea-2024-20260826", "kba-2024-20260826"),
        "versions": make_versions(),
        "database_sha256": "b" * 64,
        "observation_count": 10,
        "published_value_count": 3,
        "warnings": (),
    }
    values.update(overrides)
    return SnapshotManifest(**values)  # type: ignore[arg-type]


def test_snapshot_manifest_requires_utc_timestamp_and_lowercase_sha256() -> None:
    with pytest.raises(ValueError, match="UTC"):
        make_snapshot(built_at=datetime(2026, 8, 26, 10, 0))
    with pytest.raises(ValueError, match="SHA-256"):
        make_snapshot(database_sha256="B" * 64)


def test_snapshot_manifest_requires_sorted_unique_release_ids() -> None:
    with pytest.raises(ValueError, match="sorted"):
        make_snapshot(release_ids=("kba-2024-20260826", "eea-2024-20260826"))
    with pytest.raises(ValueError, match="unique"):
        make_snapshot(release_ids=("eea-2024-20260826", "eea-2024-20260826"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"deterministic_seed": -1},
        {"observation_count": -1},
        {"published_value_count": -1},
        {"deterministic_seed": 1.0},
    ],
)
def test_snapshot_manifest_rejects_non_negative_integer_fields(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_snapshot(**overrides)


def test_snapshot_manifest_requires_stable_identifier_and_nonblank_warnings() -> None:
    with pytest.raises(ValueError, match="identifier"):
        make_snapshot(snapshot_id="snapshot/unsafe")
    with pytest.raises(ValueError, match="warning"):
        make_snapshot(warnings=("",))


def test_snapshot_versions_require_every_reproducibility_version() -> None:
    with pytest.raises(ValueError, match="version"):
        make_versions(forecast_method="")


def test_snapshot_contracts_are_frozen_and_revalidate_on_replace() -> None:
    snapshot = make_snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.observation_count = 99  # type: ignore[misc]
    with pytest.raises(ValueError, match="UTC"):
        replace(
            snapshot,
            built_at=datetime(2026, 8, 26, 10, 0, tzinfo=timezone(timedelta(hours=1))),
        )

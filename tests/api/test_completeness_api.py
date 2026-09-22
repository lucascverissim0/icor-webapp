from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from icor.api.app import create_app
from icor.domain.snapshots import SnapshotVersions

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])


class Completeness:
    def report(self):
        return SimpleNamespace(
            snapshot_id="snapshot-real-v1",
            built_at=datetime(2026, 8, 28, tzinfo=UTC),
            versions=SnapshotVersions(
                source_registry="sources-v1",
                identity_registry="identity-v1",
                reconciliation_method="reconcile-v1",
                confidence_method="confidence-v1",
                estimation_method="estimate-v1",
                survival_method="survival-v1",
                hazard_method="hazard-v1",
                forecast_method="forecast-v1",
                generation_registry="registry-v1",
                generation_resolver="resolver-v1",
            ),
            items=(
                SimpleNamespace(
                    completeness_id="completeness-de-2023",
                    geography="DE", year=2023, release_count=1,
                    observation_count=10, usable_observation_count=9,
                    assigned_observation_count=9, canonical_family_count=4,
                    sourced_generation_count=0, estimated_generation_count=4,
                    forecastable_count=7, evidence_only_count=2,
                    rejected_record_count=1, reason_codes=("annual-completeness",),
                ),
            ),
        )


def test_completeness_route_exposes_snapshot_versions_and_counts() -> None:
    with TestClient(create_app(completeness_service=Completeness())) as client:
        response = client.get("/api/completeness")

    assert response.status_code == 200
    body = response.json()
    assert body["snapshot_id"] == "snapshot-real-v1"
    assert body["versions"]["generation_registry"] == "registry-v1"
    assert body["items"][0]["forecastable_count"] == 7

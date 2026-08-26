from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from icor.api.app import create_app
from icor.infrastructure.sqlite_coverage_repository import SQLiteCoverageRepository

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    repository = SQLiteCoverageRepository(tmp_path / "coverage.sqlite3")
    with TestClient(create_app(coverage_repository=repository)) as test_client:
        yield test_client


def exact_payload(*, note: str | None = None) -> dict[str, object]:
    return {
        "match_type": "exact_configuration",
        "configuration_id": "demo-aurora-a1-camera-fr-2030",
        "brand": None,
        "model": None,
        "model_year": 2025,
        "note": note,
    }


def test_opportunities_reconcile_and_expose_score_components(client: TestClient) -> None:
    response = client.get("/api/v1/opportunities?group_by=brand")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["base_units"] == 6_560
    assert body["strategy_name"] == "demand_readiness"
    assert body["strategy_version"] == "1"
    assert body["items"][0]["score"]["demand_points"] <= 80
    assert body["items"][0]["score"]["readiness_points"] <= 20


def test_create_exact_coverage_then_ranking_reflects_committed_state(
    client: TestClient,
) -> None:
    created = client.post("/api/v1/production-coverage", json=exact_payload())

    assert created.status_code == 201
    saved = created.json()
    assert saved["brand"] == "Aurora Mobility"
    assert saved["sku"] == "DEMO-AUR-A1-CAM"
    coverage = client.get("/api/v1/production-coverage")
    assert coverage.json() == [saved]

    ranked = client.get("/api/v1/opportunities?group_by=model_year")
    assert ranked.status_code == 200
    aurora_2025 = next(
        row
        for row in ranked.json()["items"]
        if row["brand"] == "Aurora Mobility" and row["model_year"] == 2025
    )
    assert aurora_2025["exact_covered_base_units"] == 250


def test_duplicate_coverage_returns_typed_conflict(client: TestClient) -> None:
    assert client.post("/api/v1/production-coverage", json=exact_payload()).status_code == 201
    response = client.post("/api/v1/production-coverage", json=exact_payload())

    assert response.status_code == 409
    assert response.json()["code"] == "duplicate_coverage"
    assert response.json()["correlation_id"] == response.headers["x-correlation-id"]


def test_invalid_canonical_coverage_is_typed_422(client: TestClient) -> None:
    payload = exact_payload()
    payload["configuration_id"] = "missing"
    response = client.post("/api/v1/production-coverage", json=payload)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_canonical_coverage"


def test_update_and_delete_return_committed_results(client: TestClient) -> None:
    saved = client.post("/api/v1/production-coverage", json=exact_payload()).json()
    updated = client.put(
        f"/api/v1/production-coverage/{saved['coverage_id']}",
        json=exact_payload(note="Updated evidence."),
    )
    assert updated.status_code == 200
    assert updated.json()["note"] == "Updated evidence."

    deleted = client.delete(f"/api/v1/production-coverage/{saved['coverage_id']}")
    assert deleted.status_code == 200
    assert deleted.json() == {"coverage_id": saved["coverage_id"], "deleted": True}
    assert client.get("/api/v1/production-coverage").json() == []


@pytest.mark.parametrize("method", ["put", "delete"])
def test_missing_mutation_identity_is_typed_404(
    client: TestClient, method: str
) -> None:
    if method == "put":
        response = client.put("/api/v1/production-coverage/missing", json=exact_payload())
    else:
        response = client.delete("/api/v1/production-coverage/missing")

    assert response.status_code == 404
    assert response.json()["code"] == "coverage_not_found"


def test_group_drill_down_returns_only_contributing_configurations(
    client: TestClient,
) -> None:
    ranked = client.get("/api/v1/opportunities?group_by=brand").json()
    aurora = next(row for row in ranked["items"] if row["brand"] == "Aurora Mobility")

    response = client.get(
        f"/api/v1/opportunities/{aurora['group_id']}/configurations?group_by=brand"
    )

    assert response.status_code == 200
    assert response.json()
    assert {row["configuration"]["brand"] for row in response.json()} == {
        "Aurora Mobility"
    }


def test_missing_group_drill_down_is_typed_404(client: TestClient) -> None:
    response = client.get(
        "/api/v1/opportunities/missing/configurations?group_by=brand"
    )

    assert response.status_code == 404
    assert response.json()["code"] == "opportunity_not_found"

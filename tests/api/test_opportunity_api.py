from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from icor.api.app import create_app
from icor.infrastructure.demo_planner_repository import DemoPlannerRepository
from icor.infrastructure.sqlite_coverage_repository import SQLiteCoverageRepository

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])
FIXTURE = Path(__file__).parents[2] / "data" / "demo" / "planner-v1.json"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    repository = SQLiteCoverageRepository(tmp_path / "coverage.sqlite3")
    with TestClient(
        create_app(
            repository=DemoPlannerRepository.from_path(FIXTURE),
            coverage_repository=repository,
        )
    ) as test_client:
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
    score = body["items"][0]["score"]
    assert score["demand_points"] <= 80
    assert score["readiness_points"] <= 20
    assert score["demand_population"] == body["demand_population"]
    assert score["demand_basis"] == body["demand_basis"]
    assert 1 <= score["demand_rank"] <= score["demand_population"]


def test_the_same_group_scores_identically_with_and_without_a_filter(
    client: TestClient,
) -> None:
    """The acceptance test for the whole change.

    A reader who filters a market or types a search still wants the vehicle's
    real standing in the market. The units on the row follow the filter; the
    whole score object does not move at all.
    """

    everything = client.get("/api/v1/opportunities?group_by=model").json()
    by_group = {item["group_id"]: item for item in everything["items"]}
    brand = everything["items"][0]["brand"]

    narrowed = client.get(f"/api/v1/opportunities?group_by=model&q={brand}").json()

    assert narrowed["items"]
    for item in narrowed["items"]:
        assert item["score"] == by_group[item["group_id"]]["score"], item["group_id"]
    assert narrowed["demand_population"] == everything["demand_population"]


def test_a_page_that_matches_nothing_still_reports_what_scores_compare_against(
    client: TestClient,
) -> None:
    body = client.get(
        "/api/v1/opportunities?group_by=model&q=zzzznotavehicle"
    ).json()

    assert body["items"] == []
    assert body["demand_population"] > 0
    assert body["demand_basis"]


def test_client_release_rejects_groupings_without_generation_identity(
    tmp_path: Path,
) -> None:
    app = create_app(
        repository=DemoPlannerRepository.from_path(FIXTURE),
        coverage_repository=SQLiteCoverageRepository(tmp_path / "coverage.sqlite3"),
        client_release=True,
    )
    with TestClient(app) as release_client:
        response = release_client.get("/api/v1/opportunities?group_by=brand")

    assert response.status_code == 422
    assert response.json()["code"] == "client_release_scope"


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


def test_opportunity_contributions_return_only_the_fields_needed_by_the_detail(
    client: TestClient,
) -> None:
    ranked = client.get("/api/v1/opportunities?group_by=model_year").json()
    aurora = next(row for row in ranked["items"] if row["brand"] == "Aurora Mobility")

    response = client.get(
        f"/api/v1/opportunities/{aurora['group_id']}/contributions"
        "?group_by=model_year"
    )

    assert response.status_code == 200
    rows = response.json()
    assert rows
    assert set(rows[0]) == {
        "configuration_id",
        "market",
        "forecast_horizon",
        "generation",
        "body_style",
        "demand",
    }
    assert sum(row["demand"]["base_units"] for row in rows) == aurora["demand"]["base_units"]


def test_opportunity_detail_returns_the_exact_ranked_group(client: TestClient) -> None:
    ranked = client.get("/api/v1/opportunities?group_by=model_year").json()
    aurora = next(row for row in ranked["items"] if row["brand"] == "Aurora Mobility")

    response = client.get(
        f"/api/v1/opportunities/{aurora['group_id']}?group_by=model_year"
    )

    assert response.status_code == 200
    assert response.json() == aurora


def test_opportunity_fleet_returns_horizon_totals_by_world_region(
    client: TestClient,
) -> None:
    ranked = client.get("/api/v1/opportunities?group_by=brand").json()
    northstar = next(
        row
        for row in ranked["items"]
        if row["brand"] == "Northstar Automotive"
    )

    response = client.get(
        f"/api/v1/opportunities/{northstar['group_id']}/fleet?group_by=brand"
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "world_region": "Europe",
            "forecast_horizon": 2030,
            "estimated_fleet_units": 95_500,
        },
    ]


def test_missing_group_drill_down_is_typed_404(client: TestClient) -> None:
    response = client.get(
        "/api/v1/opportunities/missing/configurations?group_by=brand"
    )

    assert response.status_code == 404
    assert response.json()["code"] == "opportunity_not_found"

    fleet = client.get("/api/v1/opportunities/missing/fleet?group_by=brand")
    assert fleet.status_code == 404
    assert fleet.json()["code"] == "opportunity_not_found"


def test_search_text_narrows_the_ranking_and_its_summary(client: TestClient) -> None:
    """A filtered page must not report the unfiltered total underneath it."""
    everything = client.get("/api/v1/opportunities?group_by=model").json()
    brand = everything["items"][0]["brand"]

    narrowed = client.get(f"/api/v1/opportunities?group_by=model&q={brand}").json()

    assert narrowed["total"] >= 1
    assert narrowed["total"] <= everything["total"]
    assert all(
        brand.casefold() in item["brand"].casefold()
        or brand.casefold() in (item["model"] or "").casefold()
        for item in narrowed["items"]
    )
    assert narrowed["summary"]["base_units"] <= everything["summary"]["base_units"]


def test_search_text_that_matches_nothing_is_an_empty_page_not_an_error(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/opportunities?group_by=model&q=zzzznotavehicle")

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["summary"]["base_units"] == 0


def test_search_text_is_a_literal_not_a_wildcard(client: TestClient) -> None:
    """`%` must match a percent sign, not every row."""
    response = client.get("/api/v1/opportunities?group_by=model&q=%25")

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_overlong_search_text_is_refused(client: TestClient) -> None:
    response = client.get(f"/api/v1/opportunities?group_by=model&q={'a' * 65}")

    assert response.status_code == 422


def test_sort_by_demand_orders_by_units_not_score(client: TestClient) -> None:
    body = client.get("/api/v1/opportunities?group_by=model&sort=demand").json()

    units = [item["demand"]["base_units"] for item in body["items"]]
    assert units == sorted(units, reverse=True)


def test_sort_by_vehicle_orders_alphabetically(client: TestClient) -> None:
    body = client.get("/api/v1/opportunities?group_by=model&sort=vehicle").json()

    labels = [
        (item["brand"].casefold(), (item["model"] or "").casefold())
        for item in body["items"]
    ]
    assert labels == sorted(labels)


def test_default_sort_is_still_the_score(client: TestClient) -> None:
    default = client.get("/api/v1/opportunities?group_by=model").json()
    explicit = client.get("/api/v1/opportunities?group_by=model&sort=score").json()

    assert [item["group_id"] for item in default["items"]] == [
        item["group_id"] for item in explicit["items"]
    ]
    points = [item["score"]["total_points"] for item in default["items"]]
    assert points == sorted(points, reverse=True)


def test_an_unknown_sort_is_refused(client: TestClient) -> None:
    response = client.get("/api/v1/opportunities?group_by=model&sort=cheapest")

    assert response.status_code == 422

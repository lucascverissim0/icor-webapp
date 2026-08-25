from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from icor.api.app import create_app
from icor.domain.planner import PlanningConfiguration

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_reports_fixture_readiness(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "fixture_ready": True,
        "data_version": "demo-planner-v1",
    }
    assert response.headers["x-correlation-id"]


def test_options_expose_sorted_filters_and_demonstration_scenario(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/planner/options")

    assert response.status_code == 200
    body = response.json()
    assert body["markets"] == ["DE", "FR"]
    assert body["horizons"] == [2028, 2030]
    assert body["brands"] == [
        "Aurora Mobility",
        "Meridian Motors",
        "Northstar Automotive",
        "Velora Works",
    ]
    assert body["evidence_statuses"] == ["demonstration"]
    assert body["scenario"]["evidence_status"] == "demonstration"
    assert body["scenario"]["data_version"] == "demo-planner-v1"


def test_configurations_filter_sort_summarize_and_paginate(client: TestClient) -> None:
    response = client.get(
        "/api/v1/planner/configurations",
        params=[
            ("market", "FR"),
            ("horizon", "2030"),
            ("sort", "base_demand"),
            ("direction", "desc"),
            ("page", "1"),
            ("page_size", "2"),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert body["page"] == 1
    assert body["page_size"] == 2
    assert body["pages"] == 2
    assert body["summary"] == {
        "candidate_count": 3,
        "downside_units": 2130,
        "base_units": 2720,
        "upside_units": 3390,
    }
    assert [row["configuration_id"] for row in body["items"]] == [
        "demo-aurora-a1-camera-fr-2030",
        "demo-northstar-n7-hud-fr-2030",
    ]
    assert body["items"][0]["demand"]["base_units"] == 1240
    assert body["items"][0]["evidence_status"] == "demonstration"


@pytest.mark.parametrize(
    "query",
    [
        "page=0",
        "page_size=101",
        "sort=unsupported",
        "direction=sideways",
        "horizon=not-a-year",
    ],
)
def test_invalid_queries_use_safe_problem_shape(client: TestClient, query: str) -> None:
    response = client.get(f"/api/v1/planner/configurations?{query}")

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_request"
    assert body["message"] == "One or more request values are invalid."
    assert body["correlation_id"] == response.headers["x-correlation-id"]
    assert body["field_errors"]
    assert "traceback" not in response.text.casefold()


def test_detail_preserves_unknowns_and_traceability(client: TestClient) -> None:
    response = client.get(
        "/api/v1/planner/configurations/demo-aurora-a1-camera-fr-2030"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["equipment"]["hud"] is None
    assert body["demand"] == {
        "downside_units": 980,
        "base_units": 1240,
        "upside_units": 1510,
    }
    assert body["identity_confidence"]["reason"]
    assert body["sources"][0]["name"] == "Synthetic vehicle scenario"
    assert body["data_version"] == "demo-planner-v1"


def test_missing_configuration_is_typed_404(client: TestClient) -> None:
    response = client.get("/api/v1/planner/configurations/missing")

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "configuration_not_found"
    assert body["message"] == "The requested configuration was not found."
    assert body["correlation_id"] == response.headers["x-correlation-id"]
    assert body["field_errors"] == []


class BrokenRepository:
    def list_all(self) -> tuple[PlanningConfiguration, ...]:
        raise RuntimeError("private fixture detail must never escape")

    def get(self, configuration_id: str) -> PlanningConfiguration | None:
        raise RuntimeError(f"private identifier {configuration_id} must never escape")


def test_unexpected_errors_are_sanitized() -> None:
    with TestClient(
        create_app(repository=BrokenRepository()),
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/api/v1/planner/configurations")

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert body["message"] == "The planner service could not complete the request."
    assert "private" not in response.text.casefold()
    assert "traceback" not in response.text.casefold()


def test_cors_allows_only_the_configured_local_web_origin(client: TestClient) -> None:
    allowed = client.options(
        "/api/v1/planner/options",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    denied = client.options(
        "/api/v1/planner/options",
        headers={
            "Origin": "https://example.invalid",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "access-control-allow-origin" not in denied.headers


def test_openapi_is_versioned_and_documents_problem_responses() -> None:
    schema = create_app().openapi()

    assert schema["openapi"].startswith("3.1.")
    assert schema["info"]["version"] == "1.0.0"
    responses = schema["paths"]["/api/v1/planner/configurations"]["get"]["responses"]
    assert "422" in responses
    assert "ProblemResponse" in str(responses["422"])

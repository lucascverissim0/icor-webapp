from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from icor.api.app import create_app
from icor.application.registrations import (
    RegistrationPage,
    RegistrationQuery,
    RegistrationRow,
    RegistrationSummary,
)
from icor.infrastructure.snapshot_store import SnapshotStore, SnapshotUnavailableError

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])


class StubRegistrationService:
    def __init__(self) -> None:
        self.queries: list[RegistrationQuery] = []

    def summary(self) -> RegistrationSummary:
        return RegistrationSummary(
            snapshot_id="snapshot-real-2024",
            status="candidate",
            built_at=datetime(2026, 8, 27, 12, 0, tzinfo=UTC),
            database_sha256="a" * 64,
            identity_registry="exact-normalized-model-family-v1",
            geographies=("EU27",),
            years=(2024,),
            total_registrations=Decimal("15"),
            model_count=1,
            model_year_available=False,
            release_ids=("eea-co2cars-2024-final-v30-r1",),
        )

    def ranking(self, query: RegistrationQuery) -> RegistrationPage:
        query.validate()
        self.queries.append(query)
        return RegistrationPage(
            items=(
                RegistrationRow(
                    rank=1,
                    vehicle_id="vehicle-example-alpha",
                    make="Example Motors",
                    model="Alpha",
                    model_year=None,
                    registrations=Decimal("15"),
                    status="derived_observed",
                    evidence_confidence=79,
                    input_observation_count=2,
                    release_ids=("eea-co2cars-2024-final-v30-r1",),
                    source_ids=("eea-co2-monitoring",),
                ),
            ),
            total=1,
            total_registrations=Decimal("15"),
            page=query.page,
            page_size=query.page_size,
            pages=1,
            snapshot_id="snapshot-real-2024",
        )


def test_registration_summary_serializes_real_snapshot_scope() -> None:
    client = TestClient(create_app(registration_service=StubRegistrationService()))

    response = client.get("/api/v1/registrations/summary")

    assert response.status_code == 200
    assert response.json() == {
        "snapshot_id": "snapshot-real-2024",
        "status": "candidate",
        "built_at": "2026-08-27T12:00:00Z",
        "database_sha256": "a" * 64,
        "identity_registry": "exact-normalized-model-family-v1",
        "geographies": ["EU27"],
        "years": [2024],
        "total_registrations": "15",
        "model_count": 1,
        "model_year_available": False,
        "release_ids": ["eea-co2cars-2024-final-v30-r1"],
    }


def test_registration_ranking_passes_bounded_filters() -> None:
    service = StubRegistrationService()
    client = TestClient(create_app(registration_service=service))

    response = client.get(
        "/api/v1/registrations/ranking",
        params={
            "geography": "EU27",
            "year": 2024,
            "search": "Alpha",
            "page": 2,
            "page_size": 10,
        },
    )

    assert response.status_code == 200
    assert service.queries == [
        RegistrationQuery(
            geography="EU27", year=2024, search="Alpha", page=2, page_size=10
        )
    ]
    assert response.json()["items"][0] == {
        "rank": 1,
        "vehicle_id": "vehicle-example-alpha",
        "make": "Example Motors",
        "model": "Alpha",
        "model_year": None,
        "registrations": "15",
        "status": "derived_observed",
        "evidence_confidence": 79,
        "input_observation_count": 2,
        "release_ids": ["eea-co2cars-2024-final-v30-r1"],
        "source_ids": ["eea-co2-monitoring"],
    }


def test_registration_routes_fail_closed_without_real_snapshot(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ICOR_EVIDENCE_CANDIDATE", raising=False)
    client = TestClient(
        create_app(snapshot_root=Path("C:/local/missing-active-root"))
    )

    response = client.get("/api/v1/registrations/summary")

    assert response.status_code == 503
    assert response.json()["code"] == "registration_data_unavailable"
    assert "demo" not in response.json()["message"].casefold()


def test_registration_query_validation_is_typed_and_bounded() -> None:
    service = StubRegistrationService()
    client = TestClient(create_app(registration_service=service))

    response = client.get(
        "/api/v1/registrations/ranking", params={"page_size": 101}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert service.queries == []


def test_openapi_describes_real_registration_contract() -> None:
    app = create_app(registration_service=StubRegistrationService())

    assert "/api/v1/registrations/summary" in app.openapi()["paths"]
    assert "/api/v1/registrations/ranking" in app.openapi()["paths"]
    assert "demonstration contract" not in app.description.casefold()


def test_candidate_path_is_not_a_runtime_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ICOR_EVIDENCE_CANDIDATE", "C:/local/candidate")

    app = create_app(snapshot_root=Path("C:/local/missing-active-root"))

    assert app.state.evidence_service is None
    assert app.state.registration_service is None


def test_active_root_is_preferred_for_registration_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_root = "C:/local/evidence"
    opened = []
    monkeypatch.setenv("ICOR_EVIDENCE_ACTIVE_ROOT", active_root)
    monkeypatch.setenv("ICOR_EVIDENCE_CANDIDATE", "C:/local/candidate")

    def unavailable(store):
        opened.append(store.root)
        raise SnapshotUnavailableError("missing")

    monkeypatch.setattr(SnapshotStore, "open_active_snapshot", unavailable)

    app = create_app()

    assert opened == [Path(active_root)]
    assert app.state.registration_service is None

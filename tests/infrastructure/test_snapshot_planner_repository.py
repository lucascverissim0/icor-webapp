from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from icor.domain.evidence import ConfidenceBand
from icor.domain.generations import GenerationIdentityKind
from icor.domain.planner import PlannerQuery
from icor.domain.snapshots import SnapshotManifest, SnapshotStatus, SnapshotVersions
from icor.infrastructure.snapshot_planner_repository import SnapshotPlannerRepository


def _manifest() -> SnapshotManifest:
    return SnapshotManifest(
        snapshot_id="snapshot-real-v1",
        status=SnapshotStatus.CANDIDATE,
        built_at=datetime(2026, 8, 28, 10, 45, tzinfo=UTC),
        deterministic_seed=20260827,
        release_ids=("eea-2020",),
        versions=SnapshotVersions(
            source_registry="sources-v1",
            identity_registry="identity-v1",
            reconciliation_method="precedence-v1",
            confidence_method="confidence-v1",
            estimation_method="interpolation-v1",
            survival_method="survival-v1",
            hazard_method="hazard-v1",
            forecast_method="forecast-v1",
            generation_registry="generation-registry-v1",
            generation_resolver="generation-resolver-v1",
        ),
        database_sha256="a" * 64,
        observation_count=3,
        published_value_count=3,
        warnings=(),
    )


class Ledger:
    def list_releases(self):
        return (
            SimpleNamespace(
                release_id="eea-2020",
                publisher="European Environment Agency",
                source_url="https://example.test/eea",
            ),
        )

    def list_vehicles(self):
        return (
            SimpleNamespace(
                vehicle_id="vehicle-volkswagen-golf-eu",
                make="Volkswagen",
                model="Golf",
            ),
        )

    def list_generations(self):
        return (
            SimpleNamespace(
                generation_id="generation-volkswagen-golf-eu",
                canonical_vehicle_id="vehicle-volkswagen-golf-eu",
                display_name="estimated-generation-1 (2020-2022)",
                start_month=date(2020, 1, 1),
                end_month=date(2022, 12, 1),
                identity_kind=GenerationIdentityKind.ESTIMATED,
                body_style=None,
                facelift=None,
                confidence_reasons=("annual-window-estimate",),
                evidence_ids=("observation-golf-2020",),
            ),
        )

    def list_cohort_estimates(self):
        return (
            SimpleNamespace(
                cohort_id="cohort-golf-de-2020",
                generation_id="generation-volkswagen-golf-eu",
                geography="DE",
                registration_cohort_year=2020,
                active_fleet_p10=Decimal("80"),
                active_fleet_p50=Decimal("90"),
                active_fleet_p90=Decimal("95"),
                input_observation_ids=("observation-golf-2020",),
                confidence=ConfidenceBand.LOW,
                reason_codes=("observed-registration-cohort",),
            ),
        )

    def list_opportunity_estimates(self):
        return (
            SimpleNamespace(
                opportunity_id="opportunity-golf-de-2028",
                generation_id="generation-volkswagen-golf-eu",
                geography="DE",
                horizon_year=2028,
                p10=Decimal("10.2"),
                p50=Decimal("12.6"),
                p90=Decimal("15.8"),
                active_fleet_p50=Decimal("90"),
                input_cohort_ids=("cohort-golf-de-2020",),
                confidence=ConfidenceBand.LOW,
                assumption_ids=("assumption-hazard-v1",),
                reason_codes=("uncalibrated-fitment-and-hazard",),
            ),
        )


class SQLiteLedger(Ledger):
    def __init__(self, path: Path) -> None:
        self.path = path


@pytest.fixture
def sqlite_repository(tmp_path: Path) -> SnapshotPlannerRepository:
    path = tmp_path / "planner.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE canonical_vehicle (
                vehicle_id TEXT PRIMARY KEY, make TEXT, model TEXT
            );
            CREATE TABLE generation_entry (
                generation_id TEXT PRIMARY KEY, display_name TEXT, start_month TEXT,
                end_month TEXT, identity_kind TEXT, body_style TEXT, facelift TEXT,
                confidence_reasons TEXT, evidence_ids TEXT
            );
            CREATE TABLE cohort_estimate (
                cohort_id TEXT PRIMARY KEY, registration_cohort_year INTEGER
            );
            CREATE TABLE opportunity_estimate (
                opportunity_id TEXT PRIMARY KEY, generation_id TEXT,
                canonical_vehicle_id TEXT, geography TEXT, horizon_year INTEGER,
                p10 TEXT, p50 TEXT, p90 TEXT, active_fleet_p50 TEXT,
                confidence TEXT, assumption_ids TEXT, reason_codes TEXT
            );
            CREATE TABLE opportunity_input (opportunity_id TEXT, cohort_id TEXT);
            CREATE TABLE planner_option (
                option_kind TEXT, option_value TEXT, sort_key TEXT
            );
            """
        )
        for suffix, make, model, demand in (
            ("golf", "Volkswagen", "Golf", "13"),
            ("polo", "Volkswagen", "Polo", "8"),
        ):
            vehicle_id = f"vehicle-{suffix}"
            generation_id = f"generation-{suffix}"
            cohort_id = f"cohort-{suffix}"
            opportunity_id = f"opportunity-{suffix}"
            connection.execute(
                "INSERT INTO canonical_vehicle VALUES (?, ?, ?)",
                (vehicle_id, make, model),
            )
            connection.execute(
                "INSERT INTO generation_entry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    generation_id, f"{model} generation", "2020-01-01", "2024-12-01",
                    "estimated", "Hatchback", None, '["estimated"]', '["obs"]',
                ),
            )
            connection.execute(
                "INSERT INTO cohort_estimate VALUES (?, 2020)", (cohort_id,)
            )
            connection.execute(
                "INSERT INTO opportunity_estimate VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    opportunity_id, generation_id, vehicle_id, "DE", 2028,
                    str(int(demand) - 2), demand, str(int(demand) + 3), "90",
                    "low", '["assumption"]', '["reason"]',
                ),
            )
            connection.execute(
                "INSERT INTO opportunity_input VALUES (?, ?)",
                (opportunity_id, cohort_id),
            )
        connection.executemany(
            "INSERT INTO planner_option VALUES (?, ?, ?)",
            (
                ("market", "DE", "de"),
                ("horizon", "2028", "2028"),
                ("brand", "Volkswagen", "volkswagen"),
                ("model", "Golf", "golf"),
                ("model", "Polo", "polo"),
            ),
        )
    return SnapshotPlannerRepository(SQLiteLedger(path), _manifest())


def test_snapshot_adapter_exposes_generation_opportunity_without_fitment_claims() -> None:
    repository = SnapshotPlannerRepository(Ledger(), _manifest())

    rows = repository.list_all()

    assert len(rows) == 1
    row = rows[0]
    assert row.configuration_id == "opportunity-golf-de-2028"
    assert row.sku is None
    assert row.part_family is None
    assert row.generation_id == "generation-volkswagen-golf-eu"
    assert row.generation_identity_kind == "estimated"
    assert row.demand.downside_units == 10
    assert row.demand.base_units == 13
    assert row.demand.upside_units == 16
    assert row.vehicle_exposure_units == 90
    assert row.assumption_ids == ("assumption-hazard-v1",)
    assert row.evidence_ids == ("observation-golf-2020",)
    assert row.data_version == "snapshot-real-v1"
    assert row.model_year_demand[0].model_year == 2020


def test_snapshot_adapter_reports_one_shared_snapshot_version_set() -> None:
    repository = SnapshotPlannerRepository(Ledger(), _manifest())

    assert repository.snapshot_id == "snapshot-real-v1"
    assert repository.versions.generation_registry == "generation-registry-v1"
    assert repository.get("opportunity-golf-de-2028") == repository.list_all()[0]
    assert repository.get("missing") is None


def test_sqlite_options_search_and_detail_are_bounded(
    sqlite_repository: SnapshotPlannerRepository,
) -> None:
    options = sqlite_repository.options()
    page = sqlite_repository.search(PlannerQuery(page=1, page_size=1))
    detail = sqlite_repository.get("opportunity-golf")
    demand = sqlite_repository.list_model_year_demand(
        "opportunity-golf", page=1, page_size=1
    )

    assert options.markets == ("DE",)
    assert options.models == ("Golf", "Polo")
    assert page.total == 2
    assert len(page.items) == 1
    assert page.items[0].configuration_id == "opportunity-golf"
    assert detail is not None and detail.configuration_id == "opportunity-golf"
    assert demand == detail.model_year_demand
    assert sqlite_repository._records is None


def test_interactive_sqlite_methods_reject_an_unbounded_projection(
    sqlite_repository: SnapshotPlannerRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = sqlite_repository._query_sqlite

    def guarded_query(
        path: Path,
        where: str,
        parameters: tuple[object, ...],
        order_by: str,
        limit: int | None,
        offset: int,
    ):
        if limit is None:
            raise AssertionError("interactive query attempted an unbounded projection")
        return query(path, where, parameters, order_by, limit, offset)

    monkeypatch.setattr(sqlite_repository, "_query_sqlite", guarded_query)
    assert len(sqlite_repository.search(PlannerQuery(page_size=1)).items) == 1
    assert sqlite_repository.get("opportunity-golf") is not None

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from icor.application.opportunities import (
    OpportunityFleetEstimate,
    OpportunityGroupBy,
    OpportunityQuery,
)
from icor.application.ranking import DemandReadinessV1
from icor.application.worked_models import IcorWorkedModelCatalog
from icor.domain.evidence import ConfidenceBand
from icor.domain.generations import GenerationIdentityKind
from icor.domain.planner import PlannerQuery
from icor.domain.snapshots import SnapshotManifest, SnapshotStatus, SnapshotVersions
from icor.infrastructure.snapshot_opportunity_repository import SnapshotOpportunityRepository
from icor.infrastructure.snapshot_planner_repository import SnapshotPlannerRepository
from icor.infrastructure.sqlite_coverage_repository import SQLiteCoverageRepository


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
                survival_method="survival-v1",
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
                hazard_method="hazard-v1",
                forecast_method="forecast-v1",
                uncertainty_method="uncertainty-v1",
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
                cohort_id TEXT PRIMARY KEY, registration_cohort_year INTEGER,
                active_fleet_p10 TEXT, active_fleet_p50 TEXT, active_fleet_p90 TEXT,
                survival_method TEXT
            );
            CREATE TABLE opportunity_estimate (
                opportunity_id TEXT PRIMARY KEY, generation_id TEXT,
                canonical_vehicle_id TEXT, geography TEXT, horizon_year INTEGER,
                p10 TEXT, p50 TEXT, p90 TEXT, active_fleet_p50 TEXT,
                hazard_method TEXT, forecast_method TEXT, uncertainty_method TEXT,
                confidence TEXT, assumption_ids TEXT, reason_codes TEXT
            );
            CREATE TABLE opportunity_input (
                opportunity_id TEXT, cohort_id TEXT, input_position INTEGER DEFAULT 0
            );
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
                "INSERT INTO cohort_estimate VALUES "
                "(?, 2020, '80', '90', '95', 'survival-v1')",
                (cohort_id,),
            )
            connection.execute(
                "INSERT INTO opportunity_estimate VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    opportunity_id, generation_id, vehicle_id, "DE", 2028,
                    str(int(demand) - 2), demand, str(int(demand) + 3), "90",
                    "hazard-v1", "forecast-v1", "uncertainty-v1",
                    "low", '["assumption"]', '["reason"]',
                ),
            )
            connection.execute(
                "INSERT INTO opportunity_input (opportunity_id, cohort_id) VALUES (?, ?)",
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


def test_sqlite_search_aggregates_opportunities_before_loading_page_lineage(
    sqlite_repository: SnapshotPlannerRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []
    connect = sqlite_repository._connect

    def traced_connection(path: Path):  # type: ignore[no-untyped-def]
        connection = connect(path)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(sqlite_repository, "_connect", traced_connection)

    page = sqlite_repository.search(PlannerQuery(page=1, page_size=1))

    assert len(page.items) == 1
    summary = next(statement for statement in statements if "candidate_count" in statement)
    assert "opportunity_input" not in summary
    paged = next(statement for statement in statements if "LIMIT 1 OFFSET 0" in statement)
    attribution = next(
        statement for statement in statements if "cohort_attribution AS" in statement
    )
    assert "opportunity_input" not in paged
    assert "o.opportunity_id IN ('opportunity-golf')" in attribution


def test_sqlite_search_reuses_a_bounded_immutable_page(
    sqlite_repository: SnapshotPlannerRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    search = sqlite_repository._search_sqlite

    def counted_search(path: Path, query: PlannerQuery):
        nonlocal calls
        calls += 1
        return search(path, query)

    monkeypatch.setattr(sqlite_repository, "_search_sqlite", counted_search)
    query = PlannerQuery(page=1, page_size=1)

    assert sqlite_repository.search(query) == sqlite_repository.search(query)
    assert calls == 1


def test_sqlite_opportunity_ranking_and_drill_down_are_bounded(
    sqlite_repository: SnapshotPlannerRepository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = SnapshotOpportunityRepository(
        sqlite_repository,
        SQLiteCoverageRepository(tmp_path / "coverage.sqlite3"),
        DemandReadinessV1(),
    )
    query = OpportunityQuery(
        group_by=OpportunityGroupBy.MODEL, page=1, page_size=1
    )
    calls: list[int | None] = []
    original = sqlite_repository._query_sqlite

    def guarded_query(path, where, parameters, order_by, limit, offset):  # type: ignore[no-untyped-def]
        calls.append(limit)
        if limit is None:
            raise AssertionError("opportunity drill-down attempted an unbounded query")
        return original(path, where, parameters, order_by, limit, offset)

    monkeypatch.setattr(sqlite_repository, "_query_sqlite", guarded_query)
    page = repository.search(query)

    assert page.total == 2
    assert page.pages == 2
    assert len(page.items) == 1
    assert page.summary.base_units == 21
    assert page.items[0].brand == "Volkswagen"
    assert page.items[0].coverage_status.value == "uncovered"
    rows = repository.drill_down(page.items[0].group_id, query, 1, 1)
    assert len(rows) == 1
    assert calls == [1]


def test_model_year_opportunities_allocate_and_reconcile_every_input_cohort(
    sqlite_repository: SnapshotPlannerRepository,
    tmp_path: Path,
) -> None:
    path = sqlite_repository._ledger.path
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO cohort_estimate VALUES "
            "('cohort-golf-2021', 2021, '20', '30', '40', 'survival-v1')"
        )
        connection.execute(
            "INSERT INTO opportunity_input VALUES "
            "('opportunity-golf', 'cohort-golf-2021', 1)"
        )
        connection.execute(
            """CREATE TABLE opportunity_cohort_attribution (
                opportunity_id TEXT, cohort_id TEXT,
                registration_cohort_year INTEGER,
                downside_units INTEGER, base_units INTEGER, upside_units INTEGER
            )"""
        )
        connection.executemany(
            "INSERT INTO opportunity_cohort_attribution VALUES (?, ?, ?, ?, ?, ?)",
            (
                ("opportunity-golf", "cohort-golf", 2020, 8, 10, 12),
                ("opportunity-golf", "cohort-golf-2021", 2021, 3, 3, 4),
                ("opportunity-polo", "cohort-polo", 2020, 6, 8, 11),
            ),
        )
    repository = SnapshotOpportunityRepository(
        sqlite_repository,
        SQLiteCoverageRepository(tmp_path / "coverage.sqlite3"),
        DemandReadinessV1(),
    )
    query = OpportunityQuery(
        group_by=OpportunityGroupBy.MODEL_YEAR, page=1, page_size=10
    )

    page = repository.search(query)
    golf = {
        row.model_year: row
        for row in page.items
        if row.brand == "Volkswagen" and row.model == "Golf"
    }

    assert page.summary.base_units == 21
    assert {
        year: (
            row.demand.downside_units,
            row.demand.base_units,
            row.demand.upside_units,
        )
        for year, row in golf.items()
    } == {2020: (8, 10, 12), 2021: (3, 3, 4)}
    detail = sqlite_repository.get("opportunity-golf")
    assert detail is not None
    assert sum(row.demand.base_units for row in detail.model_year_demand) == 13
    assert [row.model_year for row in detail.model_year_demand] == [2020, 2021]
    drill_down = repository.drill_down(golf[2021].group_id, query, 1, 10)
    assert len(drill_down) == 1
    assert drill_down[0].model_year_demand.model_year == 2021
    assert drill_down[0].model_year_demand.demand.base_units == 3
    assert repository.fleet_estimates(golf[2021].group_id, query) == (
        OpportunityFleetEstimate("Europe", 2028, 30),
    )
    contributions = repository.contributions(golf[2021].group_id, query)
    assert [
        (
            row.market,
            row.forecast_horizon,
            row.demand.downside_units,
            row.demand.base_units,
            row.demand.upside_units,
        )
        for row in contributions
    ] == [("DE", 2028, 3, 3, 4)]


def test_cached_ranking_row_makes_clicked_detail_lookup_constant_time(
    sqlite_repository: SnapshotPlannerRepository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = SnapshotOpportunityRepository(
        sqlite_repository,
        SQLiteCoverageRepository(tmp_path / "coverage.sqlite3"),
        DemandReadinessV1(),
    )
    listed_query = OpportunityQuery(
        group_by=OpportunityGroupBy.MODEL_YEAR, page=1, page_size=100
    )
    page = repository.search(listed_query)
    selected = page.items[0]

    def unexpected_score(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("cached clicked row was scored again")

    monkeypatch.setattr(repository, "_scored_cte", unexpected_score)

    detail = repository.get(
        selected.group_id,
        OpportunityQuery(group_by=OpportunityGroupBy.MODEL_YEAR),
    )

    assert detail == selected


def test_sqlite_opportunity_exposes_reviewed_generation_and_legacy_icor_readiness(
    sqlite_repository: SnapshotPlannerRepository,
    tmp_path: Path,
) -> None:
    worked = tmp_path / "worked.txt"
    worked.write_text('{\n  "vw golf": {2020: "G8"}\n}\n', encoding="utf-8")
    repository = SnapshotOpportunityRepository(
        sqlite_repository,
        SQLiteCoverageRepository(tmp_path / "coverage.sqlite3"),
        DemandReadinessV1(),
        worked_models=IcorWorkedModelCatalog.from_path(worked),
    )

    page = repository.search(
        OpportunityQuery(group_by=OpportunityGroupBy.MODEL_YEAR, page=1, page_size=10)
    )
    golf = next(row for row in page.items if row.model == "Golf")

    assert golf.model_year == 2020
    assert golf.generation_name == "Golf Mk8"
    assert golf.generation_basis == "manufacturer_generation_window"
    assert golf.icor_worked_base_units == 13
    assert golf.fallback_covered_base_units == 13
    assert golf.score.readiness_points == 10


def test_verified_only_opportunities_exclude_unreviewed_models(
    sqlite_repository: SnapshotPlannerRepository,
    tmp_path: Path,
) -> None:
    repository = SnapshotOpportunityRepository(
        sqlite_repository,
        SQLiteCoverageRepository(tmp_path / "coverage.sqlite3"),
        DemandReadinessV1(),
        verified_only=True,
    )

    page = repository.search(
        OpportunityQuery(group_by=OpportunityGroupBy.MODEL_YEAR, page=1, page_size=10)
    )

    assert page.total == 1
    assert [(row.brand, row.model, row.generation_name) for row in page.items] == [
        ("Volkswagen", "Golf", "Golf Mk8")
    ]


def test_sqlite_opportunity_search_scores_once_without_bulk_lineage_grouping(
    sqlite_repository: SnapshotPlannerRepository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = SnapshotOpportunityRepository(
        sqlite_repository,
        SQLiteCoverageRepository(tmp_path / "coverage.sqlite3"),
        DemandReadinessV1(),
    )
    statements: list[str] = []
    connect = repository._connect

    def traced_connection():  # type: ignore[no-untyped-def]
        connection = connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(repository, "_connect", traced_connection)

    page = repository.search(
        OpportunityQuery(group_by=OpportunityGroupBy.BRAND, page=1, page_size=1)
    )
    repeated = repository.search(
        OpportunityQuery(group_by=OpportunityGroupBy.BRAND, page=1, page_size=1)
    )

    assert len(page.items) == 1
    assert repeated == page
    scored = [statement for statement in statements if "WITH base_atoms" in statement]
    assert len(scored) == 1
    assert "GROUP BY o.opportunity_id" not in scored[0]
    assert "summary_total" in scored[0]
    base_atoms = scored[0].split("WITH base_atoms AS", 1)[1].split(
        "), atoms AS", 1
    )[0]
    assert "opportunity_input" not in base_atoms

    model_year_cte, unused = repository._scored_cte(
        OpportunityQuery(
            group_by=OpportunityGroupBy.MODEL_YEAR, page=1, page_size=1
        ),
        include_coverage=False,
    )
    assert "cohort_attribution AS" in model_year_cte
    assert "i.input_position = 0" not in model_year_cte
    assert "GROUP BY o.opportunity_id" not in model_year_cte


def test_row_provenance_is_read_from_the_served_rows_not_the_manifest(
    sqlite_repository: SnapshotPlannerRepository,
) -> None:
    """The test that would have caught all four instances of this defect.

    The manifest says `survival-v1`. The rows say otherwise. The channel must
    report what produced the numbers it is actually serving.
    """

    with sqlite3.connect(sqlite_repository._ledger.path) as connection:
        connection.execute(
            "UPDATE cohort_estimate SET survival_method = ?",
            ("uk-dft-licensed-stock-band-v1",),
        )

    page = sqlite_repository.search(PlannerQuery(page=1, page_size=10))

    assert page.items
    for record in page.items:
        assert record.row_methods is not None
        assert record.row_methods.survival_method == "uk-dft-licensed-stock-band-v1"
        assert record.row_methods.survival_method != "survival-v1"


def test_cohorts_disagreeing_about_the_curve_are_reported_as_mixed(
    sqlite_repository: SnapshotPlannerRepository,
) -> None:
    """A snapshot mixing curves must say so rather than pick a plausible winner."""

    with sqlite3.connect(sqlite_repository._ledger.path) as connection:
        opportunity_id, cohort_id = connection.execute(
            "SELECT opportunity_id, cohort_id FROM opportunity_input LIMIT 1"
        ).fetchone()
        connection.execute(
            "UPDATE cohort_estimate SET survival_method = ? WHERE cohort_id = ?",
            ("constant-annual-retention-v1", cohort_id),
        )
        connection.execute(
            "INSERT INTO cohort_estimate VALUES "
            "('cohort-mixed', 2021, '10', '20', '30', 'uk-dft-licensed-stock-band-v1')"
        )
        connection.execute(
            "INSERT INTO opportunity_input VALUES (?, 'cohort-mixed', 1)",
            (opportunity_id,),
        )

    page = sqlite_repository.search(PlannerQuery(page=1, page_size=10))

    mixed = [
        record
        for record in page.items
        if record.row_methods is not None
        and record.row_methods.survival_method.startswith("mixed:")
    ]
    assert mixed


def test_snapshot_scope_and_row_scope_provenance_stay_distinct(
    sqlite_repository: SnapshotPlannerRepository,
) -> None:
    """A client must be able to tell these apart, so they are separate fields."""

    page = sqlite_repository.search(PlannerQuery(page=1, page_size=10))

    record = page.items[0]
    assert record.method_versions is not None
    assert record.method_versions.survival_method == "survival-v1"
    assert record.row_methods is not None
    assert record.row_methods.hazard_method == "hazard-v1"
    assert record.row_methods.uncertainty_method == "uncertainty-v1"

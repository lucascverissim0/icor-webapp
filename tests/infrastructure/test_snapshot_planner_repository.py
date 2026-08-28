from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

from icor.domain.evidence import ConfidenceBand
from icor.domain.generations import GenerationIdentityKind
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

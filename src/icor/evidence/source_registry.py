"""Reviewed production composition for official evidence source adapters."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from icor.application.completeness import CompletenessService
from icor.application.generation_mapping import GenerationMappingService
from icor.application.generation_planning import GenerationPlanningService
from icor.application.snapshot_build import EvidenceLoader
from icor.domain.snapshots import SnapshotVersions
from icor.evidence.identity import (
    ExactNormalizedIdentityResolver,
    IdentityAttributingRepository,
)
from icor.evidence.sources.eea import EEAAnnualAggregateLoader, EEAPassengerCarLoader
from icor.evidence.sources.kba import KBAFZ10Loader
from icor.evidence.sources.uk_dft import (
    UKActiveFleetLoader,
    UKFirstRegistrationLoader,
    UKVehicleAgeLoader,
)
from icor.forecasting.survival import load_promoted_survival_model
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

OFFICIAL_SOURCE_VERSIONS = SnapshotVersions(
    source_registry="official-sources-v1",
    identity_registry="exact-normalized-model-family-v1",
    reconciliation_method="dependency-precedence-v1",
    confidence_method="source-evidence-v1",
    estimation_method="linear-gap-interpolation-v1",
    survival_method="uk-dft-licensed-stock-band-v1",
    hazard_method="france-insurance-windshield-hazard-v2",
    forecast_method="validated-recency-damped-ensemble-v2",
    uncertainty_method="quantile-matched-split-normal-propagation-v2",
    generation_registry="public-generation-registry-v2",
    generation_resolver="generation-resolver-v1",
)


def official_loader_registry() -> Mapping[str, EvidenceLoader]:
    """Return a fresh fail-closed map of parser names reviewed for local use."""

    return {
        "eea_co2_cars_zip_v1": EEAPassengerCarLoader(),
        "eea_co2_cars_annual_aggregate_csv_v1": EEAAnnualAggregateLoader(),
        "kba_fz10_xlsx_v1": KBAFZ10Loader(),
        "uk_dft_veh0120_csv_v1": UKActiveFleetLoader(),
        "uk_dft_veh0160_csv_v1": UKFirstRegistrationLoader(),
        "uk_dft_veh0124_csv_v1": UKVehicleAgeLoader(),
    }


def official_repository_transformer(
    repository: SQLiteEvidenceRepository, reviewed_at: datetime
) -> IdentityAttributingRepository:
    """Attribute official observations through the reviewed conservative resolver."""

    return IdentityAttributingRepository(
        repository,
        ExactNormalizedIdentityResolver(),
        reviewed_at=reviewed_at,
    )


def official_repository_finalizer(
    repository: SQLiteEvidenceRepository,
    reviewed_at: datetime,
) -> None:
    """Materialize deterministic generation assignments after official loading."""

    result = GenerationMappingService(
        registry_version=OFFICIAL_SOURCE_VERSIONS.generation_registry,
        resolver_version=OFFICIAL_SOURCE_VERSIONS.generation_resolver,
    ).apply(repository, reviewed_at=reviewed_at)
    if result.unassigned_ids or result.assigned_count != result.usable_count:
        raise ValueError("official generation mapping is incomplete")
    planning = GenerationPlanningService(
        survival=load_promoted_survival_model(),
    ).apply(repository, seed=20260827)
    if planning.cohort_count == 0 or planning.opportunity_count == 0:
        raise ValueError("official generation planning is incomplete")
    if CompletenessService().materialize(repository) == 0:
        raise ValueError("official completeness reporting is unavailable")

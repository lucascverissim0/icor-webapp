"""Planner use cases independent of storage and transport adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from icor.domain.planner import (
    EvidenceStatus,
    ModelYearDemand,
    PlannerPage,
    PlannerQuery,
    PlanningConfiguration,
    filter_sort_paginate,
)
from icor.domain.snapshots import SnapshotVersions


class PlannerRepository(Protocol):
    def list_all(self) -> tuple[PlanningConfiguration, ...]: ...

    def get(self, configuration_id: str) -> PlanningConfiguration | None: ...

    def list_model_year_demand(self) -> tuple[ModelYearDemand, ...]: ...


@dataclass(frozen=True, slots=True)
class ScenarioMetadata:
    name: str
    description: str
    evidence_status: EvidenceStatus
    data_version: str
    updated_at: datetime
    versions: SnapshotVersions | None = None


@dataclass(frozen=True, slots=True)
class PlannerOptions:
    markets: tuple[str, ...]
    horizons: tuple[int, ...]
    brands: tuple[str, ...]
    models: tuple[str, ...]
    evidence_statuses: tuple[EvidenceStatus, ...]
    scenario: ScenarioMetadata


_EVIDENCE_RANK = {
    EvidenceStatus.DEMONSTRATION: 1,
    EvidenceStatus.PROTOTYPE: 2,
    EvidenceStatus.VALIDATED: 3,
}


class PlannerService:
    def __init__(self, repository: PlannerRepository) -> None:
        self._repository = repository

    def options(self) -> PlannerOptions:
        records = self._repository.list_all()
        if not records:
            raise ValueError("planner repository contains no configurations")
        data_versions = {record.data_version for record in records}
        if len(data_versions) != 1:
            raise ValueError("planner repository contains mixed data versions")
        evidence_statuses = tuple(
            sorted(
                {record.evidence_status for record in records},
                key=_EVIDENCE_RANK.__getitem__,
            )
        )
        is_validated_snapshot = evidence_statuses == (EvidenceStatus.VALIDATED,)
        return PlannerOptions(
            markets=tuple(sorted({record.market for record in records})),
            horizons=tuple(sorted({record.forecast_horizon for record in records})),
            brands=tuple(sorted({record.brand for record in records})),
            models=tuple(sorted({record.model for record in records})),
            evidence_statuses=evidence_statuses,
            scenario=ScenarioMetadata(
                name=(
                    "Generation replacement opportunity baseline"
                    if is_validated_snapshot
                    else "Windshield demand planning demonstration"
                ),
                description=(
                    "Official registration history projected to generation-level "
                    "replacement opportunity ranges. This is not exact fitment demand."
                    if is_validated_snapshot
                    else "Synthetic configuration-level demand for product workflow review."
                ),
                evidence_status=evidence_statuses[0],
                data_version=data_versions.pop(),
                updated_at=max(record.updated_at for record in records),
                versions=getattr(self._repository, "versions", None),
            ),
        )

    def search(self, query: PlannerQuery) -> PlannerPage:
        return filter_sort_paginate(self._repository.list_all(), query)

    def detail(self, configuration_id: str) -> PlanningConfiguration | None:
        return self._repository.get(configuration_id)

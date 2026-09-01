"""Build immutable query projections before an evidence snapshot is sealed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class QueryProjectionRepository(Protocol):
    def rebuild_query_projections(self) -> dict[str, int]: ...


@dataclass(frozen=True, slots=True)
class SnapshotQueryProjectionResult:
    registration_family_count: int
    registration_label_count: int
    evidence_release_count: int
    planner_option_count: int


class SnapshotQueryProjectionService:
    def apply(
        self, repository: QueryProjectionRepository
    ) -> SnapshotQueryProjectionResult:
        counts = repository.rebuild_query_projections()
        return SnapshotQueryProjectionResult(
            registration_family_count=counts["registration_families"],
            registration_label_count=counts["registration_labels"],
            evidence_release_count=counts["evidence_releases"],
            planner_option_count=counts["planner_options"],
        )

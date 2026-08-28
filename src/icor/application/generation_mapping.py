"""Deterministic batch generation mapping for canonical evidence observations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from icor.domain.evidence import MappingStatus, Measure, Observation
from icor.domain.generations import GenerationAssignment, GenerationEntry
from icor.generations.estimator import EstimatedGenerationBuilder
from icor.generations.registry import GenerationRegistry
from icor.generations.resolver import GenerationResolver, ResolutionRequest
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

_UNUSABLE_STATUSES = frozenset(
    {MappingStatus.AMBIGUOUS, MappingStatus.REJECTED, MappingStatus.UNRESOLVED}
)
_BATCH_SIZE = 2_000


@dataclass(frozen=True, slots=True)
class GenerationMappingResult:
    usable_count: int
    assigned_count: int
    evidence_only_count: int
    generation_count: int
    unassigned_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _UsableObservation:
    observation: Observation
    cohort_year: int


class GenerationMappingService:
    def __init__(
        self,
        *,
        registry_version: str = "generation-registry-v1",
        resolver_version: str = "generation-resolver-v1",
    ) -> None:
        self.registry_version = registry_version
        self.resolver_version = resolver_version

    def apply(
        self,
        repository: SQLiteEvidenceRepository,
        *,
        reviewed_at: datetime,
    ) -> GenerationMappingResult:
        if reviewed_at.tzinfo is None or reviewed_at.utcoffset() != timedelta(0):
            raise ValueError("generation mapping review time must be UTC")
        vehicles = {item.vehicle_id: item for item in repository.list_vehicles()}
        histories: dict[str, list[_UsableObservation]] = defaultdict(list)
        evidence_only_count = 0
        for observation in repository.list_observations():
            if (
                observation.canonical_vehicle_id is None
                or observation.mapping_status in _UNUSABLE_STATUSES
            ):
                continue
            cohort_year = _cohort_year(observation)
            if cohort_year is None:
                evidence_only_count += 1
                continue
            histories[observation.canonical_vehicle_id].append(
                _UsableObservation(observation, cohort_year)
            )

        estimator = EstimatedGenerationBuilder(self.registry_version)
        generations: list[GenerationEntry] = []
        for vehicle_id in sorted(histories):
            vehicle = vehicles.get(vehicle_id)
            if vehicle is None:
                raise ValueError("canonical observation vehicle is unavailable")
            history = histories[vehicle_id]
            generations.extend(
                estimator.build(
                    canonical_vehicle_id=vehicle_id,
                    market=vehicle.market,
                    observed_years=tuple(item.cohort_year for item in history),
                    evidence_ids=tuple(
                        sorted({item.observation.release_id for item in history})
                    ),
                )
            )
        generations.sort(key=lambda item: item.generation_id)
        _write_batches(repository.add_generations, generations)

        registry = GenerationRegistry(tuple(generations))
        resolver = GenerationResolver(registry, resolver_version=self.resolver_version)
        assignments: list[GenerationAssignment] = []
        assigned_count = 0
        unassigned_ids: list[str] = []
        for vehicle_id in sorted(histories):
            vehicle = vehicles[vehicle_id]
            for item in sorted(
                histories[vehicle_id],
                key=lambda candidate: candidate.observation.observation_id,
            ):
                try:
                    assignment = resolver.resolve(
                        ResolutionRequest(
                            observation_id=item.observation.observation_id,
                            canonical_vehicle_id=vehicle_id,
                            market=vehicle.market,
                            registration_cohort_year=item.cohort_year,
                            exact_generation_id=None,
                            descriptor_generation_ids=(),
                            launched_generation_ids=(),
                            reviewed_at=reviewed_at,
                        )
                    )
                except ValueError:
                    unassigned_ids.append(item.observation.observation_id)
                    continue
                assignments.append(assignment)
                assigned_count += 1
                if len(assignments) >= _BATCH_SIZE:
                    repository.add_generation_assignments(assignments)
                    assignments.clear()
        if assignments:
            repository.add_generation_assignments(assignments)
        usable_count = sum(len(history) for history in histories.values())
        return GenerationMappingResult(
            usable_count=usable_count,
            assigned_count=assigned_count,
            evidence_only_count=evidence_only_count,
            generation_count=len(generations),
            unassigned_ids=tuple(unassigned_ids),
        )


def _cohort_year(observation: Observation) -> int | None:
    if observation.registration_cohort_year is not None:
        return observation.registration_cohort_year
    if observation.manufacture_year is not None:
        return observation.manufacture_year
    if (
        observation.measure is Measure.NEW_REGISTRATIONS
        and observation.period_start.year == observation.period_end.year
    ):
        return observation.period_end.year
    return None


def _write_batches(write, records: list[GenerationEntry]) -> None:
    for start in range(0, len(records), _BATCH_SIZE):
        write(records[start : start + _BATCH_SIZE])

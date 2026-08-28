"""Materialize exact source, identity, generation, and forecastability coverage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from icor.domain.cohorts import CompletenessRecord
from icor.domain.evidence import MappingStatus, Measure
from icor.domain.generations import GenerationIdentityKind
from icor.evidence.normalization import stable_evidence_id

_UNUSABLE = frozenset(
    {MappingStatus.AMBIGUOUS, MappingStatus.REJECTED, MappingStatus.UNRESOLVED}
)


@dataclass(slots=True)
class _Counts:
    release_ids: set[str] = field(default_factory=set)
    observation_count: int = 0
    usable_count: int = 0
    assigned_count: int = 0
    vehicle_ids: set[str] = field(default_factory=set)
    sourced_generation_ids: set[str] = field(default_factory=set)
    estimated_generation_ids: set[str] = field(default_factory=set)
    forecastable_count: int = 0
    evidence_only_count: int = 0
    rejected_count: int = 0
    source_scope: bool = False


class CompletenessService:
    def materialize(self, repository) -> int:
        assignments = {
            item.observation_id: item for item in repository.list_generation_assignments()
        }
        generations = {
            item.generation_id: item for item in repository.list_generations()
        }
        counts: dict[tuple[str, int], _Counts] = defaultdict(_Counts)
        for observation in repository.list_observations():
            year = observation.registration_cohort_year or observation.period_end.year
            target = counts[observation.geography, year]
            target.release_ids.add(observation.release_id)
            target.observation_count += 1
            usable = (
                observation.canonical_vehicle_id is not None
                and observation.mapping_status not in _UNUSABLE
            )
            if not usable:
                continue
            target.usable_count += 1
            target.vehicle_ids.add(observation.canonical_vehicle_id)
            assignment = assignments.get(observation.observation_id)
            if assignment is None:
                target.evidence_only_count += 1
                continue
            target.assigned_count += 1
            generation = generations.get(assignment.selected_generation_id)
            if generation is None:
                raise ValueError("completeness generation is unavailable")
            if generation.identity_kind is GenerationIdentityKind.ESTIMATED:
                target.estimated_generation_ids.add(generation.generation_id)
            else:
                target.sourced_generation_ids.add(generation.generation_id)
            if observation.measure is Measure.NEW_REGISTRATIONS:
                target.forecastable_count += 1
            else:
                target.evidence_only_count += 1

        for release in repository.list_releases():
            if release.rejected_record_count == 0:
                continue
            target = counts[release.geography, release.coverage_end.year]
            target.release_ids.add(release.release_id)
            target.rejected_count += release.rejected_record_count
            target.source_scope = True

        records = tuple(
            CompletenessRecord(
                completeness_id=stable_evidence_id(
                    "completeness", geography, str(year)
                ),
                geography=geography,
                year=year,
                release_count=len(item.release_ids),
                observation_count=item.observation_count,
                usable_observation_count=item.usable_count,
                assigned_observation_count=item.assigned_count,
                canonical_family_count=len(item.vehicle_ids),
                sourced_generation_count=len(item.sourced_generation_ids),
                estimated_generation_count=len(item.estimated_generation_ids),
                forecastable_count=item.forecastable_count,
                evidence_only_count=item.evidence_only_count,
                rejected_record_count=item.rejected_count,
                reason_codes=(
                    "annual-observation-and-generation-completeness",
                    (
                        "source-scope-rejections-not-geographically-apportioned"
                        if item.source_scope
                        else "source-scope-rejections-reported-separately"
                    ),
                ),
            )
            for (geography, year), item in sorted(counts.items())
        )
        if not records:
            raise ValueError("completeness requires source or observation records")
        for start in range(0, len(records), 2_000):
            repository.add_completeness_records(records[start : start + 2_000])
        return len(records)

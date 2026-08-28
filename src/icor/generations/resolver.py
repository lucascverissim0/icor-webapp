"""Deterministic precedence for one hard generation assignment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from icor.domain.evidence import ConfidenceBand
from icor.domain.generations import (
    AssignmentMethod,
    GenerationAlternative,
    GenerationAssignment,
    GenerationEntry,
)
from icor.evidence.normalization import stable_evidence_id
from icor.generations.registry import GenerationRegistry


@dataclass(frozen=True, slots=True)
class ResolutionRequest:
    observation_id: str
    canonical_vehicle_id: str
    market: str
    registration_cohort_year: int
    exact_generation_id: str | None
    descriptor_generation_ids: tuple[str, ...]
    launched_generation_ids: tuple[str, ...]
    reviewed_at: datetime


class GenerationResolver:
    def __init__(
        self,
        registry: GenerationRegistry,
        *,
        resolver_version: str = "generation-resolver-v1",
    ) -> None:
        self.registry = registry
        self.resolver_version = resolver_version

    def resolve(self, request: ResolutionRequest) -> GenerationAssignment:
        candidates = self.registry.candidates(
            request.canonical_vehicle_id,
            request.market,
            request.registration_cohort_year,
        )
        if not candidates:
            raise ValueError("generation registry has no candidate for observation")

        if request.exact_generation_id is not None:
            selected = self._require_candidate(request.exact_generation_id, candidates)
            return self._assignment(
                request,
                selected,
                candidates,
                AssignmentMethod.EXACT_IDENTIFIER,
                ConfidenceBand.HIGH,
                Decimal("0.95"),
                "exact-identifier",
            )

        descriptor_matches = tuple(
            candidate
            for candidate in candidates
            if candidate.generation_id in request.descriptor_generation_ids
        )
        if len(descriptor_matches) == 1:
            return self._assignment(
                request,
                descriptor_matches[0],
                candidates,
                AssignmentMethod.DESCRIPTOR_OVERLAP,
                ConfidenceBand.MEDIUM,
                Decimal("0.70"),
                "descriptor-supported-overlap",
            )
        if len(candidates) == 1:
            return self._assignment(
                request,
                candidates[0],
                candidates,
                AssignmentMethod.UNIQUE_WINDOW,
                ConfidenceBand.HIGH,
                Decimal("0.85"),
                "unique-active-window",
            )

        coverage = {
            candidate.generation_id: self._active_months(
                candidate, request.registration_cohort_year
            )
            for candidate in candidates
        }
        maximum = max(coverage.values())
        leaders = tuple(
            candidate
            for candidate in candidates
            if coverage[candidate.generation_id] == maximum
        )
        if len(leaders) == 1:
            selected = leaders[0]
            method = AssignmentMethod.ACTIVE_MONTH_COVERAGE
            reason = "greatest-active-month-coverage"
        else:
            launched = tuple(
                candidate
                for candidate in leaders
                if candidate.generation_id in request.launched_generation_ids
            )
            pool = launched or leaders
            selected = max(
                pool,
                key=lambda candidate: (candidate.start_month, candidate.generation_id),
            )
            method = AssignmentMethod.NEWER_LAUNCH_TIEBREAK
            reason = "newer-launched-generation-tiebreak"
        ordered = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -coverage[candidate.generation_id],
                    -candidate.start_month.toordinal(),
                    candidate.generation_id,
                ),
            )
        )
        return self._assignment(
            request,
            selected,
            ordered,
            method,
            ConfidenceBand.LOW,
            Decimal("0.55"),
            reason,
        )

    @staticmethod
    def _require_candidate(
        generation_id: str, candidates: tuple[GenerationEntry, ...]
    ) -> GenerationEntry:
        selected = next(
            (item for item in candidates if item.generation_id == generation_id),
            None,
        )
        if selected is None:
            raise ValueError("exact generation is outside the active candidate set")
        return selected

    @staticmethod
    def _active_months(candidate: GenerationEntry, year: int) -> int:
        first = candidate.start_month.month if candidate.start_month.year == year else 1
        last = (
            candidate.end_month.month
            if candidate.end_month is not None and candidate.end_month.year == year
            else 12
        )
        return last - first + 1

    def _assignment(
        self,
        request: ResolutionRequest,
        selected: GenerationEntry,
        ordered_candidates: tuple[GenerationEntry, ...],
        method: AssignmentMethod,
        confidence: ConfidenceBand,
        weight: Decimal,
        reason: str,
    ) -> GenerationAssignment:
        alternatives = tuple(
            GenerationAlternative(item.generation_id, rank, reason)
            for rank, item in enumerate(
                (item for item in ordered_candidates if item != selected),
                start=2,
            )
        )
        return GenerationAssignment(
            assignment_id=stable_evidence_id(
                "generation-assignment",
                request.observation_id,
                selected.generation_id,
                self.resolver_version,
            ),
            observation_id=request.observation_id,
            selected_generation_id=selected.generation_id,
            alternatives=alternatives,
            method=method,
            evidence_ids=selected.evidence_ids,
            confidence=confidence,
            reason_codes=(reason,),
            training_weight=weight,
            resolver_version=self.resolver_version,
            registry_version=selected.registry_version,
            reviewed_at=request.reviewed_at,
        )

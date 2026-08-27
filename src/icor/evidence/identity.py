"""Conservative canonical identity attribution for source observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from icor.domain.evidence import (
    CanonicalVehicle,
    EvidenceConfidence,
    IdentityMapping,
    MappingStatus,
    Observation,
)
from icor.evidence.normalization import stable_evidence_id
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

_GENERIC_LABELS = frozenset(
    {
        "(not reported)",
        "not reported",
        "other",
        "others",
        "sonstige",
        "unknown",
        "unbekannt",
    }
)
_IDENTITY_CAP = 79


@dataclass(frozen=True, slots=True)
class ResolvedIdentity:
    observation: Observation
    vehicle: CanonicalVehicle | None
    mapping: IdentityMapping | None


class ExactNormalizedIdentityResolver:
    def resolve(
        self, observation: Observation, *, reviewed_at: datetime
    ) -> ResolvedIdentity:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        if observation.mapping_status is MappingStatus.REJECTED:
            return ResolvedIdentity(observation, None, None)
        normalized_make = observation.normalized_make
        normalized_model = observation.normalized_model
        if normalized_make is None or normalized_model is None:
            return ResolvedIdentity(observation, None, None)
        if normalized_make in _GENERIC_LABELS or normalized_model in _GENERIC_LABELS:
            rejected = replace(
                observation,
                canonical_vehicle_id=None,
                mapping_status=MappingStatus.REJECTED,
                validation_flags=tuple(
                    dict.fromkeys((*observation.validation_flags, "generic_vehicle_label"))
                ),
                evidence_confidence=_unpublished_confidence(observation.evidence_confidence),
            )
            return ResolvedIdentity(rejected, None, None)

        vehicle_id = stable_evidence_id(
            "vehicle-model", normalized_make, normalized_model, "europe"
        )
        vehicle = CanonicalVehicle(
            vehicle_id=vehicle_id,
            make=_display_label(observation.original_make),
            model=_display_label(observation.original_model),
            model_year=None,
            market="Europe",
        )
        attributed = replace(
            observation,
            canonical_vehicle_id=vehicle_id,
            mapping_status=MappingStatus.NORMALIZED_LABEL,
            evidence_confidence=_resolved_confidence(observation.evidence_confidence),
        )
        mapping = IdentityMapping(
            mapping_id=stable_evidence_id(
                "mapping", observation.observation_id, "exact-normalized-v1"
            ),
            observation_id=observation.observation_id,
            canonical_vehicle_id=vehicle_id,
            status=MappingStatus.NORMALIZED_LABEL,
            reason=(
                "Exact source make/model labels matched after conservative Unicode, "
                "case, and whitespace normalization; model year remains unknown."
            ),
            reviewed_at=reviewed_at,
        )
        return ResolvedIdentity(attributed, vehicle, mapping)


class IdentityAttributingRepository:
    def __init__(
        self,
        repository: SQLiteEvidenceRepository,
        resolver: ExactNormalizedIdentityResolver,
        *,
        reviewed_at: datetime,
    ) -> None:
        self._repository = repository
        self._resolver = resolver
        self._reviewed_at = reviewed_at

    def add_observations(self, observations: Sequence[Observation]) -> None:
        resolved = tuple(
            self._resolver.resolve(observation, reviewed_at=self._reviewed_at)
            for observation in observations
        )
        vehicles: dict[str, CanonicalVehicle] = {}
        mappings: list[IdentityMapping] = []
        for result in resolved:
            if result.vehicle is not None:
                vehicles.setdefault(result.vehicle.vehicle_id, result.vehicle)
            if result.mapping is not None:
                mappings.append(result.mapping)
        self._repository.add_identity_attributions(
            tuple(vehicles.values()),
            tuple(result.observation for result in resolved),
            tuple(mappings),
        )


def _display_label(value: str) -> str:
    return " ".join(value.split())


def _resolved_confidence(confidence: EvidenceConfidence) -> EvidenceConfidence:
    cap = _IDENTITY_CAP
    if confidence.applied_cap is not None:
        cap = min(cap, confidence.applied_cap)
    return replace(
        confidence,
        identity=10,
        reasons=tuple(
            dict.fromkeys(
                (
                    *confidence.reasons,
                    "Exact normalized make/model identity; model year is unavailable.",
                )
            )
        ),
        applied_cap=cap,
    )


def _unpublished_confidence(confidence: EvidenceConfidence) -> EvidenceConfidence:
    return replace(
        confidence,
        identity=0,
        reasons=tuple(
            dict.fromkeys(
                (*confidence.reasons, "Generic source label is excluded from publication.")
            )
        ),
    )

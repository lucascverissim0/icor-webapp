"""Vehicle-generation registry and deterministic assignment contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from re import fullmatch

from icor.domain.evidence import ConfidenceBand

_IDENTIFIER = r"[a-z0-9][a-z0-9._-]{0,79}"


class GenerationIdentityKind(StrEnum):
    MANUFACTURER_CONFIRMED = "manufacturer_confirmed"
    REGISTRY_CORROBORATED = "registry_corroborated"
    ESTIMATED = "estimated"


class AssignmentMethod(StrEnum):
    EXACT_IDENTIFIER = "exact_identifier"
    DESCRIPTOR_OVERLAP = "descriptor_overlap"
    UNIQUE_WINDOW = "unique_window"
    ACTIVE_MONTH_COVERAGE = "active_month_coverage"
    NEWER_LAUNCH_TIEBREAK = "newer_launch_tiebreak"
    ESTIMATED_GENERATION = "estimated_generation"


def _identifier(value: str, label: str) -> None:
    if type(value) is not str or fullmatch(_IDENTIFIER, value) is None:
        raise ValueError(f"{label} identifier is invalid")


def _text(value: str, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} is required")


def _text_tuple(value: tuple[str, ...], label: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} are required")
    for item in value:
        _text(item, label)


@dataclass(frozen=True, slots=True)
class GenerationEntry:
    generation_id: str
    canonical_vehicle_id: str
    display_name: str
    market: str
    start_month: date
    end_month: date | None
    identity_kind: GenerationIdentityKind
    body_style: str | None
    facelift: str | None
    platform: str | None
    evidence_ids: tuple[str, ...]
    dependency_groups: tuple[str, ...]
    confidence_reasons: tuple[str, ...]
    registry_version: str

    def __post_init__(self) -> None:
        _identifier(self.generation_id, "generation")
        _identifier(self.canonical_vehicle_id, "canonical vehicle")
        for value, label in (
            (self.display_name, "generation display name"),
            (self.market, "generation market"),
            (self.registry_version, "generation registry version"),
        ):
            _text(value, label)
        if type(self.start_month) is not date or self.start_month.day != 1:
            raise ValueError("generation start month must use first-day month precision")
        if self.end_month is not None:
            if type(self.end_month) is not date or self.end_month.day != 1:
                raise ValueError("generation end month must use first-day month precision")
            if self.end_month < self.start_month:
                raise ValueError("generation months must be ordered")
        if not isinstance(self.identity_kind, GenerationIdentityKind):
            raise ValueError("generation identity kind is unsupported")
        if self.identity_kind is GenerationIdentityKind.ESTIMATED and fullmatch(
            r"estimated-generation-[1-9][0-9]* \([12][0-9]{3}-[12][0-9]{3}\)",
            self.display_name,
        ) is None:
            raise ValueError("estimated generation label must not claim an official designation")
        for value, label in (
            (self.body_style, "body style"),
            (self.facelift, "facelift"),
            (self.platform, "platform"),
        ):
            if value is not None:
                _text(value, label)
        _text_tuple(self.evidence_ids, "generation evidence IDs")
        _text_tuple(self.dependency_groups, "generation dependency groups")
        _text_tuple(self.confidence_reasons, "generation confidence reasons")


@dataclass(frozen=True, slots=True)
class GenerationAlternative:
    generation_id: str
    rank: int
    loss_reason: str

    def __post_init__(self) -> None:
        _identifier(self.generation_id, "alternative generation")
        if type(self.rank) is not int or self.rank < 2:
            raise ValueError("alternative generation rank must be at least two")
        _text(self.loss_reason, "alternative loss reason")


@dataclass(frozen=True, slots=True)
class GenerationAssignment:
    assignment_id: str
    observation_id: str
    selected_generation_id: str
    alternatives: tuple[GenerationAlternative, ...]
    method: AssignmentMethod
    evidence_ids: tuple[str, ...]
    confidence: ConfidenceBand
    reason_codes: tuple[str, ...]
    training_weight: Decimal
    resolver_version: str
    registry_version: str
    reviewed_at: datetime

    def __post_init__(self) -> None:
        _identifier(self.assignment_id, "generation assignment")
        _identifier(self.observation_id, "observation")
        _identifier(self.selected_generation_id, "selected generation")
        if not isinstance(self.alternatives, tuple):
            raise ValueError("generation alternatives must be a tuple")
        alternative_ids = tuple(item.generation_id for item in self.alternatives)
        if self.selected_generation_id in alternative_ids:
            raise ValueError("selected generation cannot also be an alternative")
        if len(alternative_ids) != len(set(alternative_ids)):
            raise ValueError("generation alternatives must be unique")
        if tuple(item.rank for item in self.alternatives) != tuple(
            range(2, len(self.alternatives) + 2)
        ):
            raise ValueError("generation alternative ranks must be contiguous")
        if not isinstance(self.method, AssignmentMethod):
            raise ValueError("generation assignment method is unsupported")
        _text_tuple(self.evidence_ids, "generation assignment evidence IDs")
        if not isinstance(self.confidence, ConfidenceBand):
            raise ValueError("generation assignment confidence is unsupported")
        _text_tuple(self.reason_codes, "generation assignment reason codes")
        if (
            not isinstance(self.training_weight, Decimal)
            or not self.training_weight.is_finite()
            or not Decimal(0) <= self.training_weight <= Decimal(1)
        ):
            raise ValueError("generation training weight must be between zero and one")
        _text(self.resolver_version, "generation resolver version")
        _text(self.registry_version, "generation registry version")
        if (
            not isinstance(self.reviewed_at, datetime)
            or self.reviewed_at.tzinfo is None
            or self.reviewed_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("generation assignment review time must be UTC")

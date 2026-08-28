"""Generation-cohort, opportunity, and completeness contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from re import fullmatch

from icor.domain.evidence import ConfidenceBand

_IDENTIFIER = r"[a-z0-9][a-z0-9._-]{0,79}"


def _identifier(value: str, label: str) -> None:
    if type(value) is not str or fullmatch(_IDENTIFIER, value) is None:
        raise ValueError(f"{label} identifier is invalid")


def _text(value: str, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} is required")


def _identifier_tuple(value: tuple[str, ...], label: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} are required")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must be unique")
    for item in value:
        _identifier(item, label)


def _text_tuple(value: tuple[str, ...], label: str) -> None:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} are required")
    for item in value:
        _text(item, label)


def _year(value: int, label: str) -> None:
    if type(value) is not int or not 1900 <= value <= 2200:
        raise ValueError(f"{label} is invalid")


def _non_negative(value: Decimal, label: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(f"{label} must be a finite non-negative decimal")


def _ordered_interval(p10: Decimal, p50: Decimal, p90: Decimal, label: str) -> None:
    for value in (p10, p50, p90):
        _non_negative(value, label)
    if not p10 <= p50 <= p90:
        raise ValueError(f"{label} interval must be ordered")


@dataclass(frozen=True, slots=True)
class CohortEstimate:
    cohort_id: str
    generation_id: str
    canonical_vehicle_id: str
    geography: str
    registration_cohort_year: int
    as_of_year: int
    registrations: Decimal
    active_fleet_p10: Decimal
    active_fleet_p50: Decimal
    active_fleet_p90: Decimal
    input_observation_ids: tuple[str, ...]
    survival_method: str
    confidence: ConfidenceBand
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.cohort_id, "cohort")
        _identifier(self.generation_id, "generation")
        _identifier(self.canonical_vehicle_id, "canonical vehicle")
        _text(self.geography, "cohort geography")
        _year(self.registration_cohort_year, "registration cohort year")
        _year(self.as_of_year, "cohort as-of year")
        if self.as_of_year < self.registration_cohort_year:
            raise ValueError("cohort as-of year cannot precede registration cohort year")
        _non_negative(self.registrations, "cohort registrations")
        _ordered_interval(
            self.active_fleet_p10,
            self.active_fleet_p50,
            self.active_fleet_p90,
            "active fleet",
        )
        _identifier_tuple(self.input_observation_ids, "cohort input observation IDs")
        _text(self.survival_method, "cohort survival method")
        if not isinstance(self.confidence, ConfidenceBand):
            raise ValueError("cohort confidence is unsupported")
        _text_tuple(self.reason_codes, "cohort reason codes")


@dataclass(frozen=True, slots=True)
class OpportunityEstimate:
    opportunity_id: str
    generation_id: str
    canonical_vehicle_id: str
    geography: str
    horizon_year: int
    p10: Decimal
    p50: Decimal
    p90: Decimal
    active_fleet_p50: Decimal
    input_cohort_ids: tuple[str, ...]
    hazard_method: str
    forecast_method: str
    confidence: ConfidenceBand
    assumption_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.opportunity_id, "opportunity")
        _identifier(self.generation_id, "generation")
        _identifier(self.canonical_vehicle_id, "canonical vehicle")
        _text(self.geography, "opportunity geography")
        _year(self.horizon_year, "opportunity horizon year")
        _ordered_interval(self.p10, self.p50, self.p90, "opportunity")
        _non_negative(self.active_fleet_p50, "opportunity active fleet")
        _identifier_tuple(self.input_cohort_ids, "opportunity input cohort IDs")
        _text(self.hazard_method, "opportunity hazard method")
        _text(self.forecast_method, "opportunity forecast method")
        if not isinstance(self.confidence, ConfidenceBand):
            raise ValueError("opportunity confidence is unsupported")
        _identifier_tuple(self.assumption_ids, "opportunity assumption IDs")
        _text_tuple(self.reason_codes, "opportunity reason codes")


@dataclass(frozen=True, slots=True)
class CompletenessRecord:
    completeness_id: str
    geography: str
    year: int
    release_count: int
    observation_count: int
    usable_observation_count: int
    assigned_observation_count: int
    canonical_family_count: int
    sourced_generation_count: int
    estimated_generation_count: int
    forecastable_count: int
    evidence_only_count: int
    rejected_record_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.completeness_id, "completeness")
        _text(self.geography, "completeness geography")
        _year(self.year, "completeness year")
        for field_name in (
            "release_count",
            "observation_count",
            "usable_observation_count",
            "assigned_observation_count",
            "canonical_family_count",
            "sourced_generation_count",
            "estimated_generation_count",
            "forecastable_count",
            "evidence_only_count",
            "rejected_record_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name.replace('_', ' ')} must be non-negative")
        if self.usable_observation_count > self.observation_count:
            raise ValueError("usable observations cannot exceed observations")
        if self.assigned_observation_count > self.usable_observation_count:
            raise ValueError("assigned observations cannot exceed usable observations")
        _text_tuple(self.reason_codes, "completeness reason codes")

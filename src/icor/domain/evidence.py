"""Immutable source-evidence and published-value domain contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from re import fullmatch


class Measure(StrEnum):
    NEW_REGISTRATIONS = "new_registrations"
    ACTIVE_FLEET = "active_fleet"


class PublicationStatus(StrEnum):
    FINAL = "final"
    PROVISIONAL = "provisional"
    CORRECTED = "corrected"
    SUPERSEDED = "superseded"


class PeriodPrecision(StrEnum):
    DAY = "day"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class MappingStatus(StrEnum):
    EXACT_IDENTIFIER = "exact_identifier"
    CURATED_ALIAS = "curated_alias"
    NORMALIZED_LABEL = "normalized_label"
    REVIEWED_PROBABLE = "reviewed_probable"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class ValueStatus(StrEnum):
    OBSERVED = "observed"
    RECONCILED = "reconciled"
    ESTIMATED = "estimated"
    FORECAST = "forecast"


class ConfidenceBand(StrEnum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_IDENTIFIER_PATTERN = r"[a-z0-9][a-z0-9._-]{0,79}"
_SHA256_PATTERN = r"[0-9a-f]{64}"
_NON_PUBLISHABLE_STATUSES = frozenset(
    {MappingStatus.AMBIGUOUS, MappingStatus.REJECTED, MappingStatus.UNRESOLVED}
)


def _require_identifier(value: str, label: str) -> None:
    if type(value) is not str or fullmatch(_IDENTIFIER_PATTERN, value) is None:
        raise ValueError(f"{label} identifier is invalid")


def _require_text(value: str, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} is required")


def _require_utc(value: datetime, label: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be a UTC datetime")


def _require_date(value: date, label: str) -> None:
    if type(value) is not date:
        raise ValueError(f"{label} must be a date")


def _require_non_negative_int(value: int, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _require_enum(value: object, enum_type: type[StrEnum], label: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{label} is unsupported")


def _require_finite_decimal(value: object, label: str) -> None:
    if not isinstance(value, Decimal):
        raise ValueError(f"{label} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{label} must be finite")


@dataclass(frozen=True, slots=True)
class EvidenceConfidence:
    authority: int
    publication_status: int
    coverage: int
    identity: int
    independent_agreement: int
    reasons: tuple[str, ...]
    applied_cap: int | None = None

    def __post_init__(self) -> None:
        components = self.components
        maxima = {
            "authority": 25,
            "publication_status": 10,
            "coverage": 25,
            "identity": 20,
            "independent_agreement": 20,
        }
        for name, maximum in maxima.items():
            value = components[name]
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be an integer between 0 and {maximum}")
        if not isinstance(self.reasons, tuple) or not self.reasons or any(
            type(reason) is not str or not reason.strip() for reason in self.reasons
        ):
            raise ValueError("confidence reasons are required")
        if self.applied_cap is not None and (
            type(self.applied_cap) is not int or not 0 <= self.applied_cap <= 100
        ):
            raise ValueError("confidence cap must be an integer between 0 and 100")

    @property
    def components(self) -> dict[str, int]:
        return {
            "authority": self.authority,
            "publication_status": self.publication_status,
            "coverage": self.coverage,
            "identity": self.identity,
            "independent_agreement": self.independent_agreement,
        }

    @property
    def raw_total(self) -> int:
        return sum(self.components.values())

    @property
    def total(self) -> int:
        if self.applied_cap is None:
            return self.raw_total
        return min(self.raw_total, self.applied_cap)

    @property
    def band(self) -> ConfidenceBand:
        if self.total >= 80:
            return ConfidenceBand.HIGH
        if self.total >= 60:
            return ConfidenceBand.MEDIUM
        if self.total >= 40:
            return ConfidenceBand.LOW
        return ConfidenceBand.VERY_LOW


@dataclass(frozen=True, slots=True)
class ReleaseManifest:
    release_id: str
    source_id: str
    publisher: str
    source_url: str
    retrieved_at: datetime
    published_at: datetime
    coverage_start: date
    coverage_end: date
    geography: str
    geography_version: str
    measure: Measure
    unit: str
    publication_status: PublicationStatus
    dependency_group: str
    terms_url: str
    permitted_local_use: str
    artifact_path: str
    artifact_bytes: int
    sha256: str
    parser_name: str
    parser_version: str
    expected_schema: str
    raw_record_count: int
    accepted_record_count: int
    rejected_record_count: int
    quarantined_record_count: int

    def __post_init__(self) -> None:
        _require_identifier(self.release_id, "release")
        _require_identifier(self.source_id, "source")
        for value, label in (
            (self.publisher, "publisher"),
            (self.source_url, "source URL"),
            (self.geography, "geography"),
            (self.geography_version, "geography version"),
            (self.unit, "unit"),
            (self.dependency_group, "dependency group"),
            (self.terms_url, "terms URL"),
            (self.permitted_local_use, "permitted local use"),
            (self.artifact_path, "artifact path"),
            (self.parser_name, "parser name"),
            (self.parser_version, "parser version"),
            (self.expected_schema, "expected schema"),
        ):
            _require_text(value, label)
        _require_utc(self.retrieved_at, "retrieved_at")
        _require_utc(self.published_at, "published_at")
        _require_date(self.coverage_start, "coverage start")
        _require_date(self.coverage_end, "coverage end")
        if self.coverage_start > self.coverage_end:
            raise ValueError("coverage dates must be ordered")
        _require_enum(self.measure, Measure, "measure")
        _require_enum(self.publication_status, PublicationStatus, "publication status")
        if type(self.sha256) is not str or fullmatch(_SHA256_PATTERN, self.sha256) is None:
            raise ValueError("SHA-256 must be a lowercase 64-character hexadecimal digest")
        for value, label in (
            (self.artifact_bytes, "artifact bytes"),
            (self.raw_record_count, "raw record count"),
            (self.accepted_record_count, "accepted record count"),
            (self.rejected_record_count, "rejected record count"),
            (self.quarantined_record_count, "quarantined record count"),
        ):
            _require_non_negative_int(value, label)
        if (
            self.accepted_record_count + self.rejected_record_count + self.quarantined_record_count
            != self.raw_record_count
        ):
            raise ValueError("record counts must reconcile with the raw record count")


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    release_id: str
    original_row_locator: str
    geography: str
    geography_version: str
    period_start: date
    period_end: date
    period_precision: PeriodPrecision
    measure: Measure
    value: Decimal
    unit: str
    publication_status: PublicationStatus
    original_make: str
    original_model: str
    original_model_year: str | None
    original_type: str | None
    source_make_identifier: str | None
    source_model_identifier: str | None
    normalized_make: str | None
    normalized_model: str | None
    normalized_model_year: int | None
    canonical_vehicle_id: str | None
    mapping_status: MappingStatus
    transformation_notes: tuple[str, ...]
    validation_flags: tuple[str, ...]
    evidence_confidence: EvidenceConfidence

    def __post_init__(self) -> None:
        _require_identifier(self.observation_id, "observation")
        _require_identifier(self.release_id, "release")
        for value, label in (
            (self.original_row_locator, "original row locator"),
            (self.geography, "geography"),
            (self.geography_version, "geography version"),
            (self.unit, "unit"),
            (self.original_make, "original make"),
            (self.original_model, "original model"),
        ):
            _require_text(value, label)
        _require_date(self.period_start, "period start")
        _require_date(self.period_end, "period end")
        if self.period_start > self.period_end:
            raise ValueError("period dates must be ordered")
        _require_enum(self.period_precision, PeriodPrecision, "period precision")
        _require_enum(self.measure, Measure, "measure")
        _require_enum(self.publication_status, PublicationStatus, "publication status")
        _require_enum(self.mapping_status, MappingStatus, "mapping status")
        _require_finite_decimal(self.value, "observation value")
        if self.value < 0:
            raise ValueError("observation value must be non-negative")
        if (
            self.measure in {Measure.NEW_REGISTRATIONS, Measure.ACTIVE_FLEET}
            and self.value != self.value.to_integral_value()
        ):
            raise ValueError("count observation value must be an integer")
        if self.normalized_model_year is not None and type(self.normalized_model_year) is not int:
            raise ValueError("normalized model year must be an integer")
        if self.canonical_vehicle_id is not None:
            _require_identifier(self.canonical_vehicle_id, "canonical vehicle")
        if (
            self.mapping_status not in _NON_PUBLISHABLE_STATUSES
            and self.canonical_vehicle_id is None
        ):
            raise ValueError("resolved mapping requires a canonical vehicle")
        if not isinstance(self.transformation_notes, tuple) or not isinstance(
            self.validation_flags, tuple
        ):
            raise ValueError("observation notes and flags must be tuples")
        if any(type(note) is not str or not note.strip() for note in self.transformation_notes):
            raise ValueError("transformation notes must be nonblank")
        if any(type(flag) is not str or not flag.strip() for flag in self.validation_flags):
            raise ValueError("validation flags must be nonblank")
        if not isinstance(self.evidence_confidence, EvidenceConfidence):
            raise ValueError("evidence confidence is required")


@dataclass(frozen=True, slots=True)
class CanonicalVehicle:
    vehicle_id: str
    make: str
    model: str
    model_year: int | None
    market: str

    def __post_init__(self) -> None:
        _require_identifier(self.vehicle_id, "canonical vehicle")
        for value, label in (
            (self.make, "canonical vehicle make"),
            (self.model, "canonical vehicle model"),
            (self.market, "canonical vehicle market"),
        ):
            _require_text(value, label)
        if self.model_year is not None and type(self.model_year) is not int:
            raise ValueError("canonical vehicle model year must be an integer")


@dataclass(frozen=True, slots=True)
class IdentityMapping:
    mapping_id: str
    observation_id: str
    canonical_vehicle_id: str | None
    status: MappingStatus
    reason: str
    reviewed_at: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.mapping_id, "mapping")
        _require_identifier(self.observation_id, "observation")
        _require_enum(self.status, MappingStatus, "mapping status")
        _require_text(self.reason, "mapping reason")
        _require_utc(self.reviewed_at, "reviewed_at")
        if self.canonical_vehicle_id is not None:
            _require_identifier(self.canonical_vehicle_id, "canonical vehicle")
        if self.status not in _NON_PUBLISHABLE_STATUSES and self.canonical_vehicle_id is None:
            raise ValueError("resolved mapping requires a canonical vehicle")


@dataclass(frozen=True, slots=True)
class PublishedValue:
    value_id: str
    status: ValueStatus
    measure: Measure
    unit: str
    geography: str
    geography_version: str
    period_start: date
    period_end: date
    canonical_vehicle_id: str
    mapping_status: MappingStatus
    value: Decimal
    p10: Decimal | None
    p50: Decimal | None
    p90: Decimal | None
    input_ids: tuple[str, ...]
    method_version: str
    evidence_confidence: EvidenceConfidence
    forecast_confidence: int | None
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.value_id, "published value")
        _require_identifier(self.canonical_vehicle_id, "canonical vehicle")
        for value, label in (
            (self.unit, "unit"),
            (self.geography, "geography"),
            (self.geography_version, "geography version"),
            (self.method_version, "method version"),
        ):
            _require_text(value, label)
        _require_enum(self.status, ValueStatus, "value status")
        _require_enum(self.measure, Measure, "measure")
        _require_enum(self.mapping_status, MappingStatus, "mapping status")
        if self.mapping_status in _NON_PUBLISHABLE_STATUSES:
            raise ValueError(f"{self.mapping_status.value} mapping cannot publish a model value")
        _require_date(self.period_start, "period start")
        _require_date(self.period_end, "period end")
        if self.period_start > self.period_end:
            raise ValueError("period dates must be ordered")
        _require_finite_decimal(self.value, "published value")
        if self.value < 0:
            raise ValueError("published value must be non-negative")
        intervals = (self.p10, self.p50, self.p90)
        if any(interval is not None for interval in intervals):
            if any(not isinstance(interval, Decimal) for interval in intervals):
                raise ValueError("p10, p50, and p90 must be supplied together")
            if any(not interval.is_finite() for interval in intervals):  # type: ignore[union-attr]
                raise ValueError("published intervals must be finite")
            if not 0 <= self.p10 <= self.p50 <= self.p90:  # type: ignore[operator]
                raise ValueError("published intervals require p10 <= p50 <= p90")
        if not isinstance(self.input_ids, tuple) or not self.input_ids:
            raise ValueError("published value input IDs are required")
        for input_id in self.input_ids:
            _require_identifier(input_id, "input")
        if len(set(self.input_ids)) != len(self.input_ids):
            raise ValueError("published value input IDs must be unique")
        if not isinstance(self.evidence_confidence, EvidenceConfidence):
            raise ValueError("evidence confidence is required")
        if self.status is ValueStatus.FORECAST:
            if (
                type(self.forecast_confidence) is not int
                or not 0 <= self.forecast_confidence <= 100
            ):
                raise ValueError("forecast confidence is required for forecast values")
        elif self.forecast_confidence is not None:
            raise ValueError("forecast confidence is only permitted for forecast values")
        if not isinstance(self.warnings, tuple) or any(
            type(warning) is not str or not warning.strip() for warning in self.warnings
        ):
            raise ValueError("published warnings must be nonblank")

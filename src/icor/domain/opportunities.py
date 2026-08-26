"""Opportunity-ranking and production-coverage value semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from unicodedata import category

from icor.domain.planner import DemandRange


class CoverageMatchType(StrEnum):
    EXACT_CONFIGURATION = "exact_configuration"
    VEHICLE_YEAR_FALLBACK = "vehicle_year_fallback"


class CoverageStatus(StrEnum):
    EXACT_COVERED = "exact_covered"
    FALLBACK_ONLY = "fallback_only"
    MIXED = "mixed"
    UNCOVERED = "uncovered"


@dataclass(frozen=True, slots=True)
class ProductionCoverage:
    coverage_id: str
    match_type: CoverageMatchType
    configuration_id: str | None
    brand: str
    model: str
    model_year: int
    sku: str | None
    note: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        required = (self.coverage_id, self.brand, self.model)
        if any(not value.strip() for value in required) or type(self.model_year) is not int:
            raise ValueError("coverage identity is required")
        if self.match_type is CoverageMatchType.EXACT_CONFIGURATION:
            if self.configuration_id is None or not self.configuration_id.strip():
                raise ValueError("exact coverage requires a configuration identity")
        elif self.configuration_id is not None or self.sku is not None:
            raise ValueError("fallback coverage cannot claim a configuration or SKU")
        if self.note is not None and (
            len(self.note) > 500 or any(category(character) == "Cc" for character in self.note)
        ):
            raise ValueError("coverage note contains unsupported text")
        if not isinstance(self.created_at, datetime) or not isinstance(
            self.updated_at, datetime
        ):
            raise ValueError("coverage timestamps must be UTC datetimes")
        if (
            self.created_at.utcoffset() != timedelta(0)
            or self.updated_at.utcoffset() != timedelta(0)
            or self.updated_at < self.created_at
        ):
            raise ValueError("coverage timestamps must be UTC and ordered")


@dataclass(frozen=True, slots=True)
class OpportunityCandidate:
    group_id: str
    demand: DemandRange
    exact_covered_base_units: int
    fallback_covered_base_units: int
    uncovered_base_units: int

    def __post_init__(self) -> None:
        units = (
            self.exact_covered_base_units,
            self.fallback_covered_base_units,
            self.uncovered_base_units,
        )
        if not self.group_id.strip() or not all(
            type(value) is int and value >= 0 for value in units
        ):
            raise ValueError("opportunity coverage units require a canonical identity")
        if sum(units) != self.demand.base_units:
            raise ValueError("opportunity coverage units must reconcile with base demand")


@dataclass(frozen=True, slots=True)
class OpportunityScore:
    group_id: str
    demand_percentile: float
    demand_points: float
    readiness_ratio: float
    readiness_points: float
    total_points: float
    strategy_name: str
    strategy_version: str
    explanation: str

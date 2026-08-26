"""Production-coverage repository boundary and application errors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from icor.application.planner import PlannerRepository
from icor.domain.opportunities import CoverageMatchType, ProductionCoverage


class DuplicateCoverageError(RuntimeError):
    """Raised when an exact or fallback canonical identity already exists."""


class CoverageSchemaError(RuntimeError):
    """Raised when the local database schema cannot be used safely."""


class CanonicalCoverageError(ValueError):
    """Raised when requested coverage cannot resolve to forecast truth."""


class CoverageNotFoundError(LookupError):
    """Raised when a requested coverage identity does not exist."""


class CoverageRepository(Protocol):
    def list_all(self) -> tuple[ProductionCoverage, ...]: ...

    def get(self, coverage_id: str) -> ProductionCoverage | None: ...

    def create(self, coverage: ProductionCoverage) -> ProductionCoverage: ...

    def update(self, coverage: ProductionCoverage) -> ProductionCoverage | None: ...

    def delete(self, coverage_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class CreateCoverageCommand:
    match_type: CoverageMatchType
    configuration_id: str | None
    brand: str | None
    model: str | None
    model_year: int
    note: str | None


class ProductionCoverageService:
    def __init__(
        self,
        planner_repository: PlannerRepository,
        coverage_repository: CoverageRepository,
        *,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._planner_repository = planner_repository
        self._coverage_repository = coverage_repository
        self._clock = clock or (lambda: datetime.now(UTC))
        self._id_factory = id_factory or (lambda: uuid4().hex)

    def list_all(self) -> tuple[ProductionCoverage, ...]:
        return self._coverage_repository.list_all()

    def get(self, coverage_id: str) -> ProductionCoverage | None:
        return self._coverage_repository.get(coverage_id)

    def create(self, command: CreateCoverageCommand) -> ProductionCoverage:
        configuration_id, brand, model, sku = self._resolve(command)
        timestamp = self._clock()
        coverage = ProductionCoverage(
            coverage_id=self._id_factory(),
            match_type=command.match_type,
            configuration_id=configuration_id,
            brand=brand,
            model=model,
            model_year=command.model_year,
            sku=sku,
            note=command.note,
            created_at=timestamp,
            updated_at=timestamp,
        )
        return self._coverage_repository.create(coverage)

    def update(
        self, coverage_id: str, command: CreateCoverageCommand
    ) -> ProductionCoverage:
        existing = self._coverage_repository.get(coverage_id)
        if existing is None:
            raise CoverageNotFoundError("Production coverage was not found")
        configuration_id, brand, model, sku = self._resolve(command)
        coverage = ProductionCoverage(
            coverage_id=existing.coverage_id,
            match_type=command.match_type,
            configuration_id=configuration_id,
            brand=brand,
            model=model,
            model_year=command.model_year,
            sku=sku,
            note=command.note,
            created_at=existing.created_at,
            updated_at=self._clock(),
        )
        saved = self._coverage_repository.update(coverage)
        if saved is None:
            raise CoverageNotFoundError("Production coverage was not found")
        return saved

    def delete(self, coverage_id: str) -> None:
        if not self._coverage_repository.delete(coverage_id):
            raise CoverageNotFoundError("Production coverage was not found")

    def _resolve(
        self, command: CreateCoverageCommand
    ) -> tuple[str | None, str, str, str | None]:
        if type(command.model_year) is not int:
            raise CanonicalCoverageError("A canonical model year is required")
        if command.match_type is CoverageMatchType.EXACT_CONFIGURATION:
            if command.configuration_id is None:
                raise CanonicalCoverageError("A canonical exact configuration is required")
            configuration = self._planner_repository.get(command.configuration_id)
            if configuration is None or not any(
                row.model_year == command.model_year
                for row in configuration.model_year_demand
            ):
                raise CanonicalCoverageError(
                    "The exact configuration and model year are not canonical"
                )
            return (
                configuration.configuration_id,
                configuration.brand,
                configuration.model,
                configuration.sku,
            )
        if not command.brand or not command.model:
            raise CanonicalCoverageError("A canonical brand and model are required")
        canonical = any(
            configuration.brand == command.brand
            and configuration.model == command.model
            and any(
                row.model_year == command.model_year
                for row in configuration.model_year_demand
            )
            for configuration in self._planner_repository.list_all()
        )
        if not canonical:
            raise CanonicalCoverageError(
                "The fallback brand, model, and model year are not canonical"
            )
        return None, command.brand, command.model, None

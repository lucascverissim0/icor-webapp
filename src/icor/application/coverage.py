"""Production-coverage repository boundary and application errors."""

from __future__ import annotations

from typing import Protocol

from icor.domain.opportunities import ProductionCoverage


class DuplicateCoverageError(RuntimeError):
    """Raised when an exact or fallback canonical identity already exists."""


class CoverageSchemaError(RuntimeError):
    """Raised when the local database schema cannot be used safely."""


class CoverageRepository(Protocol):
    def list_all(self) -> tuple[ProductionCoverage, ...]: ...

    def get(self, coverage_id: str) -> ProductionCoverage | None: ...

    def create(self, coverage: ProductionCoverage) -> ProductionCoverage: ...

    def update(self, coverage: ProductionCoverage) -> ProductionCoverage | None: ...

    def delete(self, coverage_id: str) -> bool: ...

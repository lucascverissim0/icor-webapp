"""Versioned transactional SQLite adapter for shared local coverage state."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from icor.application.coverage import CoverageSchemaError, DuplicateCoverageError
from icor.domain.opportunities import CoverageMatchType, ProductionCoverage

SCHEMA_VERSION = 1


class SQLiteCoverageRepository:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._initialize()
        except sqlite3.DatabaseError as error:
            raise CoverageSchemaError("Coverage database cannot be read safely") from error

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            version_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
            ).fetchone()
            if version_table is None:
                with connection:
                    connection.execute(
                        "CREATE TABLE schema_version (version INTEGER NOT NULL)"
                    )
                    connection.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (SCHEMA_VERSION,),
                    )
                    self._create_schema(connection)
                return
            rows = connection.execute("SELECT version FROM schema_version").fetchall()
            if len(rows) != 1 or rows[0]["version"] != SCHEMA_VERSION:
                raise CoverageSchemaError("Coverage database uses an unsupported schema version")
            coverage_table = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'production_coverage'"
            ).fetchone()
            if coverage_table is None:
                raise CoverageSchemaError("Coverage database schema is incomplete")

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE production_coverage (
                coverage_id TEXT PRIMARY KEY,
                match_type TEXT NOT NULL CHECK (
                    match_type IN ('exact_configuration', 'vehicle_year_fallback')
                ),
                configuration_id TEXT,
                brand TEXT NOT NULL,
                model TEXT NOT NULL,
                model_year INTEGER NOT NULL,
                sku TEXT,
                note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (
                    (match_type = 'exact_configuration' AND configuration_id IS NOT NULL)
                    OR
                    (match_type = 'vehicle_year_fallback'
                        AND configuration_id IS NULL AND sku IS NULL)
                )
            )
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX production_coverage_exact_identity
            ON production_coverage (configuration_id, model_year)
            WHERE match_type = 'exact_configuration'
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX production_coverage_fallback_identity
            ON production_coverage (brand, model, model_year)
            WHERE match_type = 'vehicle_year_fallback'
            """
        )

    def list_all(self) -> tuple[ProductionCoverage, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM production_coverage ORDER BY coverage_id"
            ).fetchall()
        return tuple(_to_coverage(row) for row in rows)

    def get(self, coverage_id: str) -> ProductionCoverage | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM production_coverage WHERE coverage_id = ?",
                (coverage_id,),
            ).fetchone()
        return None if row is None else _to_coverage(row)

    def create(self, coverage: ProductionCoverage) -> ProductionCoverage:
        values = _coverage_values(coverage)
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO production_coverage (
                        coverage_id, match_type, configuration_id, brand, model,
                        model_year, sku, note, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        except sqlite3.IntegrityError as error:
            raise DuplicateCoverageError("Production coverage already exists") from error
        return coverage

    def update(self, coverage: ProductionCoverage) -> ProductionCoverage | None:
        try:
            with closing(self._connect()) as connection, connection:
                cursor = connection.execute(
                    """
                    UPDATE production_coverage SET
                        match_type = ?, configuration_id = ?, brand = ?, model = ?,
                        model_year = ?, sku = ?, note = ?, updated_at = ?
                    WHERE coverage_id = ?
                    """,
                    (
                        coverage.match_type.value,
                        coverage.configuration_id,
                        coverage.brand,
                        coverage.model,
                        coverage.model_year,
                        coverage.sku,
                        coverage.note,
                        coverage.updated_at.isoformat(),
                        coverage.coverage_id,
                    ),
                )
                if cursor.rowcount == 0:
                    return None
        except sqlite3.IntegrityError as error:
            raise DuplicateCoverageError("Production coverage already exists") from error
        return self.get(coverage.coverage_id)

    def delete(self, coverage_id: str) -> bool:
        with closing(self._connect()) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM production_coverage WHERE coverage_id = ?",
                (coverage_id,),
            )
        return cursor.rowcount > 0


def _coverage_values(coverage: ProductionCoverage) -> tuple[object, ...]:
    return (
        coverage.coverage_id,
        coverage.match_type.value,
        coverage.configuration_id,
        coverage.brand,
        coverage.model,
        coverage.model_year,
        coverage.sku,
        coverage.note,
        coverage.created_at.isoformat(),
        coverage.updated_at.isoformat(),
    )


def _to_coverage(row: sqlite3.Row) -> ProductionCoverage:
    return ProductionCoverage(
        coverage_id=row["coverage_id"],
        match_type=CoverageMatchType(row["match_type"]),
        configuration_id=row["configuration_id"],
        brand=row["brand"],
        model=row["model"],
        model_year=row["model_year"],
        sku=row["sku"],
        note=row["note"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )

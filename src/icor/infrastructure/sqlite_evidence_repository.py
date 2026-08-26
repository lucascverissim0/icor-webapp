"""SQLite v1 implementation of the append-only evidence ledger."""
# ruff: noqa: E501

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar

from icor.domain.evidence import (
    CanonicalVehicle,
    EvidenceConfidence,
    IdentityMapping,
    MappingStatus,
    Measure,
    Observation,
    PeriodPrecision,
    PublicationStatus,
    PublishedValue,
    ReleaseManifest,
    ValueStatus,
)
from icor.domain.snapshots import SnapshotManifest, SnapshotStatus, SnapshotVersions
from icor.evidence.serialization import canonical_json_bytes


class EvidenceSchemaError(RuntimeError):
    """The ledger schema is unsupported or has been corrupted."""


class DuplicateEvidenceError(RuntimeError):
    """An immutable evidence identity or natural key already exists."""


class ImmutableEvidenceError(RuntimeError):
    """A write would mutate the ledger or refer to unavailable evidence."""


_SCHEMA_VERSION = 1
_NON_PUBLISHABLE_STATUSES = frozenset(
    {MappingStatus.AMBIGUOUS.value, MappingStatus.REJECTED.value, MappingStatus.UNRESOLVED.value}
)
_T = TypeVar("_T")


class SQLiteEvidenceRepository:
    """A versioned, immutable evidence ledger backed by SQLite."""

    def __init__(self, path: Path, writable: bool = False) -> None:
        self.path = Path(path)
        self.writable = writable
        if writable:
            self._prepare_writable_schema()
        else:
            if not self.path.is_file():
                raise EvidenceSchemaError("read-only evidence database does not exist")
            with self._connect() as connection:
                self._validate_schema(connection)

    @property
    def schema_version(self) -> int:
        with self._connect() as connection:
            return self._validate_schema(connection)

    def add_release(self, release: ReleaseManifest) -> None:
        self._write(lambda connection: self._insert_release(connection, release))

    def add_observations(self, observations: Sequence[Observation]) -> None:
        self._write(lambda connection: self._insert_observations(connection, observations))

    def add_vehicle(self, vehicle: CanonicalVehicle) -> None:
        self._write(lambda connection: self._insert_vehicle(connection, vehicle))

    def add_mapping(self, mapping: IdentityMapping) -> None:
        self._write(lambda connection: self._insert_mapping(connection, mapping))

    def add_published_values(self, values: Sequence[PublishedValue]) -> None:
        self._write(lambda connection: self._insert_published_values(connection, values))

    def add_snapshot(self, snapshot: SnapshotManifest) -> None:
        self._write(lambda connection: self._insert_snapshot(connection, snapshot))

    def get_release(self, release_id: str) -> ReleaseManifest | None:
        return self._get_one("SELECT * FROM source_release WHERE release_id = ?", (release_id,), self._release)

    def get_observation(self, observation_id: str) -> Observation | None:
        return self._get_one("SELECT * FROM observation WHERE observation_id = ?", (observation_id,), self._observation)

    def get_vehicle(self, vehicle_id: str) -> CanonicalVehicle | None:
        return self._get_one("SELECT * FROM canonical_vehicle WHERE vehicle_id = ?", (vehicle_id,), self._vehicle)

    def get_mapping(self, mapping_id: str) -> IdentityMapping | None:
        return self._get_one("SELECT * FROM identity_mapping WHERE mapping_id = ?", (mapping_id,), self._mapping)

    def get_published_value(self, value_id: str) -> PublishedValue | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM published_value WHERE value_id = ?", (value_id,)).fetchone()
            return None if row is None else self._published_value(connection, row)

    def get_snapshot(self, snapshot_id: str) -> SnapshotManifest | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM snapshot WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
            return None if row is None else self._snapshot(connection, row)

    def list_releases(self) -> tuple[ReleaseManifest, ...]:
        return self._list("SELECT * FROM source_release ORDER BY release_id", self._release)

    def list_observations(self) -> tuple[Observation, ...]:
        return self._list("SELECT * FROM observation ORDER BY observation_id", self._observation)

    def list_vehicles(self) -> tuple[CanonicalVehicle, ...]:
        return self._list("SELECT * FROM canonical_vehicle ORDER BY vehicle_id", self._vehicle)

    def list_mappings(self) -> tuple[IdentityMapping, ...]:
        return self._list("SELECT * FROM identity_mapping ORDER BY mapping_id", self._mapping)

    def list_published_values(self) -> tuple[PublishedValue, ...]:
        with self._connect() as connection:
            return tuple(
                self._published_value(connection, row)
                for row in connection.execute("SELECT * FROM published_value ORDER BY value_id")
            )

    def list_snapshots(self) -> tuple[SnapshotManifest, ...]:
        with self._connect() as connection:
            return tuple(
                self._snapshot(connection, row)
                for row in connection.execute("SELECT * FROM snapshot ORDER BY snapshot_id")
            )

    def _prepare_writable_schema(self) -> None:
        existing = self.path.exists() and self.path.stat().st_size > 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            has_version_table = self._table_exists(connection, "schema_version")
            if not existing and not has_version_table:
                self._migrate_v1(connection)
            else:
                self._validate_schema(connection)

    @contextmanager
    def _connect(self) -> Any:
        if self.writable:
            connection = sqlite3.connect(self.path)
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        else:
            connection = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
            connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _write(self, operation: Callable[[sqlite3.Connection], None]) -> None:
        if not self.writable:
            raise ImmutableEvidenceError("evidence repository is read-only")
        try:
            with self._connect() as connection, connection:
                operation(connection)
        except sqlite3.IntegrityError as error:
            raise self._integrity_error(error) from error

    def _migrate_v1(self, connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in self._migration_statements():
                connection.execute(statement)
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()

    def _migration_statements(self) -> tuple[str, ...]:
        checks = {
            "measure": self._enum_check(Measure),
            "publication_status": self._enum_check(PublicationStatus),
            "period_precision": self._enum_check(PeriodPrecision),
            "mapping_status": self._enum_check(MappingStatus),
            "value_status": self._enum_check(ValueStatus),
            "snapshot_status": self._enum_check(SnapshotStatus),
        }
        return tuple(
            statement.strip()
            for statement in f"""
                CREATE TABLE schema_version (version INTEGER NOT NULL CHECK (version = 1));
                INSERT INTO schema_version (version) VALUES (1);
                CREATE TABLE source_release (
                    release_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, publisher TEXT NOT NULL,
                    source_url TEXT NOT NULL, retrieved_at TEXT NOT NULL, published_at TEXT NOT NULL,
                    coverage_start TEXT NOT NULL, coverage_end TEXT NOT NULL, geography TEXT NOT NULL,
                    geography_version TEXT NOT NULL, measure TEXT NOT NULL CHECK (measure IN {checks['measure']}),
                    unit TEXT NOT NULL, publication_status TEXT NOT NULL CHECK (publication_status IN {checks['publication_status']}),
                    dependency_group TEXT NOT NULL, terms_url TEXT NOT NULL, permitted_local_use TEXT NOT NULL,
                    artifact_path TEXT NOT NULL, artifact_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
                    parser_name TEXT NOT NULL, parser_version TEXT NOT NULL, expected_schema TEXT NOT NULL,
                    raw_record_count INTEGER NOT NULL, accepted_record_count INTEGER NOT NULL,
                    rejected_record_count INTEGER NOT NULL, quarantined_record_count INTEGER NOT NULL
                );
                CREATE TABLE canonical_vehicle (
                    vehicle_id TEXT PRIMARY KEY, make TEXT NOT NULL, model TEXT NOT NULL,
                    model_year INTEGER NOT NULL, market TEXT NOT NULL
                );
                CREATE TABLE observation (
                    observation_id TEXT PRIMARY KEY, release_id TEXT NOT NULL REFERENCES source_release(release_id),
                    original_row_locator TEXT NOT NULL, geography TEXT NOT NULL, geography_version TEXT NOT NULL,
                    period_start TEXT NOT NULL, period_end TEXT NOT NULL,
                    period_precision TEXT NOT NULL CHECK (period_precision IN {checks['period_precision']}),
                    measure TEXT NOT NULL CHECK (measure IN {checks['measure']}), value TEXT NOT NULL, unit TEXT NOT NULL,
                    publication_status TEXT NOT NULL CHECK (publication_status IN {checks['publication_status']}),
                    original_make TEXT NOT NULL, original_model TEXT NOT NULL, original_model_year TEXT,
                    original_type TEXT, source_make_identifier TEXT, source_model_identifier TEXT,
                    normalized_make TEXT, normalized_model TEXT, normalized_model_year INTEGER,
                    canonical_vehicle_id TEXT REFERENCES canonical_vehicle(vehicle_id),
                    mapping_status TEXT NOT NULL CHECK (mapping_status IN {checks['mapping_status']}),
                    transformation_notes TEXT NOT NULL, validation_flags TEXT NOT NULL,
                    confidence_authority INTEGER NOT NULL, confidence_publication_status INTEGER NOT NULL,
                    confidence_coverage INTEGER NOT NULL, confidence_identity INTEGER NOT NULL,
                    confidence_independent_agreement INTEGER NOT NULL, confidence_reasons TEXT NOT NULL,
                    confidence_applied_cap INTEGER
                );
                CREATE TABLE identity_mapping (
                    mapping_id TEXT PRIMARY KEY, observation_id TEXT NOT NULL REFERENCES observation(observation_id),
                    canonical_vehicle_id TEXT REFERENCES canonical_vehicle(vehicle_id),
                    status TEXT NOT NULL CHECK (status IN {checks['mapping_status']}), reason TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL
                );
                CREATE TABLE published_value (
                    value_id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK (status IN {checks['value_status']}),
                    measure TEXT NOT NULL CHECK (measure IN {checks['measure']}), unit TEXT NOT NULL,
                    geography TEXT NOT NULL, geography_version TEXT NOT NULL, period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL, canonical_vehicle_id TEXT NOT NULL REFERENCES canonical_vehicle(vehicle_id),
                    mapping_status TEXT NOT NULL CHECK (mapping_status IN {checks['mapping_status']}), value TEXT NOT NULL,
                    p10 TEXT, p50 TEXT, p90 TEXT, method_version TEXT NOT NULL,
                    confidence_authority INTEGER NOT NULL, confidence_publication_status INTEGER NOT NULL,
                    confidence_coverage INTEGER NOT NULL, confidence_identity INTEGER NOT NULL,
                    confidence_independent_agreement INTEGER NOT NULL, confidence_reasons TEXT NOT NULL,
                    confidence_applied_cap INTEGER, forecast_confidence INTEGER, warnings TEXT NOT NULL
                );
                CREATE TABLE published_value_input (
                    value_id TEXT NOT NULL REFERENCES published_value(value_id),
                    observation_id TEXT NOT NULL REFERENCES observation(observation_id), input_position INTEGER NOT NULL,
                    PRIMARY KEY (value_id, observation_id), UNIQUE (value_id, input_position)
                );
                CREATE TABLE snapshot (
                    snapshot_id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK (status IN {checks['snapshot_status']}),
                    built_at TEXT NOT NULL, deterministic_seed INTEGER NOT NULL, versions TEXT NOT NULL,
                    database_sha256 TEXT NOT NULL, observation_count INTEGER NOT NULL,
                    published_value_count INTEGER NOT NULL, warnings TEXT NOT NULL
                );
                CREATE TABLE snapshot_release (
                    snapshot_id TEXT NOT NULL REFERENCES snapshot(snapshot_id),
                    release_id TEXT NOT NULL REFERENCES source_release(release_id),
                    release_position INTEGER NOT NULL,
                    PRIMARY KEY (snapshot_id, release_position), UNIQUE (snapshot_id, release_id)
                );
                CREATE UNIQUE INDEX canonical_vehicle_identity_idx
                ON canonical_vehicle (make, model, model_year, market);
                CREATE UNIQUE INDEX observation_release_row_locator_idx
                ON observation (release_id, original_row_locator);
                """.split(";")
            if statement.strip()
        )

    @staticmethod
    def _enum_check(enum_type: type[Any]) -> str:
        return "(" + ", ".join(repr(member.value) for member in enum_type) + ")"

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone() is not None

    def _validate_schema(self, connection: sqlite3.Connection) -> int:
        if not self._table_exists(connection, "schema_version"):
            raise EvidenceSchemaError("schema version table is missing")
        try:
            rows = connection.execute("SELECT version FROM schema_version").fetchall()
        except sqlite3.DatabaseError as error:
            raise EvidenceSchemaError("schema version table is corrupt") from error
        if len(rows) != 1 or type(rows[0]["version"]) is not int:
            raise EvidenceSchemaError("schema version table is corrupt")
        version = rows[0]["version"]
        if version > _SCHEMA_VERSION:
            raise EvidenceSchemaError("evidence schema is newer than this application")
        if version != _SCHEMA_VERSION:
            raise EvidenceSchemaError("schema version is unsupported")
        self._validate_v1_structure(connection)
        return version

    def _validate_v1_structure(self, connection: sqlite3.Connection) -> None:
        expected_tables, expected_indexes = self._v1_schema_contract()
        actual_tables = {
            row["name"]: self._normalize_schema_sql(row["sql"])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        actual_indexes = {
            row["name"]: self._normalize_schema_sql(row["sql"])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL"
            )
        }
        if actual_tables != expected_tables or actual_indexes != expected_indexes:
            raise EvidenceSchemaError("schema is structurally invalid")

    def _v1_schema_contract(self) -> tuple[dict[str, str], dict[str, str]]:
        tables: dict[str, str] = {}
        indexes: dict[str, str] = {}
        for statement in self._migration_statements():
            tokens = statement.split()
            if statement.startswith("CREATE TABLE"):
                tables[tokens[2]] = self._normalize_schema_sql(statement)
            elif statement.startswith("CREATE") and "INDEX" in tokens:
                indexes[tokens[tokens.index("INDEX") + 1]] = self._normalize_schema_sql(statement)
        return tables, indexes

    @staticmethod
    def _normalize_schema_sql(statement: str) -> str:
        tokens: list[str] = []
        index = 0
        while index < len(statement):
            character = statement[index]
            if character.isspace():
                index += 1
                continue
            if character in {"'", '"', "`"}:
                quote = character
                start = index
                index += 1
                while index < len(statement):
                    if statement[index] != quote:
                        index += 1
                        continue
                    if index + 1 < len(statement) and statement[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                tokens.append(statement[start:index])
                continue
            if character.isalnum() or character in {"_", "$"}:
                start = index
                index += 1
                while index < len(statement) and (
                    statement[index].isalnum() or statement[index] in {"_", "$"}
                ):
                    index += 1
                tokens.append(statement[start:index].lower())
                continue
            tokens.append(character.lower())
            index += 1
        if tokens and tokens[-1] == ";":
            tokens.pop()
        return " ".join(tokens)

    def _insert_release(self, connection: sqlite3.Connection, release: ReleaseManifest) -> None:
        connection.execute(
            """INSERT INTO source_release VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                release.release_id, release.source_id, release.publisher, release.source_url,
                self._datetime(release.retrieved_at), self._datetime(release.published_at),
                self._date(release.coverage_start), self._date(release.coverage_end), release.geography,
                release.geography_version, release.measure.value, release.unit,
                release.publication_status.value, release.dependency_group, release.terms_url,
                release.permitted_local_use, release.artifact_path, release.artifact_bytes, release.sha256,
                release.parser_name, release.parser_version, release.expected_schema, release.raw_record_count,
                release.accepted_record_count, release.rejected_record_count, release.quarantined_record_count,
            ),
        )

    def _insert_observations(
        self, connection: sqlite3.Connection, observations: Sequence[Observation]
    ) -> None:
        for observation in observations:
            self._require_reference(connection, "source_release", "release_id", observation.release_id)
            if observation.canonical_vehicle_id is not None:
                self._require_reference(
                    connection, "canonical_vehicle", "vehicle_id", observation.canonical_vehicle_id
                )
            connection.execute(
                """INSERT INTO observation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    observation.observation_id, observation.release_id, observation.original_row_locator,
                    observation.geography, observation.geography_version, self._date(observation.period_start),
                    self._date(observation.period_end), observation.period_precision.value,
                    observation.measure.value, str(observation.value), observation.unit,
                    observation.publication_status.value, observation.original_make, observation.original_model,
                    observation.original_model_year, observation.original_type,
                    observation.source_make_identifier, observation.source_model_identifier,
                    observation.normalized_make, observation.normalized_model,
                    observation.normalized_model_year, observation.canonical_vehicle_id,
                    observation.mapping_status.value, self._json(observation.transformation_notes),
                    self._json(observation.validation_flags), *self._confidence(observation.evidence_confidence),
                ),
            )

    def _insert_vehicle(self, connection: sqlite3.Connection, vehicle: CanonicalVehicle) -> None:
        connection.execute(
            "INSERT INTO canonical_vehicle VALUES (?, ?, ?, ?, ?)",
            (vehicle.vehicle_id, vehicle.make, vehicle.model, vehicle.model_year, vehicle.market),
        )

    def _insert_mapping(self, connection: sqlite3.Connection, mapping: IdentityMapping) -> None:
        self._require_reference(connection, "observation", "observation_id", mapping.observation_id)
        if mapping.canonical_vehicle_id is not None:
            self._require_reference(
                connection, "canonical_vehicle", "vehicle_id", mapping.canonical_vehicle_id
            )
        connection.execute(
            "INSERT INTO identity_mapping VALUES (?, ?, ?, ?, ?, ?)",
            (
                mapping.mapping_id, mapping.observation_id, mapping.canonical_vehicle_id,
                mapping.status.value, mapping.reason, self._datetime(mapping.reviewed_at),
            ),
        )

    def _insert_published_values(
        self, connection: sqlite3.Connection, values: Sequence[PublishedValue]
    ) -> None:
        for value in values:
            self._require_reference(connection, "canonical_vehicle", "vehicle_id", value.canonical_vehicle_id)
            for input_id in value.input_ids:
                row = connection.execute(
                    """SELECT canonical_vehicle_id, mapping_status, measure, unit, geography,
                    geography_version, period_start, period_end FROM observation
                    WHERE observation_id = ?""",
                    (input_id,),
                ).fetchone()
                if row is None:
                    raise ImmutableEvidenceError(f"evidence reference does not exist: {input_id}")
                if row["mapping_status"] in _NON_PUBLISHABLE_STATUSES:
                    raise ImmutableEvidenceError("unresolved observation cannot publish a model value")
                if (
                    row["canonical_vehicle_id"] != value.canonical_vehicle_id
                    or row["measure"] != value.measure.value
                    or row["unit"] != value.unit
                    or row["geography"] != value.geography
                    or row["geography_version"] != value.geography_version
                    or row["period_start"] != self._date(value.period_start)
                    or row["period_end"] != self._date(value.period_end)
                ):
                    raise ImmutableEvidenceError("input observation is semantically incompatible")
            connection.execute(
                """INSERT INTO published_value VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    value.value_id, value.status.value, value.measure.value, value.unit, value.geography,
                    value.geography_version, self._date(value.period_start), self._date(value.period_end),
                    value.canonical_vehicle_id, value.mapping_status.value, str(value.value),
                    self._decimal(value.p10), self._decimal(value.p50), self._decimal(value.p90),
                    value.method_version, *self._confidence(value.evidence_confidence),
                    value.forecast_confidence, self._json(value.warnings),
                ),
            )
            for position, input_id in enumerate(value.input_ids):
                connection.execute(
                    "INSERT INTO published_value_input VALUES (?, ?, ?)",
                    (value.value_id, input_id, position),
                )

    def _insert_snapshot(self, connection: sqlite3.Connection, snapshot: SnapshotManifest) -> None:
        for release_id in snapshot.release_ids:
            self._require_reference(connection, "source_release", "release_id", release_id)
        connection.execute(
            "INSERT INTO snapshot VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.snapshot_id, snapshot.status.value, self._datetime(snapshot.built_at),
                snapshot.deterministic_seed, self._json(snapshot.versions),
                snapshot.database_sha256, snapshot.observation_count, snapshot.published_value_count,
                self._json(snapshot.warnings),
            ),
        )
        for position, release_id in enumerate(snapshot.release_ids):
            connection.execute(
                "INSERT INTO snapshot_release VALUES (?, ?, ?)",
                (snapshot.snapshot_id, release_id, position),
            )

    @staticmethod
    def _require_reference(connection: sqlite3.Connection, table: str, column: str, value: str) -> None:
        if connection.execute(f"SELECT 1 FROM {table} WHERE {column} = ?", (value,)).fetchone() is None:
            raise ImmutableEvidenceError(f"evidence reference does not exist: {value}")

    @staticmethod
    def _integrity_error(error: sqlite3.IntegrityError) -> DuplicateEvidenceError:
        message = str(error)
        if "observation.release_id, observation.original_row_locator" in message:
            return DuplicateEvidenceError("observation source row locator already exists")
        return DuplicateEvidenceError(f"immutable evidence already exists: {message}")

    def _get_one(
        self, query: str, parameters: tuple[str], decode: Callable[[sqlite3.Row], _T]
    ) -> _T | None:
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
            return None if row is None else decode(row)

    def _list(self, query: str, decode: Callable[[sqlite3.Row], _T]) -> tuple[_T, ...]:
        with self._connect() as connection:
            return tuple(decode(row) for row in connection.execute(query))

    @staticmethod
    def _date(value: date) -> str:
        return value.isoformat()

    @staticmethod
    def _datetime(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def _decimal(value: Decimal | None) -> str | None:
        return None if value is None else str(value)

    @staticmethod
    def _json(value: object) -> str:
        return canonical_json_bytes(value).decode("utf-8")

    @staticmethod
    def _confidence(value: EvidenceConfidence) -> tuple[Any, ...]:
        return (
            value.authority, value.publication_status, value.coverage, value.identity,
            value.independent_agreement, SQLiteEvidenceRepository._json(value.reasons), value.applied_cap,
        )

    @staticmethod
    def _confidence_from(row: sqlite3.Row) -> EvidenceConfidence:
        return EvidenceConfidence(
            row["confidence_authority"], row["confidence_publication_status"],
            row["confidence_coverage"], row["confidence_identity"],
            row["confidence_independent_agreement"], tuple(json.loads(row["confidence_reasons"])),
            row["confidence_applied_cap"],
        )

    @staticmethod
    def _release(row: sqlite3.Row) -> ReleaseManifest:
        return ReleaseManifest(
            release_id=row["release_id"], source_id=row["source_id"], publisher=row["publisher"],
            source_url=row["source_url"], retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
            published_at=datetime.fromisoformat(row["published_at"]),
            coverage_start=date.fromisoformat(row["coverage_start"]), coverage_end=date.fromisoformat(row["coverage_end"]),
            geography=row["geography"], geography_version=row["geography_version"],
            measure=Measure(row["measure"]), unit=row["unit"],
            publication_status=PublicationStatus(row["publication_status"]),
            dependency_group=row["dependency_group"], terms_url=row["terms_url"],
            permitted_local_use=row["permitted_local_use"], artifact_path=row["artifact_path"],
            artifact_bytes=row["artifact_bytes"], sha256=row["sha256"], parser_name=row["parser_name"],
            parser_version=row["parser_version"], expected_schema=row["expected_schema"],
            raw_record_count=row["raw_record_count"], accepted_record_count=row["accepted_record_count"],
            rejected_record_count=row["rejected_record_count"], quarantined_record_count=row["quarantined_record_count"],
        )

    @classmethod
    def _observation(cls, row: sqlite3.Row) -> Observation:
        return Observation(
            observation_id=row["observation_id"], release_id=row["release_id"],
            original_row_locator=row["original_row_locator"], geography=row["geography"],
            geography_version=row["geography_version"], period_start=date.fromisoformat(row["period_start"]),
            period_end=date.fromisoformat(row["period_end"]),
            period_precision=PeriodPrecision(row["period_precision"]), measure=Measure(row["measure"]),
            value=Decimal(row["value"]), unit=row["unit"],
            publication_status=PublicationStatus(row["publication_status"]),
            original_make=row["original_make"], original_model=row["original_model"],
            original_model_year=row["original_model_year"], original_type=row["original_type"],
            source_make_identifier=row["source_make_identifier"], source_model_identifier=row["source_model_identifier"],
            normalized_make=row["normalized_make"], normalized_model=row["normalized_model"],
            normalized_model_year=row["normalized_model_year"], canonical_vehicle_id=row["canonical_vehicle_id"],
            mapping_status=MappingStatus(row["mapping_status"]),
            transformation_notes=tuple(json.loads(row["transformation_notes"])),
            validation_flags=tuple(json.loads(row["validation_flags"])),
            evidence_confidence=cls._confidence_from(row),
        )

    @staticmethod
    def _vehicle(row: sqlite3.Row) -> CanonicalVehicle:
        return CanonicalVehicle(row["vehicle_id"], row["make"], row["model"], row["model_year"], row["market"])

    @staticmethod
    def _mapping(row: sqlite3.Row) -> IdentityMapping:
        return IdentityMapping(
            mapping_id=row["mapping_id"], observation_id=row["observation_id"],
            canonical_vehicle_id=row["canonical_vehicle_id"], status=MappingStatus(row["status"]),
            reason=row["reason"], reviewed_at=datetime.fromisoformat(row["reviewed_at"]),
        )

    @classmethod
    def _published_value(cls, connection: sqlite3.Connection, row: sqlite3.Row) -> PublishedValue:
        inputs = connection.execute(
            "SELECT observation_id FROM published_value_input WHERE value_id = ? ORDER BY input_position",
            (row["value_id"],),
        )
        return PublishedValue(
            value_id=row["value_id"], status=ValueStatus(row["status"]), measure=Measure(row["measure"]),
            unit=row["unit"], geography=row["geography"], geography_version=row["geography_version"],
            period_start=date.fromisoformat(row["period_start"]), period_end=date.fromisoformat(row["period_end"]),
            canonical_vehicle_id=row["canonical_vehicle_id"], mapping_status=MappingStatus(row["mapping_status"]),
            value=Decimal(row["value"]), p10=cls._load_decimal(row["p10"]),
            p50=cls._load_decimal(row["p50"]), p90=cls._load_decimal(row["p90"]),
            input_ids=tuple(input_row["observation_id"] for input_row in inputs),
            method_version=row["method_version"], evidence_confidence=cls._confidence_from(row),
            forecast_confidence=row["forecast_confidence"], warnings=tuple(json.loads(row["warnings"])),
        )

    @staticmethod
    def _load_decimal(value: str | None) -> Decimal | None:
        return None if value is None else Decimal(value)

    @classmethod
    def _snapshot(cls, connection: sqlite3.Connection, row: sqlite3.Row) -> SnapshotManifest:
        versions = SnapshotVersions(**json.loads(row["versions"]))
        release_ids = tuple(
            member["release_id"]
            for member in connection.execute(
                """SELECT release_id FROM snapshot_release WHERE snapshot_id = ?
                ORDER BY release_position""",
                (row["snapshot_id"],),
            )
        )
        return SnapshotManifest(
            snapshot_id=row["snapshot_id"], status=SnapshotStatus(row["status"]),
            built_at=datetime.fromisoformat(row["built_at"]), deterministic_seed=row["deterministic_seed"],
            release_ids=release_ids, versions=versions,
            database_sha256=row["database_sha256"], observation_count=row["observation_count"],
            published_value_count=row["published_value_count"], warnings=tuple(json.loads(row["warnings"])),
        )

"""Read-only audit of whether a snapshot can support promotion-grade backtests."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

_MAX_NORMAL_PUBLICATION_LAG_DAYS = 550
_MINIMUM_TEMPORAL_ORIGINS = 3


@dataclass(frozen=True, slots=True)
class SnapshotReadiness:
    snapshot_id: str
    annual_release_years: tuple[int, ...]
    origin_eligible_release_years: tuple[int, ...]
    observed_registration_rows: int
    assigned_registration_rows: int
    assignment_confidence_counts: tuple[tuple[str, int], ...]
    interpolated_cohort_rows: int
    forecast_cohort_rows: int
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "ready_for_promotion_benchmark": self.ready,
            "annual_release_years": list(self.annual_release_years),
            "origin_eligible_release_years": list(self.origin_eligible_release_years),
            "minimum_temporal_origins": _MINIMUM_TEMPORAL_ORIGINS,
            "observed_registration_rows": self.observed_registration_rows,
            "assigned_registration_rows": self.assigned_registration_rows,
            "assignment_confidence_counts": dict(self.assignment_confidence_counts),
            "interpolated_cohort_rows": self.interpolated_cohort_rows,
            "forecast_cohort_rows": self.forecast_cohort_rows,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


def audit_snapshot_readiness(database_path: Path, snapshot_id: str) -> SnapshotReadiness:
    """Inspect immutable ledger metadata without reading materialized values as truth."""

    resolved = database_path.resolve()
    if not resolved.is_file():
        raise ValueError("snapshot database is unavailable")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    try:
        release_rows = connection.execute(
            """SELECT coverage_start, coverage_end, published_at
            FROM source_release
            WHERE measure = 'new_registrations'
            ORDER BY coverage_end, release_id"""
        ).fetchall()
        annual_years, eligible_years = _release_years(release_rows)
        observed_rows, assigned_rows = connection.execute(
            """SELECT COUNT(*), COUNT(assignment.observation_id)
            FROM observation
            LEFT JOIN generation_assignment assignment
              ON assignment.observation_id = observation.observation_id
            WHERE observation.measure = 'new_registrations'
              AND observation.canonical_vehicle_id IS NOT NULL"""
        ).fetchone()
        confidence_counts = tuple(
            (str(confidence), int(count))
            for confidence, count in connection.execute(
                """SELECT assignment.confidence, COUNT(*)
                FROM observation
                JOIN generation_assignment assignment
                  ON assignment.observation_id = observation.observation_id
                WHERE observation.measure = 'new_registrations'
                  AND observation.canonical_vehicle_id IS NOT NULL
                GROUP BY assignment.confidence
                ORDER BY assignment.confidence"""
            )
        )
        interpolated, forecast = connection.execute(
            """SELECT
              SUM(CASE WHEN reason_codes LIKE '%estimated-registration-cohort%' THEN 1 ELSE 0 END),
              SUM(CASE WHEN reason_codes LIKE '%forecast-registration-cohort%' THEN 1 ELSE 0 END)
            FROM cohort_estimate"""
        ).fetchone()
    except sqlite3.Error as error:
        raise ValueError(
            "snapshot database does not expose the forecasting audit schema"
        ) from error
    finally:
        connection.close()

    blockers: list[str] = []
    if len(eligible_years) < _MINIMUM_TEMPORAL_ORIGINS:
        blockers.append("insufficient_as_of_publication_vintages")
    if observed_rows == 0:
        blockers.append("no_observed_registration_targets")
    if assigned_rows < observed_rows:
        blockers.append("observed_registration_rows_without_generation_assignment")
    warnings = (
        ("materialized_cohorts_include_interpolation",)
        if int(interpolated or 0) > 0
        else ()
    )
    return SnapshotReadiness(
        snapshot_id=snapshot_id,
        annual_release_years=annual_years,
        origin_eligible_release_years=eligible_years,
        observed_registration_rows=int(observed_rows),
        assigned_registration_rows=int(assigned_rows),
        assignment_confidence_counts=confidence_counts,
        interpolated_cohort_rows=int(interpolated or 0),
        forecast_cohort_rows=int(forecast or 0),
        blockers=tuple(blockers),
        warnings=warnings,
    )


def _release_years(
    rows: list[tuple[str, str, str]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    annual: set[int] = set()
    eligible: set[int] = set()
    for coverage_start_text, coverage_end_text, published_at_text in rows:
        coverage_start = date.fromisoformat(coverage_start_text)
        coverage_end = date.fromisoformat(coverage_end_text)
        published_at = datetime.fromisoformat(published_at_text).date()
        if coverage_start.year != coverage_end.year:
            continue
        annual.add(coverage_end.year)
        lag = (published_at - coverage_end).days
        if 0 <= lag <= _MAX_NORMAL_PUBLICATION_LAG_DAYS:
            eligible.add(coverage_end.year)
    return tuple(sorted(annual)), tuple(sorted(eligible))

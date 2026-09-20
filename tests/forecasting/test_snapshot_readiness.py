from __future__ import annotations

import sqlite3
from pathlib import Path

from icor.forecasting.snapshot_readiness import audit_snapshot_readiness


def database(tmp_path: Path, *, vintages: int = 3, interpolated: bool = False) -> Path:
    path = tmp_path / "evidence.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE source_release (
              release_id TEXT, measure TEXT, coverage_start TEXT,
              coverage_end TEXT, published_at TEXT
            );
            CREATE TABLE observation (
              observation_id TEXT, measure TEXT, canonical_vehicle_id TEXT
            );
            CREATE TABLE generation_assignment (
              observation_id TEXT, confidence TEXT
            );
            CREATE TABLE cohort_estimate (reason_codes TEXT);
            """
        )
        for position in range(vintages):
            year = 2020 + position
            connection.execute(
                "INSERT INTO source_release VALUES (?, 'new_registrations', ?, ?, ?)",
                (
                    f"release-{year}",
                    f"{year}-01-01",
                    f"{year}-12-31",
                    f"{year + 1}-06-30T00:00:00+00:00",
                ),
            )
        connection.execute(
            "INSERT INTO observation VALUES ('observation-1', 'new_registrations', 'vehicle-1')"
        )
        connection.execute(
            "INSERT INTO generation_assignment VALUES ('observation-1', 'high')"
        )
        connection.execute(
            "INSERT INTO cohort_estimate VALUES (?)",
            (
                '["estimated-registration-cohort"]'
                if interpolated
                else '["observed-registration-cohort"]',
            ),
        )
    return path


def test_complete_release_vintages_and_observed_rows_are_ready(tmp_path: Path) -> None:
    report = audit_snapshot_readiness(database(tmp_path), "snapshot-new")

    assert report.ready
    assert report.origin_eligible_release_years == (2020, 2021, 2022)
    assert report.assignment_confidence_counts == (("high", 1),)


def test_backfills_and_interpolated_cohorts_fail_closed(tmp_path: Path) -> None:
    path = database(tmp_path, vintages=1, interpolated=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """UPDATE source_release SET published_at = '2026-08-01T00:00:00+00:00'
            WHERE release_id = 'release-2020'"""
        )

    report = audit_snapshot_readiness(path, "snapshot-old")

    assert not report.ready
    assert report.origin_eligible_release_years == ()
    assert report.blockers == (
        "insufficient_as_of_publication_vintages",
    )
    assert report.warnings == ("materialized_cohorts_include_interpolation",)

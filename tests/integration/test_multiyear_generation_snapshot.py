from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from icor.evidence.release_manifests import load_snapshot_manifest
from icor.evidence.serialization import sha256_file
from icor.evidence.validation import SnapshotValidator
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository


def _candidate() -> Path:
    value = os.getenv("ICOR_REAL_SNAPSHOT")
    if not value:
        pytest.skip("ICOR_REAL_SNAPSHOT is not configured")
    return Path(value)


def test_real_snapshot_has_complete_multiyear_generation_products() -> None:
    candidate = _candidate()
    manifest = load_snapshot_manifest(candidate / "snapshot.json")
    repository = SQLiteEvidenceRepository(candidate / "evidence.sqlite3")
    eea_years = {
        release.coverage_end.year
        for release in repository.list_releases()
        if release.source_id == "eea-co2-monitoring"
    }

    assert eea_years == set(range(2010, 2025))
    assert len(manifest.release_ids) == 20
    assert not manifest.warnings
    assert SnapshotValidator().validate(repository, manifest).can_promote

    connection = sqlite3.connect(candidate / "evidence.sqlite3")
    try:
        missing = connection.execute(
            """SELECT COUNT(*) FROM observation
            LEFT JOIN generation_assignment USING (observation_id)
            WHERE observation.canonical_vehicle_id IS NOT NULL
            AND observation.mapping_status NOT IN ('ambiguous','rejected','unresolved')
            AND (observation.registration_cohort_year IS NOT NULL
                OR observation.manufacture_year IS NOT NULL
                OR (observation.measure = 'new_registrations'
                    AND SUBSTR(observation.period_start,1,4)
                        = SUBSTR(observation.period_end,1,4)))
            AND generation_assignment.assignment_id IS NULL"""
        ).fetchone()[0]
        horizons = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT horizon_year FROM opportunity_estimate"
            )
        }
        completeness_count = connection.execute(
            "SELECT COUNT(*) FROM completeness_record"
        ).fetchone()[0]
    finally:
        connection.close()

    assert missing == 0
    assert horizons == {2028, 2031}
    assert completeness_count > 0


def test_real_snapshot_database_digest_matches_manifest() -> None:
    candidate = _candidate()
    manifest = load_snapshot_manifest(candidate / "snapshot.json")

    assert sha256_file(candidate / "evidence.sqlite3") == manifest.database_sha256

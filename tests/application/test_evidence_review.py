from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from icor.application.evidence_review import EvidenceObservationQuery, EvidenceReviewService
from icor.domain.evidence import (
    EvidenceConfidence,
    MappingStatus,
    Measure,
    Observation,
    PeriodPrecision,
    PublicationStatus,
    ReleaseManifest,
)
from icor.domain.snapshots import SnapshotManifest, SnapshotStatus, SnapshotVersions
from icor.evidence.serialization import canonical_json_bytes, sha256_file
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository


def _release(release_id: str, source_id: str, measure: Measure, rows: int) -> ReleaseManifest:
    return ReleaseManifest(
        release_id=release_id,
        source_id=source_id,
        publisher=f"{source_id.upper()} publisher",
        source_url=f"https://example.test/{source_id}",
        retrieved_at=datetime(2026, 8, 27, tzinfo=UTC),
        published_at=datetime(2026, 4, 29, tzinfo=UTC),
        coverage_start=date(2024, 1, 1),
        coverage_end=date(2025, 12, 31),
        geography="GB" if source_id == "dft" else "EEA",
        geography_version="official-v1",
        measure=measure,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group=f"{source_id}-register",
        terms_url="https://example.test/terms",
        permitted_local_use="Attribution required.",
        artifact_path="artifact.csv",
        artifact_bytes=10,
        sha256="a" * 64,
        parser_name=f"{source_id}_v1",
        parser_version="v1",
        expected_schema="fixture-v1",
        raw_record_count=rows,
        accepted_record_count=rows,
        rejected_record_count=0,
        quarantined_record_count=0,
    )


def _observation(
    identifier: str,
    release: ReleaseManifest,
    make: str,
    model: str,
    value: int,
    *,
    status: MappingStatus = MappingStatus.UNRESOLVED,
) -> Observation:
    return Observation(
        observation_id=identifier,
        release_id=release.release_id,
        original_row_locator=f"row:{identifier}",
        geography="GB" if release.source_id == "dft" else "DE",
        geography_version=release.geography_version,
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        period_precision=PeriodPrecision.YEAR,
        measure=release.measure,
        value=Decimal(value),
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        original_make=make,
        original_model=model,
        original_model_year=None,
        original_type="publisher type",
        source_make_identifier=make,
        source_model_identifier=model,
        normalized_make=make.casefold(),
        normalized_model=model.casefold(),
        normalized_model_year=None,
        canonical_vehicle_id=None,
        mapping_status=status,
        transformation_notes=("Preserved publisher label.",),
        validation_flags=() if status is MappingStatus.UNRESOLVED else ("aggregate_label",),
        evidence_confidence=EvidenceConfidence(
            25, 10, 25, 0, 10, ("Official source; identity unresolved.",)
        ),
    )


@pytest.fixture
def candidate(tmp_path: Path) -> Path:
    candidate_path = tmp_path / "snapshot-review"
    candidate_path.mkdir()
    database = candidate_path / "evidence.sqlite3"
    repository = SQLiteEvidenceRepository(database, writable=True)
    eea = _release("eea-final", "eea", Measure.NEW_REGISTRATIONS, 2)
    dft = _release("dft-fleet", "dft", Measure.ACTIVE_FLEET, 1)
    repository.add_release(eea)
    repository.add_release(dft)
    repository.add_observations(
        (
            _observation("obs-eea-golf", eea, "Volkswagen", "Golf", 12),
            _observation("obs-eea-percent", eea, "Percent% Motors", "Under_score", 3),
            _observation(
                "obs-dft-other",
                dft,
                "OTHER",
                "(not reported)",
                5,
                status=MappingStatus.REJECTED,
            ),
        )
    )
    manifest = SnapshotManifest(
        snapshot_id="snapshot-review",
        status=SnapshotStatus.CANDIDATE,
        built_at=datetime(2026, 8, 27, 12, tzinfo=UTC),
        deterministic_seed=7,
        release_ids=("dft-fleet", "eea-final"),
        versions=SnapshotVersions(*("review-v1",) * 8),
        database_sha256=sha256_file(database),
        observation_count=3,
        published_value_count=0,
        warnings=(),
    )
    (candidate_path / "snapshot.json").write_bytes(canonical_json_bytes(manifest))
    return candidate_path


def test_summary_reconciles_releases_and_mapping_statuses(candidate: Path) -> None:
    summary = EvidenceReviewService.from_candidate(candidate).summary()

    assert summary.snapshot_id == "snapshot-review"
    assert summary.observation_count == 3
    assert summary.published_value_count == 0
    assert [
        (row.release_id, row.observation_count, row.total_value) for row in summary.releases
    ] == [
        ("dft-fleet", 1, Decimal("5")),
        ("eea-final", 2, Decimal("15")),
    ]
    assert summary.mapping_status_counts == {"rejected": 1, "unresolved": 2}


def test_observations_filter_and_paginate_deterministically(candidate: Path) -> None:
    service = EvidenceReviewService.from_candidate(candidate)

    page = service.list_observations(
        EvidenceObservationQuery(release_id="eea-final", search="golf", page=1, page_size=1)
    )

    assert page.total == 1
    assert page.pages == 1
    assert page.items[0].observation_id == "obs-eea-golf"
    assert page.items[0].original_make == "Volkswagen"
    assert page.items[0].mapping_status == "unresolved"
    assert page.items[0].confidence_total == 70


@pytest.mark.parametrize(
    ("search", "expected"), [("%", "obs-eea-percent"), ("_", "obs-eea-percent")]
)
def test_search_treats_sql_wildcards_as_literal_text(
    candidate: Path, search: str, expected: str
) -> None:
    page = EvidenceReviewService.from_candidate(candidate).list_observations(
        EvidenceObservationQuery(search=search, page=1, page_size=10)
    )

    assert [row.observation_id for row in page.items] == [expected]


def test_candidate_checksum_mismatch_is_rejected(candidate: Path) -> None:
    database = candidate / "evidence.sqlite3"
    database.write_bytes(database.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        EvidenceReviewService.from_candidate(candidate)


def test_candidate_manifest_counts_must_match_database(candidate: Path) -> None:
    manifest_path = candidate / "snapshot.json"
    from icor.evidence.release_manifests import load_snapshot_manifest

    manifest = load_snapshot_manifest(manifest_path)
    manifest_path.write_bytes(canonical_json_bytes(replace(manifest, observation_count=4)))

    with pytest.raises(ValueError, match="count"):
        EvidenceReviewService.from_candidate(candidate)

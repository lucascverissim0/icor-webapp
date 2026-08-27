from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from icor.application.registrations import (
    RegistrationQuery,
    RegistrationService,
    RegistrationUnavailableError,
)
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
from icor.evidence.identity import (
    ExactNormalizedIdentityResolver,
    IdentityAttributingRepository,
)
from icor.evidence.serialization import canonical_json_bytes, sha256_file
from icor.infrastructure.snapshot_store import SnapshotStore
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

BUILD_AS_OF = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _release(
    release_id: str,
    source_id: str,
    parser_name: str,
    geography: str,
    count: int,
) -> ReleaseManifest:
    return ReleaseManifest(
        release_id=release_id,
        source_id=source_id,
        publisher="Official publisher",
        source_url="https://example.test/source",
        retrieved_at=BUILD_AS_OF,
        published_at=datetime(2026, 6, 25, tzinfo=UTC),
        coverage_start=date(2024, 1, 1),
        coverage_end=date(2024, 12, 31),
        geography=geography,
        geography_version="official-2024-v1",
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="national-registers",
        terms_url="https://example.test/terms",
        permitted_local_use="Attribution required.",
        artifact_path="artifact.bin",
        artifact_bytes=1,
        sha256="a" * 64,
        parser_name=parser_name,
        parser_version="v1",
        expected_schema="v1",
        raw_record_count=count,
        accepted_record_count=count,
        rejected_record_count=0,
        quarantined_record_count=0,
    )


def _observation(
    observation_id: str,
    release_id: str,
    geography: str,
    make: str,
    model: str,
    value: str,
) -> Observation:
    return Observation(
        observation_id=observation_id,
        release_id=release_id,
        original_row_locator=f"row:{observation_id}",
        geography=geography,
        geography_version="official-2024-v1",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
        period_precision=PeriodPrecision.YEAR,
        measure=Measure.NEW_REGISTRATIONS,
        value=Decimal(value),
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        original_make=make,
        original_model=model,
        original_model_year=None,
        original_type="passenger car",
        source_make_identifier=make,
        source_model_identifier=model,
        normalized_make=make.casefold(),
        normalized_model=model.casefold(),
        normalized_model_year=None,
        canonical_vehicle_id=None,
        mapping_status=MappingStatus.UNRESOLVED,
        transformation_notes=("Official row retained.",),
        validation_flags=(),
        evidence_confidence=EvidenceConfidence(
            25, 10, 25, 0, 10, ("Official finalized evidence.",)
        ),
    )


@pytest.fixture
def mapped_candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "snapshot-test-registrations"
    candidate.mkdir()
    database = candidate / "evidence.sqlite3"
    repository = SQLiteEvidenceRepository(database, writable=True)
    eea = _release(
        "eea-co2cars-2024-final-v30-r1",
        "eea-co2-monitoring",
        "eea_co2_cars_zip_v1",
        "EEA reporting countries",
        4,
    )
    kba = _release(
        "kba-fz10-2024-12-final-v3",
        "kba-fz10",
        "kba_fz10_xlsx_v1",
        "DE",
        1,
    )
    repository.add_release(eea)
    repository.add_release(kba)
    attributing = IdentityAttributingRepository(
        repository,
        ExactNormalizedIdentityResolver(),
        reviewed_at=BUILD_AS_OF,
    )
    attributing.add_observations(
        (
            _observation("obs-eea-de-alpha", eea.release_id, "DE", "Example Motors", "Alpha", "10"),
            _observation("obs-eea-fr-alpha", eea.release_id, "FR", "Example Motors", "Alpha", "5"),
            _observation(
                "obs-eea-no-alpha", eea.release_id, "NO", "Example Motors", "Alpha", "100"
            ),
            _observation("obs-eea-de-beta", eea.release_id, "DE", "Example Motors", "Beta", "5"),
            _observation(
                "obs-kba-de-alpha", kba.release_id, "DE", "Example Motors", "Alpha", "999"
            ),
        )
    )
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("VACUUM")
    manifest = SnapshotManifest(
        snapshot_id=candidate.name,
        status=SnapshotStatus.CANDIDATE,
        built_at=BUILD_AS_OF,
        deterministic_seed=20260827,
        release_ids=tuple(sorted((eea.release_id, kba.release_id))),
        versions=SnapshotVersions(
            source_registry="official-sources-v1",
            identity_registry="exact-normalized-model-family-v1",
            reconciliation_method="not-applied-v1",
            confidence_method="source-evidence-v1",
            estimation_method="not-applied-v1",
            survival_method="not-applied-v1",
            hazard_method="not-applied-v1",
            forecast_method="not-applied-v1",
        ),
        database_sha256=sha256_file(database),
        observation_count=5,
        published_value_count=0,
        warnings=(),
    )
    (candidate / "snapshot.json").write_bytes(canonical_json_bytes(manifest))
    return candidate


def test_eu27_ranking_sums_only_final_eea_member_observations(
    mapped_candidate: Path,
) -> None:
    page = RegistrationService.from_candidate(mapped_candidate).ranking(
        RegistrationQuery(geography="EU27", year=2024, page=1, page_size=25)
    )

    assert page.total == 2
    assert page.total_registrations == Decimal("20")
    assert page.items[0].make == "Example Motors"
    assert page.items[0].model == "Alpha"
    assert page.items[0].registrations == Decimal("15")
    assert page.items[0].rank == 1
    assert page.items[0].status == "derived_observed"
    assert page.items[0].model_year is None
    assert page.items[0].source_ids == ("eea-co2-monitoring",)
    assert page.items[0].input_observation_count == 2
    assert page.items[0].evidence_confidence == 79


def test_eu27_ranking_excludes_kba_and_non_member_rows(mapped_candidate: Path) -> None:
    page = RegistrationService.from_candidate(mapped_candidate).ranking(
        RegistrationQuery()
    )

    assert page.total_registrations == Decimal("20")
    assert all("kba-fz10" not in row.source_ids for row in page.items)


def test_ranking_is_stable_paginated_and_searchable(mapped_candidate: Path) -> None:
    service = RegistrationService.from_candidate(mapped_candidate)

    first = service.ranking(RegistrationQuery(page=1, page_size=1))
    second = service.ranking(RegistrationQuery(page=2, page_size=1))
    escaped = service.ranking(RegistrationQuery(search="%_"))

    assert first.items[0].model == "Alpha"
    assert second.items[0].model == "Beta"
    assert second.items[0].rank == 2
    assert escaped.total == 0
    assert escaped.items == ()


def test_ranking_aggregates_observations_before_vehicle_lookup(
    mapped_candidate: Path,
) -> None:
    service = RegistrationService.from_candidate(mapped_candidate)

    sql, _ = service._grouped_query(None)

    assert "JOIN identity_mapping" not in sql
    assert sql.index("GROUP BY o.canonical_vehicle_id") < sql.index(
        "JOIN canonical_vehicle"
    )
    assert "o.mapping_status = 'normalized_label'" in sql


def test_summary_exposes_snapshot_and_truthful_scope(mapped_candidate: Path) -> None:
    summary = RegistrationService.from_candidate(mapped_candidate).summary()

    assert summary.snapshot_id == mapped_candidate.name
    assert summary.status == "candidate"
    assert summary.geographies == ("EU27",)
    assert summary.years == (2024,)
    assert summary.total_registrations == Decimal("20")
    assert summary.model_count == 2
    assert summary.model_year_available is False
    assert summary.release_ids == ("eea-co2cars-2024-final-v30-r1",)


@pytest.mark.parametrize(
    "query",
    (
        RegistrationQuery(geography="WORLD"),
        RegistrationQuery(year=2025),
        RegistrationQuery(page=0),
        RegistrationQuery(page_size=101),
        RegistrationQuery(search="x" * 101),
    ),
)
def test_query_rejects_unsupported_or_unbounded_inputs(query: RegistrationQuery) -> None:
    with pytest.raises(ValueError):
        query.validate()


def test_candidate_checksum_tampering_is_typed_unavailable(mapped_candidate: Path) -> None:
    database = mapped_candidate / "evidence.sqlite3"
    database.write_bytes(database.read_bytes() + b"tampered")

    with pytest.raises(RegistrationUnavailableError, match="unavailable"):
        RegistrationService.from_candidate(mapped_candidate)


def test_unresolved_identity_registry_is_typed_unavailable(mapped_candidate: Path) -> None:
    manifest_path = mapped_candidate / "snapshot.json"
    payload = manifest_path.read_bytes()
    assert b"exact-normalized-model-family-v1" in payload
    manifest_path.write_bytes(
        payload.replace(
            b"exact-normalized-model-family-v1", b"unresolved-source-labels-v1"
        )
    )

    with pytest.raises(RegistrationUnavailableError, match="unavailable"):
        RegistrationService.from_candidate(mapped_candidate)


def test_active_snapshot_configures_registration_service(
    mapped_candidate: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_service = RegistrationService.from_candidate(mapped_candidate)
    repository = SQLiteEvidenceRepository(candidate_service.database_path)
    monkeypatch.setattr(
        SnapshotStore,
        "open_active_snapshot",
        lambda self: (candidate_service.manifest, repository),
    )

    service = RegistrationService.from_active(mapped_candidate.parent)

    assert service.database_path == candidate_service.database_path
    assert service.summary().total_registrations == Decimal("20")


def test_active_snapshot_with_unresolved_registry_fails_closed(
    mapped_candidate: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_service = RegistrationService.from_candidate(mapped_candidate)
    unresolved = SnapshotManifest(
        snapshot_id=candidate_service.manifest.snapshot_id,
        status=candidate_service.manifest.status,
        built_at=candidate_service.manifest.built_at,
        deterministic_seed=candidate_service.manifest.deterministic_seed,
        release_ids=candidate_service.manifest.release_ids,
        versions=SnapshotVersions(
            source_registry="official-sources-v1",
            identity_registry="unresolved-source-labels-v1",
            reconciliation_method="not-applied-v1",
            confidence_method="source-evidence-v1",
            estimation_method="not-applied-v1",
            survival_method="not-applied-v1",
            hazard_method="not-applied-v1",
            forecast_method="not-applied-v1",
        ),
        database_sha256=candidate_service.manifest.database_sha256,
        observation_count=candidate_service.manifest.observation_count,
        published_value_count=candidate_service.manifest.published_value_count,
        warnings=candidate_service.manifest.warnings,
    )
    repository = SQLiteEvidenceRepository(candidate_service.database_path)
    monkeypatch.setattr(
        SnapshotStore,
        "open_active_snapshot",
        lambda self: (unresolved, repository),
    )

    with pytest.raises(RegistrationUnavailableError, match="unavailable"):
        RegistrationService.from_active(mapped_candidate.parent)

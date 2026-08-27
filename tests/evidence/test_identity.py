from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from icor.domain.evidence import (
    EvidenceConfidence,
    MappingStatus,
    Measure,
    Observation,
    PeriodPrecision,
    PublicationStatus,
    ReleaseManifest,
)
from icor.evidence.identity import (
    ExactNormalizedIdentityResolver,
    IdentityAttributingRepository,
)
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

BUILD_AS_OF = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _confidence() -> EvidenceConfidence:
    return EvidenceConfidence(
        authority=25,
        publication_status=10,
        coverage=25,
        identity=0,
        independent_agreement=10,
        reasons=("Canonical identity has not been resolved.",),
    )


def _observation(**overrides: object) -> Observation:
    values: dict[str, object] = {
        "observation_id": "obs-example-alpha-de",
        "release_id": "eea-2024-final",
        "original_row_locator": "row:2",
        "geography": "DE",
        "geography_version": "eea-member-state-2024",
        "period_start": date(2024, 1, 1),
        "period_end": date(2024, 12, 31),
        "period_precision": PeriodPrecision.YEAR,
        "measure": Measure.NEW_REGISTRATIONS,
        "value": Decimal("10"),
        "unit": "vehicles",
        "publication_status": PublicationStatus.FINAL,
        "original_make": " Example Motors ",
        "original_model": " Alpha ",
        "original_model_year": None,
        "original_type": "passenger car",
        "source_make_identifier": "Example Motors",
        "source_model_identifier": "Alpha",
        "normalized_make": "example motors",
        "normalized_model": "alpha",
        "normalized_model_year": None,
        "canonical_vehicle_id": None,
        "mapping_status": MappingStatus.UNRESOLVED,
        "transformation_notes": ("Source label retained.",),
        "validation_flags": (),
        "evidence_confidence": _confidence(),
    }
    values.update(overrides)
    return Observation(**values)  # type: ignore[arg-type]


def _release() -> ReleaseManifest:
    return ReleaseManifest(
        release_id="eea-2024-final",
        source_id="eea-co2-cars",
        publisher="European Environment Agency",
        source_url="https://example.test/eea",
        retrieved_at=datetime(2026, 8, 27, 10, 0, tzinfo=UTC),
        published_at=datetime(2026, 6, 25, 0, 0, tzinfo=UTC),
        coverage_start=date(2024, 1, 1),
        coverage_end=date(2024, 12, 31),
        geography="EEA reporting countries",
        geography_version="eea-member-state-2024",
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="national-registration-registers",
        terms_url="https://example.test/terms",
        permitted_local_use="CC BY 4.0",
        artifact_path="eea.zip",
        artifact_bytes=1,
        sha256="a" * 64,
        parser_name="eea_co2_cars_zip_v1",
        parser_version="v1",
        expected_schema="eea-v1",
        raw_record_count=2,
        accepted_record_count=2,
        rejected_record_count=0,
        quarantined_record_count=0,
    )


def test_exact_normalized_labels_map_to_stable_model_family() -> None:
    resolver = ExactNormalizedIdentityResolver()

    first = resolver.resolve(_observation(), reviewed_at=BUILD_AS_OF)
    second = resolver.resolve(
        _observation(
            observation_id="obs-example-alpha-fr",
            original_row_locator="row:3",
            geography="FR",
            original_make="EXAMPLE MOTORS",
            original_model="ALPHA",
        ),
        reviewed_at=BUILD_AS_OF,
    )

    assert first.vehicle is not None
    assert first.vehicle.vehicle_id == second.vehicle.vehicle_id
    assert first.vehicle.make == "Example Motors"
    assert first.vehicle.model == "Alpha"
    assert first.vehicle.model_year is None
    assert first.observation.original_make == " Example Motors "
    assert first.observation.mapping_status is MappingStatus.NORMALIZED_LABEL
    assert first.observation.canonical_vehicle_id == first.vehicle.vehicle_id
    assert first.mapping is not None
    assert first.mapping.status is MappingStatus.NORMALIZED_LABEL
    assert first.mapping.reviewed_at == BUILD_AS_OF
    assert first.observation.evidence_confidence.identity == 10
    assert first.observation.evidence_confidence.total <= 79


@pytest.mark.parametrize(
    "label",
    ("SONSTIGE", "OTHER", "UNKNOWN", "(not reported)"),
)
def test_generic_model_labels_never_publish(label: str) -> None:
    result = ExactNormalizedIdentityResolver().resolve(
        _observation(original_model=label, normalized_model=label.casefold()),
        reviewed_at=BUILD_AS_OF,
    )

    assert result.vehicle is None
    assert result.mapping is None
    assert result.observation.canonical_vehicle_id is None
    assert result.observation.mapping_status is MappingStatus.REJECTED


def test_missing_or_pre_rejected_identity_remains_unpublished() -> None:
    resolver = ExactNormalizedIdentityResolver()

    missing = resolver.resolve(
        _observation(normalized_model=None), reviewed_at=BUILD_AS_OF
    )
    rejected = resolver.resolve(
        _observation(mapping_status=MappingStatus.REJECTED), reviewed_at=BUILD_AS_OF
    )

    assert missing.observation.mapping_status is MappingStatus.UNRESOLVED
    assert missing.vehicle is None
    assert rejected.observation.mapping_status is MappingStatus.REJECTED
    assert rejected.vehicle is None


def test_punctuation_variants_are_not_fuzzily_merged() -> None:
    resolver = ExactNormalizedIdentityResolver()

    hyphenated = resolver.resolve(
        _observation(normalized_model="alpha-x"), reviewed_at=BUILD_AS_OF
    )
    spaced = resolver.resolve(
        _observation(
            observation_id="obs-example-alpha-x",
            normalized_model="alpha x",
        ),
        reviewed_at=BUILD_AS_OF,
    )

    assert hyphenated.vehicle is not None
    assert spaced.vehicle is not None
    assert hyphenated.vehicle.vehicle_id != spaced.vehicle.vehicle_id


def test_attributing_repository_persists_one_vehicle_and_mapping_per_observation(
    tmp_path: Path,
) -> None:
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)
    repository.add_release(_release())
    attributing = IdentityAttributingRepository(
        repository,
        ExactNormalizedIdentityResolver(),
        reviewed_at=BUILD_AS_OF,
    )
    first = _observation()
    second = replace(
        first,
        observation_id="obs-example-alpha-fr",
        original_row_locator="row:3",
        geography="FR",
    )

    attributing.add_observations((first, second))

    assert len(repository.list_vehicles()) == 1
    assert len(repository.list_observations()) == 2
    assert len(repository.list_mappings()) == 2
    assert {
        item.mapping_status for item in repository.list_observations()
    } == {MappingStatus.NORMALIZED_LABEL}

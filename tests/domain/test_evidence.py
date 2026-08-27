from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from icor.domain.evidence import (
    CanonicalVehicle,
    ConfidenceBand,
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


def make_confidence(total: int = 100, **overrides: object) -> EvidenceConfidence:
    components = (25, 10, 25, 20, 20)
    remaining = total
    values = []
    for component in components:
        value = min(component, max(remaining, 0))
        values.append(value)
        remaining -= value
    fields: dict[str, object] = {
        "authority": values[0],
        "publication_status": values[1],
        "coverage": values[2],
        "identity": values[3],
        "independent_agreement": values[4],
        "reasons": ("Source rubric applied.",),
        "applied_cap": None,
    }
    fields.update(overrides)
    return EvidenceConfidence(**fields)  # type: ignore[arg-type]


def make_release(**overrides: object) -> ReleaseManifest:
    values: dict[str, object] = {
        "release_id": "eea-2024-20260826",
        "source_id": "eea",
        "publisher": "European Environment Agency",
        "source_url": "https://example.test/eea/2024",
        "retrieved_at": datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        "published_at": datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        "coverage_start": date(2024, 1, 1),
        "coverage_end": date(2024, 12, 31),
        "geography": "EU",
        "geography_version": "eu-2024",
        "measure": Measure.NEW_REGISTRATIONS,
        "unit": "vehicles",
        "publication_status": PublicationStatus.FINAL,
        "dependency_group": "eea-direct",
        "terms_url": "https://example.test/eea/terms",
        "permitted_local_use": "Research and local validation are permitted.",
        "artifact_path": "eea-2024.csv",
        "artifact_bytes": 42,
        "sha256": "a" * 64,
        "parser_name": "eea_csv",
        "parser_version": "v1",
        "expected_schema": "eea-2024-v1",
        "raw_record_count": 10,
        "accepted_record_count": 8,
        "rejected_record_count": 1,
        "quarantined_record_count": 1,
    }
    values.update(overrides)
    return ReleaseManifest(**values)  # type: ignore[arg-type]


def make_observation(**overrides: object) -> Observation:
    values: dict[str, object] = {
        "observation_id": "observation-eea-de-2024-1",
        "release_id": "eea-2024-20260826",
        "original_row_locator": "row:2",
        "geography": "DE",
        "geography_version": "de-2024",
        "period_start": date(2024, 1, 1),
        "period_end": date(2024, 12, 31),
        "period_precision": PeriodPrecision.YEAR,
        "measure": Measure.NEW_REGISTRATIONS,
        "value": Decimal("1"),
        "unit": "vehicles",
        "publication_status": PublicationStatus.FINAL,
        "original_make": "Example Motors",
        "original_model": "Example One",
        "original_model_year": "2024",
        "original_type": "Passenger car",
        "source_make_identifier": "example-motors",
        "source_model_identifier": "example-one",
        "normalized_make": "example motors",
        "normalized_model": "example one",
        "normalized_model_year": 2024,
        "canonical_vehicle_id": "vehicle-example-one-2024",
        "mapping_status": MappingStatus.EXACT_IDENTIFIER,
        "transformation_notes": ("Parsed without transformation.",),
        "validation_flags": (),
        "evidence_confidence": make_confidence(),
    }
    values.update(overrides)
    return Observation(**values)  # type: ignore[arg-type]


def make_vehicle(**overrides: object) -> CanonicalVehicle:
    values: dict[str, object] = {
        "vehicle_id": "vehicle-example-one-2024",
        "make": "Example Motors",
        "model": "Example One",
        "model_year": 2024,
        "market": "DE",
    }
    values.update(overrides)
    return CanonicalVehicle(**values)  # type: ignore[arg-type]


def test_canonical_model_family_can_record_unknown_model_year() -> None:
    vehicle = make_vehicle(model_year=None)

    assert vehicle.model_year is None


def test_observation_keeps_registration_manufacture_and_model_year_separate() -> None:
    observation = make_observation(
        registration_cohort_year=2020,
        manufacture_year=2019,
        model_year=2021,
    )

    assert observation.registration_cohort_year == 2020
    assert observation.manufacture_year == 2019
    assert observation.model_year == 2021


@pytest.mark.parametrize(
    "field,value",
    (("registration_cohort_year", "2020"), ("manufacture_year", 2020.0), ("model_year", True)),
)
def test_observation_rejects_non_integer_year_semantics(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="year"):
        make_observation(**{field: value})


@pytest.mark.parametrize("model_year", ["2024", 2024.0])
def test_canonical_model_family_rejects_non_integer_known_year(model_year: object) -> None:
    with pytest.raises(ValueError, match="model year"):
        make_vehicle(model_year=model_year)


def make_mapping(**overrides: object) -> IdentityMapping:
    values: dict[str, object] = {
        "mapping_id": "mapping-eea-de-2024-1",
        "observation_id": "observation-eea-de-2024-1",
        "canonical_vehicle_id": "vehicle-example-one-2024",
        "status": MappingStatus.EXACT_IDENTIFIER,
        "reason": "Source model identifier matches the curated vehicle identifier.",
        "reviewed_at": datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
    }
    values.update(overrides)
    return IdentityMapping(**values)  # type: ignore[arg-type]


def make_published_value(**overrides: object) -> PublishedValue:
    values: dict[str, object] = {
        "value_id": "published-example-one-2024",
        "status": ValueStatus.RECONCILED,
        "measure": Measure.NEW_REGISTRATIONS,
        "unit": "vehicles",
        "geography": "DE",
        "geography_version": "de-2024",
        "period_start": date(2024, 1, 1),
        "period_end": date(2024, 12, 31),
        "canonical_vehicle_id": "vehicle-example-one-2024",
        "mapping_status": MappingStatus.EXACT_IDENTIFIER,
        "value": Decimal("10"),
        "p10": Decimal("8"),
        "p50": Decimal("10"),
        "p90": Decimal("12"),
        "input_ids": ("observation-eea-de-2024-1",),
        "method_version": "reconciliation-v1",
        "evidence_confidence": make_confidence(),
        "forecast_confidence": None,
        "warnings": (),
    }
    values.update(overrides)
    return PublishedValue(**values)  # type: ignore[arg-type]


def test_release_manifest_requires_utc_retrieval_and_sha256() -> None:
    with pytest.raises(ValueError, match="UTC"):
        make_release(retrieved_at=datetime(2026, 8, 26))
    with pytest.raises(ValueError, match="SHA-256"):
        make_release(sha256="abc")


@pytest.mark.parametrize("field", ["release_id", "source_id"])
def test_release_manifest_rejects_invalid_identifiers(field: str) -> None:
    with pytest.raises(ValueError, match="identifier"):
        make_release(**{field: "invalid/id"})


@pytest.mark.parametrize("field", ["publisher", "source_url", "terms_url", "permitted_local_use"])
def test_release_manifest_requires_publisher_url_and_terms(field: str) -> None:
    with pytest.raises(ValueError, match="required"):
        make_release(**{field: ""})


def test_release_manifest_requires_ascending_coverage_and_reconciled_record_counts() -> None:
    with pytest.raises(ValueError, match="coverage"):
        make_release(coverage_start=date(2025, 1, 1))
    with pytest.raises(ValueError, match="record counts"):
        make_release(accepted_record_count=7)


def test_release_manifest_is_immutable() -> None:
    release = make_release()
    with pytest.raises(FrozenInstanceError):
        release.publisher = "Other"  # type: ignore[misc]


def test_observation_rejects_negative_counts_and_missing_source_label() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_observation(value=Decimal("-1"))
    with pytest.raises(ValueError, match="original"):
        make_observation(original_make="")


def test_observation_requires_integral_count_values_and_units() -> None:
    with pytest.raises(ValueError, match="integer"):
        make_observation(value=Decimal("1.5"))
    with pytest.raises(ValueError, match="unit"):
        make_observation(unit="")


@pytest.mark.parametrize(
    "value",
    (
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ),
)
def test_observation_rejects_every_non_finite_decimal(value: Decimal) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_observation(value=value)


def test_observation_retains_original_labels_beside_normalized_labels() -> None:
    observation = make_observation(
        original_make="ExAmple MOTORS",
        original_model="EXAMPLE  ONE",
        normalized_make="example motors",
        normalized_model="example one",
    )
    assert observation.original_make == "ExAmple MOTORS"
    assert observation.original_model == "EXAMPLE  ONE"
    assert observation.normalized_make == "example motors"
    assert observation.normalized_model == "example one"


def test_observation_requires_ordered_dates_and_supported_publication_status() -> None:
    with pytest.raises(ValueError, match="period"):
        make_observation(period_start=date(2025, 1, 1))
    with pytest.raises(ValueError, match="publication status"):
        make_observation(publication_status="draft")


def test_identity_mapping_requires_utc_review_time_and_resolved_vehicle() -> None:
    with pytest.raises(ValueError, match="UTC"):
        make_mapping(reviewed_at=datetime(2026, 8, 26, 10, 0))
    with pytest.raises(ValueError, match="canonical vehicle"):
        make_mapping(canonical_vehicle_id=None, status=MappingStatus.EXACT_IDENTIFIER)


def test_identity_mapping_allows_unresolved_identity_without_a_vehicle() -> None:
    mapping = make_mapping(canonical_vehicle_id=None, status=MappingStatus.UNRESOLVED)
    assert mapping.status is MappingStatus.UNRESOLVED
    assert mapping.canonical_vehicle_id is None


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (0, "very_low"),
        (39, "very_low"),
        (40, "low"),
        (59, "low"),
        (60, "medium"),
        (79, "medium"),
        (80, "high"),
        (100, "high"),
    ],
)
def test_confidence_band_boundaries(score: int, band: str) -> None:
    assert make_confidence(total=score).band.value == band


def test_confidence_has_five_bounded_components_and_applies_caps() -> None:
    confidence = make_confidence(
        applied_cap=70,
        authority=25,
        publication_status=10,
        coverage=25,
        identity=20,
        independent_agreement=20,
    )
    assert confidence.components == {
        "authority": 25,
        "publication_status": 10,
        "coverage": 25,
        "identity": 20,
        "independent_agreement": 20,
    }
    assert confidence.raw_total == 100
    assert confidence.total == 70
    assert confidence.band is ConfidenceBand.MEDIUM
    with pytest.raises(ValueError, match="authority"):
        make_confidence(authority=26)


def test_unresolved_identity_cannot_publish_model_value() -> None:
    with pytest.raises(ValueError, match="unresolved"):
        make_published_value(mapping_status=MappingStatus.UNRESOLVED)


@pytest.mark.parametrize(
    "mapping_status",
    [MappingStatus.AMBIGUOUS, MappingStatus.REJECTED, MappingStatus.UNRESOLVED],
)
def test_non_publishable_identity_statuses_are_rejected(mapping_status: MappingStatus) -> None:
    with pytest.raises(ValueError, match="mapping"):
        make_published_value(mapping_status=mapping_status)


def test_published_value_requires_input_ids_and_ordered_intervals() -> None:
    with pytest.raises(ValueError, match="input"):
        make_published_value(input_ids=())
    with pytest.raises(ValueError, match="p10 <= p50 <= p90"):
        make_published_value(p10=Decimal("11"), p50=Decimal("10"), p90=Decimal("12"))


@pytest.mark.parametrize(
    "value",
    (
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ),
)
def test_published_value_rejects_every_non_finite_decimal(value: Decimal) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_published_value(value=value)


@pytest.mark.parametrize("field", ("p10", "p50", "p90"))
@pytest.mark.parametrize(
    "value",
    (
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ),
)
def test_published_intervals_reject_every_non_finite_decimal(
    field: str,
    value: Decimal,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        make_published_value(**{field: value})


def test_forecast_confidence_exists_only_for_forecasts() -> None:
    with pytest.raises(ValueError, match="forecast confidence"):
        make_published_value(forecast_confidence=75)
    assert (
        make_published_value(
            status=ValueStatus.FORECAST,
            method_version="forecast-v1",
            forecast_confidence=75,
        ).forecast_confidence
        == 75
    )


def test_canonical_vehicle_requires_a_complete_identity() -> None:
    with pytest.raises(ValueError, match="canonical vehicle"):
        make_vehicle(model="")
    with pytest.raises(ValueError, match="model year"):
        make_vehicle(model_year=2024.0)


def test_observation_replace_revalidates_contracts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        replace(make_observation(), value=Decimal("-1"))

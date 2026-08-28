import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from icor.domain.evidence import Measure, PublicationStatus, ReleaseManifest
from icor.evidence.release_manifests import (
    ManifestError,
    load_release_manifest,
    load_snapshot_manifest,
    write_release_manifest,
)


def valid_manifest_dict() -> dict[str, object]:
    return {
        "release_id": "eea-2024-20260826",
        "source_id": "eea",
        "publisher": "European Environment Agency",
        "source_url": "https://example.test/eea/2024",
        "retrieved_at": "2026-08-26T10:00:00+00:00",
        "published_at": "2026-08-01T10:00:00+00:00",
        "coverage_start": "2024-01-01",
        "coverage_end": "2024-12-31",
        "geography": "EU",
        "geography_version": "eu-2024",
        "measure": "new_registrations",
        "unit": "vehicles",
        "publication_status": "final",
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


@pytest.fixture
def release_manifest() -> ReleaseManifest:
    return ReleaseManifest(
        release_id="eea-2024-20260826",
        source_id="eea",
        publisher="European Environment Agency",
        source_url="https://example.test/eea/2024",
        retrieved_at=datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        published_at=datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        coverage_start=date(2024, 1, 1),
        coverage_end=date(2024, 12, 31),
        geography="EU",
        geography_version="eu-2024",
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="eea-direct",
        terms_url="https://example.test/eea/terms",
        permitted_local_use="Research and local validation are permitted.",
        artifact_path="eea-2024.csv",
        artifact_bytes=42,
        sha256="a" * 64,
        parser_name="eea_csv",
        parser_version="v1",
        expected_schema="eea-2024-v1",
        raw_record_count=10,
        accepted_record_count=8,
        rejected_record_count=1,
        quarantined_record_count=1,
    )


def write_json(tmp_path: Path, payload: object, name: str = "manifest.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_release_manifest_round_trips_without_information_loss(
    tmp_path: Path, release_manifest: ReleaseManifest
) -> None:
    path = tmp_path / "manifest.json"

    write_release_manifest(path, release_manifest)

    assert load_release_manifest(path) == release_manifest


def test_writer_rejects_unsafe_artifact_path(
    tmp_path: Path, release_manifest: ReleaseManifest
) -> None:
    path = tmp_path / "manifest.json"
    unsafe_manifest = replace(release_manifest, artifact_path="../artifact.csv")

    with pytest.raises(ManifestError, match="artifact path"):
        write_release_manifest(path, unsafe_manifest)

    assert not path.exists()


def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    path = write_json(tmp_path, valid_manifest_dict() | {"credential": "forbidden"})

    with pytest.raises(ManifestError, match="unknown"):
        load_release_manifest(path)


def test_manifest_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ManifestError, match="JSON"):
        load_release_manifest(path)


def test_manifest_rejects_missing_required_fields(tmp_path: Path) -> None:
    payload = valid_manifest_dict()
    del payload["publisher"]
    path = write_json(tmp_path, payload)

    with pytest.raises(ManifestError, match="missing"):
        load_release_manifest(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("measure", "unsupported", "measure"),
        ("publication_status", "draft", "publication_status"),
        ("retrieved_at", "not-a-date", "retrieved_at"),
        ("coverage_start", "2024/01/01", "coverage_start"),
        ("sha256", "A" * 64, "SHA-256"),
    ],
)
def test_manifest_rejects_invalid_enum_dates_and_hash(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path = write_json(tmp_path, valid_manifest_dict() | {field: value})

    with pytest.raises(ManifestError, match=message):
        load_release_manifest(path)


@pytest.mark.parametrize(
    "artifact_path",
    [
        "/releases/eea.csv",
        "C:\\releases\\eea.csv",
        "\\releases\\eea.csv",
        "C:releases\\eea.csv",
        "..\\eea.csv",
        "releases/../eea.csv",
    ],
)
def test_manifest_rejects_absolute_or_traversing_artifact_paths(
    tmp_path: Path, artifact_path: str
) -> None:
    path = write_json(tmp_path, valid_manifest_dict() | {"artifact_path": artifact_path})

    with pytest.raises(ManifestError, match="artifact path"):
        load_release_manifest(path)


def test_snapshot_manifest_rejects_duplicate_release_ids(tmp_path: Path) -> None:
    payload = {
        "snapshot_id": "snapshot-20260826-abc123",
        "status": "candidate",
        "built_at": "2026-08-26T10:00:00+00:00",
        "deterministic_seed": 42,
        "release_ids": ["eea-2024-20260826", "eea-2024-20260826"],
        "versions": {
            "source_registry": "source-registry-v1",
            "identity_registry": "identity-registry-v1",
            "reconciliation_method": "reconciliation-v1",
            "confidence_method": "confidence-v1",
            "estimation_method": "estimation-v1",
            "survival_method": "survival-v1",
            "hazard_method": "hazard-v1",
            "forecast_method": "forecast-v1",
        },
        "database_sha256": "b" * 64,
        "observation_count": 10,
        "published_value_count": 3,
        "warnings": [],
    }
    path = write_json(tmp_path, payload, "snapshot.json")

    with pytest.raises(ManifestError, match="unique"):
        load_snapshot_manifest(path)


def test_snapshot_manifest_loads_legacy_versions_with_generation_v0(tmp_path: Path) -> None:
    payload = {
        "snapshot_id": "snapshot-20260826-abc123",
        "status": "candidate",
        "built_at": "2026-08-26T10:00:00+00:00",
        "deterministic_seed": 42,
        "release_ids": ["eea-2024-20260826"],
        "versions": {
            "source_registry": "source-registry-v1",
            "identity_registry": "identity-registry-v1",
            "reconciliation_method": "reconciliation-v1",
            "confidence_method": "confidence-v1",
            "estimation_method": "estimation-v1",
            "survival_method": "survival-v1",
            "hazard_method": "hazard-v1",
            "forecast_method": "forecast-v1",
        },
        "database_sha256": "b" * 64,
        "observation_count": 10,
        "published_value_count": 3,
        "warnings": [],
    }
    path = write_json(tmp_path, payload, "legacy-snapshot.json")

    manifest = load_snapshot_manifest(path)

    assert manifest.versions.generation_registry == "generation-registry-v0"
    assert manifest.versions.generation_resolver == "generation-resolver-v0"


def test_manifest_rejects_non_utf8_content(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(ManifestError, match="UTF-8"):
        load_release_manifest(path)

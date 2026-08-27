from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from icor.domain.evidence import Measure, PublicationStatus, ReleaseManifest
from icor.evidence.serialization import sha256_file
from icor.evidence.sources.uk_dft import UKActiveFleetLoader, UKFirstRegistrationLoader
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository


def _release(
    tmp_path: Path,
    *,
    name: str,
    parser_name: str,
    measure: Measure,
    headers: list[str],
    rows: list[list[object]],
) -> StoredRelease:
    artifact = tmp_path / f"{name}.csv"
    with artifact.open("w", encoding="cp1252", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)
    accepted = sum(row[0] == "Cars" and (name != "veh0120" or row[5] == "Licensed") for row in rows)
    manifest = ReleaseManifest(
        release_id=f"uk-dft-{name}-2025-final",
        source_id=f"uk-dft-{name}",
        publisher="UK Department for Transport / DVLA",
        source_url=f"https://assets.publishing.service.gov.uk/{name}.csv",
        retrieved_at=datetime(2026, 8, 27, tzinfo=UTC),
        published_at=datetime(2026, 4, 29, tzinfo=UTC),
        coverage_start=date(2001 if name == "veh0160" else 1994, 1, 1),
        coverage_end=date(2025, 12, 31),
        geography="GB",
        geography_version="GB-DVLA-v1",
        measure=measure,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="uk-dvla-registration-records",
        terms_url="https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
        permitted_local_use="Open Government Licence v3.0 with attribution",
        artifact_path="artifact.csv",
        artifact_bytes=artifact.stat().st_size,
        sha256=sha256_file(artifact),
        parser_name=parser_name,
        parser_version="v1",
        expected_schema=f"{name}-2026q1-revision-v1",
        raw_record_count=len(rows),
        accepted_record_count=accepted,
        rejected_record_count=len(rows) - accepted,
        quarantined_record_count=0,
    )
    return StoredRelease(
        manifest.source_id, manifest.release_id, artifact, tmp_path / "manifest.json", manifest
    )


def test_first_registration_loader_aggregates_to_generic_model_and_final_quarters(
    tmp_path: Path,
) -> None:
    headers = [
        "BodyType",
        "Make",
        "GenModel",
        "Model",
        "Fuel",
        "2026 Q1",
        "2025 Q4",
        "2024 Q4",
        "2024 Q3",
    ]
    rows = [
        ["Cars", "MINI", "MINI COUNTRYMAN", "COUNTRYMAN COOPER", "Petrol", 9, 4, 3, "[c]"],
        ["Cars", "MINI", "MINI COUNTRYMAN", "COUNTRYMAN DIESEL", "Diesel", 8, 6, 4, 1],
        ["Motorcycles", "HONDA", "HONDA CBR", "CBR", "Petrol", 5, 5, 5, 5],
    ]
    release = _release(
        tmp_path,
        name="veh0160",
        parser_name="uk_dft_veh0160_csv_v1",
        measure=Measure.NEW_REGISTRATIONS,
        headers=headers,
        rows=rows,
    )
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)
    repository.add_release(release.manifest)

    UKFirstRegistrationLoader().load((release,), repository)

    observations = repository.list_observations()
    assert [(row.period_end, row.value) for row in observations] == [
        (date(2024, 12, 31), Decimal("7")),
        (date(2025, 12, 31), Decimal("10")),
    ]
    assert all(row.original_model == "MINI COUNTRYMAN" for row in observations)
    assert all(row.period_end.year != 2026 for row in observations)
    assert all("suppressed" in row.validation_flags for row in observations) is False


def test_active_fleet_loader_uses_licensed_stock_without_summing_sorn(
    tmp_path: Path,
) -> None:
    headers = [
        "BodyType",
        "Make",
        "GenModel",
        "Model",
        "Fuel",
        "LicenceStatus",
        "2026 Q1",
        "2025 Q4",
        "2024 Q4",
        "1994 Q4",
    ]
    rows = [
        ["Cars", "FORD", "FORD FIESTA", "FIESTA 1.0", "Petrol", "Licensed", 110, 100, 90, 0],
        ["Cars", "FORD", "FORD FIESTA", "FIESTA 1.5", "Diesel", "Licensed", 25, 20, 15, 0],
        ["Cars", "FORD", "FORD FIESTA", "FIESTA 1.0", "Petrol", "SORN", 12, 10, 8, 0],
    ]
    release = _release(
        tmp_path,
        name="veh0120",
        parser_name="uk_dft_veh0120_csv_v1",
        measure=Measure.ACTIVE_FLEET,
        headers=headers,
        rows=rows,
    )
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)
    repository.add_release(release.manifest)

    UKActiveFleetLoader().load((release,), repository)

    observations = repository.list_observations()
    assert sorted((row.measure, row.period_end, row.value) for row in observations) == [
        (Measure.ACTIVE_FLEET, date(2024, 12, 31), Decimal("105")),
        (Measure.ACTIVE_FLEET, date(2025, 12, 31), Decimal("120")),
    ]
    assert all("LicenceStatus=Licensed" in row.original_type for row in observations)


def test_first_registration_loader_does_not_publish_suppressed_group_quarter(
    tmp_path: Path,
) -> None:
    headers = ["BodyType", "Make", "GenModel", "Model", "Fuel", "2025 Q4"]
    rows = [["Cars", "MINI", "MINI COUNTRYMAN", "COUNTRYMAN", "Petrol", "[c]"]]
    release = _release(
        tmp_path,
        name="veh0160",
        parser_name="uk_dft_veh0160_csv_v1",
        measure=Measure.NEW_REGISTRATIONS,
        headers=headers,
        rows=rows,
    )
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)
    repository.add_release(release.manifest)

    UKFirstRegistrationLoader().load((release,), repository)

    assert repository.list_observations() == ()


def test_first_registration_loader_preserves_cp1252_accented_labels(tmp_path: Path) -> None:
    headers = ["BodyType", "Make", "GenModel", "Model", "Fuel", "2025 Q4"]
    rows = [["Cars", "CITROËN", "CITROËN C3", "C3", "Petrol", 7]]
    release = _release(
        tmp_path,
        name="veh0160",
        parser_name="uk_dft_veh0160_csv_v1",
        measure=Measure.NEW_REGISTRATIONS,
        headers=headers,
        rows=rows,
    )
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)
    repository.add_release(release.manifest)

    UKFirstRegistrationLoader().load((release,), repository)

    observation = repository.list_observations()[0]
    assert observation.original_make == "CITROËN"
    assert observation.original_model == "CITROËN C3"

from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from icor.domain.evidence import Measure, PublicationStatus, ReleaseManifest
from icor.evidence.serialization import sha256_file
from icor.evidence.sources.uk_dft import UKVehicleAgeLoader
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository


def test_vehicle_age_loader_keeps_first_use_and_manufacture_year_separate(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "veh0124.csv"
    with artifact.open("w", encoding="cp1252", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "BodyType",
                "Make",
                "GenModel",
                "Model",
                "YearFirstUsed",
                "YearManufacture",
                "LicenceStatus",
                "2025",
                "2024",
                "2014",
            )
        )
        writer.writerows(
            (
                (
                    "Cars", "VOLKSWAGEN", "VOLKSWAGEN GOLF", "GOLF GTI",
                    2020, 2019, "Licensed", 8, 10, "[z]",
                ),
                (
                    "Cars", "VOLKSWAGEN", "VOLKSWAGEN GOLF", "GOLF TDI",
                    2020, 2019, "Licensed", 4, 5, "[z]",
                ),
                (
                    "Cars", "VOLKSWAGEN", "VOLKSWAGEN GOLF", "GOLF GTI",
                    2020, 2019, "SORN", 2, 2, "[z]",
                ),
            )
        )
    manifest = ReleaseManifest(
        release_id="uk-dft-veh0124-am-2025-final",
        source_id="uk-dft-veh0124-am",
        publisher="UK Department for Transport / DVLA",
        source_url="https://assets.publishing.service.gov.uk/veh0124-am.csv",
        retrieved_at=datetime(2026, 8, 27, tzinfo=UTC),
        published_at=datetime(2026, 4, 29, tzinfo=UTC),
        coverage_start=date(2014, 1, 1),
        coverage_end=date(2025, 12, 31),
        geography="UK",
        geography_version="UK-DVLA-v1",
        measure=Measure.ACTIVE_FLEET,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="uk-dvla-registration-records",
        terms_url="https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
        permitted_local_use="Open Government Licence v3.0 with attribution",
        artifact_path="artifact.csv",
        artifact_bytes=artifact.stat().st_size,
        sha256=sha256_file(artifact),
        parser_name="uk_dft_veh0124_csv_v1",
        parser_version="v1",
        expected_schema="veh0124-2025-final-v1",
        raw_record_count=3,
        accepted_record_count=2,
        rejected_record_count=1,
        quarantined_record_count=0,
    )
    release = StoredRelease(
        manifest.source_id,
        manifest.release_id,
        artifact,
        tmp_path / "manifest.json",
        manifest,
    )
    repository = SQLiteEvidenceRepository(tmp_path / "age.sqlite3", writable=True)
    repository.add_release(manifest)

    UKVehicleAgeLoader().load((release,), repository)

    observations = repository.list_observations()
    assert sorted((row.period_end, row.value) for row in observations) == [
        (date(2024, 12, 31), Decimal("15")),
        (date(2025, 12, 31), Decimal("12")),
    ]
    assert {row.registration_cohort_year for row in observations} == {2020}
    assert {row.manufacture_year for row in observations} == {2019}
    assert {row.model_year for row in observations} == {None}


def test_vehicle_age_loader_preserves_missing_year_markers_without_conflating_them(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "veh0124-missing-years.csv"
    with artifact.open("w", encoding="cp1252", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "BodyType",
                "Make",
                "GenModel",
                "Model",
                "YearFirstUsed",
                "YearManufacture",
                "LicenceStatus",
                "2025",
                "2014",
            )
        )
        writer.writerows(
            (
                ("Cars", "FORD", "FORD FOCUS", "FOCUS A", 2020, "[x]", "Licensed", 3, "[z]"),
                ("Cars", "FORD", "FORD FOCUS", "FOCUS B", "[x]", 2019, "Licensed", 4, "[z]"),
                ("Cars", "FORD", "FORD FOCUS", "FOCUS C", "[x]", "[x]", "Licensed", 5, "[z]"),
            )
        )
    manifest = ReleaseManifest(
        release_id="uk-dft-veh0124-am-2025-missing-years",
        source_id="uk-dft-veh0124-am",
        publisher="UK Department for Transport / DVLA",
        source_url="https://assets.publishing.service.gov.uk/veh0124-am.csv",
        retrieved_at=datetime(2026, 8, 27, tzinfo=UTC),
        published_at=datetime(2026, 4, 29, tzinfo=UTC),
        coverage_start=date(2014, 1, 1),
        coverage_end=date(2025, 12, 31),
        geography="UK",
        geography_version="UK-DVLA-v1",
        measure=Measure.ACTIVE_FLEET,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="uk-dvla-registration-records",
        terms_url="https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
        permitted_local_use="Open Government Licence v3.0 with attribution",
        artifact_path="artifact.csv",
        artifact_bytes=artifact.stat().st_size,
        sha256=sha256_file(artifact),
        parser_name="uk_dft_veh0124_csv_v1",
        parser_version="v1",
        expected_schema="veh0124-2025-final-v1",
        raw_record_count=3,
        accepted_record_count=3,
        rejected_record_count=0,
        quarantined_record_count=0,
    )
    release = StoredRelease(
        manifest.source_id,
        manifest.release_id,
        artifact,
        tmp_path / "manifest.json",
        manifest,
    )
    repository = SQLiteEvidenceRepository(tmp_path / "missing-years.sqlite3", writable=True)
    repository.add_release(manifest)

    UKVehicleAgeLoader().load((release,), repository)

    observations = sorted(repository.list_observations(), key=lambda row: row.value)
    assert [
        (row.value, row.registration_cohort_year, row.manufacture_year)
        for row in observations
    ] == [
        (Decimal("3"), 2020, None),
        (Decimal("4"), None, 2019),
        (Decimal("5"), None, None),
    ]
    assert observations[0].validation_flags == ("manufacture_year_missing",)
    assert observations[1].validation_flags == ("registration_cohort_year_missing",)
    assert observations[2].validation_flags == ("year_semantics_missing",)

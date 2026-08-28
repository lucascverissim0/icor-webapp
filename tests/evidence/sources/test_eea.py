from __future__ import annotations

import csv
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from icor.domain.evidence import MappingStatus, Measure, PublicationStatus, ReleaseManifest
from icor.evidence.serialization import sha256_file
from icor.evidence.sources.eea import EEAAnnualAggregateLoader, EEAPassengerCarLoader
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

HEADERS = (
    "ID",
    "MS",
    "Mp",
    "VFN",
    "Mh",
    "Man",
    "MMS",
    "TAN",
    "T",
    "Va",
    "Ve",
    "Mk",
    "Cn",
    "Ct",
    "Cr",
    "M_kg_",
    "Mt",
    "Enedc_g_km_",
    "Ewltp_g_km_",
    "W_mm_",
    "At1_mm_",
    "At2_mm_",
    "Ft",
    "Fm",
    "Ec_cm3_",
    "Ep_KW_",
    "Z_Wh_km_",
    "IT",
    "Ernedc_g_km_",
    "Erwltp_g_km_",
    "De",
    "Vf",
    "R",
    "Year",
    "Status",
    "Version_file",
    "E_g_km_",
    "Er_g_km_",
    "Zr",
    "Dr",
    "Fc",
    "Ech",
    "RLFI",
)


def _row(*, identifier: str, country: str, make: str, model: str, variant: str) -> dict[str, str]:
    row = dict.fromkeys(HEADERS, "")
    row.update(
        {
            "ID": identifier,
            "MS": country,
            "TAN": "E1*2018/858*00001*01",
            "T": "TYPE1",
            "Va": variant,
            "Ve": "VERSION1",
            "Mk": make,
            "Cn": model,
            "Ct": "M1",
            "Cr": "M1",
            "Ft": "petrol",
            "R": "1",
            "Year": "2024",
            "Status": "F",
            "Version_file": "v30",
            "Dr": "2024-06-30",
        }
    )
    return row


def _stored_release(tmp_path: Path, rows: list[dict[str, str]]) -> StoredRelease:
    csv_path = tmp_path / "co2cars_2024fv30.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, HEADERS, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    artifact = tmp_path / "artifact.zip"
    with ZipFile(artifact, "w", ZIP_DEFLATED) as archive:
        archive.write(csv_path, csv_path.name)
    manifest = ReleaseManifest(
        release_id="eea-co2cars-2024-final-v30",
        source_id="eea-co2cars",
        publisher="European Environment Agency / DG CLIMA",
        source_url="https://discodata.eea.europa.eu/download/CO2Emission/latest/co2cars_2024Fv30",
        retrieved_at=datetime(2026, 8, 27, tzinfo=UTC),
        published_at=datetime(2025, 10, 31, tzinfo=UTC),
        coverage_start=date(2024, 1, 1),
        coverage_end=date(2024, 12, 31),
        geography="EEA reporting countries",
        geography_version="eea-2024-v1",
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="national-registration-authorities-2024",
        terms_url="https://creativecommons.org/licenses/by/4.0/",
        permitted_local_use="CC BY 4.0 with DG CLIMA attribution",
        artifact_path="artifact.zip",
        artifact_bytes=artifact.stat().st_size,
        sha256=sha256_file(artifact),
        parser_name="eea_co2_cars_zip_v1",
        parser_version="v1",
        expected_schema="co2cars-2024-final-v30",
        raw_record_count=len(rows),
        accepted_record_count=len(rows),
        rejected_record_count=0,
        quarantined_record_count=0,
    )
    return StoredRelease(
        manifest.source_id, manifest.release_id, artifact, tmp_path / "manifest.json", manifest
    )


def test_loader_aggregates_only_identical_documented_vehicle_keys(tmp_path: Path) -> None:
    release = _stored_release(
        tmp_path,
        [
            _row(identifier="1", country="DE", make="VOLKSWAGEN", model="GOLF", variant="A"),
            _row(identifier="2", country="DE", make="VOLKSWAGEN", model="GOLF", variant="A"),
            _row(identifier="3", country="DE", make="VOLKSWAGEN", model="GOLF", variant="B"),
            _row(identifier="4", country="FR", make="RENAULT", model="CLIO", variant="A"),
        ],
    )
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)
    repository.add_release(release.manifest)

    EEAPassengerCarLoader().load((release,), repository)

    observations = repository.list_observations()
    assert len(observations) == 3
    assert sorted(
        (row.geography, row.original_make, row.original_model, row.original_type, row.value)
        for row in observations
    ) == [
        ("DE", "VOLKSWAGEN", "GOLF", "E1*2018/858*00001*01|TYPE1|A|VERSION1|petrol", Decimal("2")),
        ("DE", "VOLKSWAGEN", "GOLF", "E1*2018/858*00001*01|TYPE1|B|VERSION1|petrol", Decimal("1")),
        ("FR", "RENAULT", "CLIO", "E1*2018/858*00001*01|TYPE1|A|VERSION1|petrol", Decimal("1")),
    ]
    assert all(row.mapping_status is MappingStatus.UNRESOLVED for row in observations)
    assert all(row.canonical_vehicle_id is None for row in observations)
    assert all(row.normalized_model_year is None for row in observations)
    assert {row.registration_cohort_year for row in observations} == {2024}


def test_loader_rejects_unidentifiable_rows_and_reconciles_manifest(tmp_path: Path) -> None:
    release = _stored_release(
        tmp_path,
        [
            _row(identifier="1", country="DE", make="VW", model="GOLF", variant="A"),
            _row(identifier="2", country="DE", make="VW", model="", variant="B"),
        ],
    )
    release = replace(
        release,
        manifest=replace(
            release.manifest,
            accepted_record_count=1,
            rejected_record_count=1,
        ),
    )
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)
    repository.add_release(release.manifest)

    EEAPassengerCarLoader().load((release,), repository)

    assert [(row.original_model, row.value) for row in repository.list_observations()] == [
        ("GOLF", Decimal("1"))
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("Status", "P", "final status"), ("Year", "2023", "2024"), ("R", "2", "weight")],
)
def test_loader_rejects_rows_outside_final_2024_contract(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    row = _row(identifier="1", country="DE", make="VW", model="GOLF", variant="A")
    row[field] = value
    release = _stored_release(tmp_path, [row])
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)
    repository.add_release(release.manifest)

    with pytest.raises(ValueError, match=message):
        EEAPassengerCarLoader().load((release,), repository)


def test_loader_rejects_unexpected_archive_member(tmp_path: Path) -> None:
    release = _stored_release(
        tmp_path, [_row(identifier="1", country="DE", make="VW", model="GOLF", variant="A")]
    )
    replacement = tmp_path / "wrong.zip"
    with ZipFile(replacement, "w", ZIP_DEFLATED) as archive:
        archive.writestr("unexpected.csv", "ID;MS\n1;DE\n")
    release = StoredRelease(
        release.source_id, release.release_id, replacement, release.manifest_path, release.manifest
    )
    repository = SQLiteEvidenceRepository(tmp_path / "evidence.sqlite3", writable=True)
    repository.add_release(release.manifest)

    with pytest.raises(ValueError, match="archive member"):
        EEAPassengerCarLoader().load((release,), repository)


def test_annual_aggregate_loader_reconciles_source_rows_and_registration_weights(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "eea-2010-final.csv"
    headers = (
        "Year",
        "Status",
        "Version_file",
        "MS",
        "Mk",
        "Cn",
        "TAN",
        "T",
        "Va",
        "Ve",
        "Ft",
        "Registrations",
        "SourceRows",
    )
    with artifact.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, headers, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            (
                {
                    "Year": "2010",
                    "Status": "F",
                    "Version_file": "v2",
                    "MS": "DE",
                    "Mk": "VOLKSWAGEN",
                    "Cn": "GOLF",
                    "TAN": "E1*2007/46*0001",
                    "T": "1K",
                    "Va": "A",
                    "Ve": "1",
                    "Ft": "petrol",
                    "Registrations": "2",
                    "SourceRows": "2",
                },
                {
                    "Year": "2010",
                    "Status": "F",
                    "Version_file": "v2",
                    "MS": "FR",
                    "Mk": "RENAULT",
                    "Cn": "CLIO",
                    "TAN": "",
                    "T": "",
                    "Va": "",
                    "Ve": "",
                    "Ft": "diesel",
                    "Registrations": "3",
                    "SourceRows": "1",
                },
                {
                    "Year": "2010",
                    "Status": "F",
                    "Version_file": "v2",
                    "MS": "DE",
                    "Mk": "",
                    "Cn": "UNKNOWN",
                    "TAN": "",
                    "T": "",
                    "Va": "",
                    "Ve": "",
                    "Ft": "",
                    "Registrations": "1",
                    "SourceRows": "1",
                },
            )
        )
    manifest = ReleaseManifest(
        release_id="eea-co2cars-2010-final-v2",
        source_id="eea-co2-monitoring",
        publisher="European Environment Agency / DG CLIMA",
        source_url="https://discodata.eea.europa.eu/sql",
        retrieved_at=datetime(2026, 8, 28, tzinfo=UTC),
        published_at=datetime(2011, 12, 31, tzinfo=UTC),
        coverage_start=date(2010, 1, 1),
        coverage_end=date(2010, 12, 31),
        geography="EEA reporting countries",
        geography_version="EEA CO2 monitoring 2010 final v2",
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="european-passenger-car-registrations-2010",
        terms_url="https://creativecommons.org/licenses/by/4.0/",
        permitted_local_use="CC BY 4.0 with attribution",
        artifact_path="artifact.csv",
        artifact_bytes=artifact.stat().st_size,
        sha256=sha256_file(artifact),
        parser_name="eea_co2_cars_annual_aggregate_csv_v1",
        parser_version="v1",
        expected_schema="EEA 2010 final v2 canonical aggregate export",
        raw_record_count=4,
        accepted_record_count=3,
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
    repository = SQLiteEvidenceRepository(tmp_path / "annual.sqlite3", writable=True)
    repository.add_release(manifest)

    EEAAnnualAggregateLoader().load((release,), repository)

    observations = repository.list_observations()
    assert [(row.geography, row.original_model, row.value) for row in observations] == [
        ("DE", "GOLF", Decimal("2")),
        ("FR", "CLIO", Decimal("3")),
    ]
    assert {row.registration_cohort_year for row in observations} == {2010}
    assert {row.manufacture_year for row in observations} == {None}
    assert {row.model_year for row in observations} == {None}

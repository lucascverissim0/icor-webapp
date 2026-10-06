from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from icor.domain.evidence import (
    MappingStatus,
    Measure,
    PeriodPrecision,
    PublicationStatus,
    ReleaseManifest,
)
from icor.evidence.serialization import sha256_file
from icor.evidence.sources.rdw import RDWRegisteredCarsLoader
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

HEADER = "merk,handelsbenaming,cohort,vehicles\n"


def _release(
    root: Path,
    body: str,
    *,
    raw: int,
    accepted: int,
    rejected: int,
    header: str = HEADER,
    **changes: object,
) -> tuple[StoredRelease, SQLiteEvidenceRepository]:
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / "artifact.csv"
    artifact.write_text(header + body, encoding="utf-8", newline="")
    manifest = ReleaseManifest(
        release_id="rdw-nl-registered-cars-test",
        source_id="rdw-gekentekende-voertuigen",
        publisher="RDW (Netherlands Vehicle Authority)",
        source_url="https://opendata.rdw.nl/resource/m9d7-ebf2",
        retrieved_at=datetime(2026, 10, 6, tzinfo=UTC),
        published_at=datetime(2026, 10, 6, tzinfo=UTC),
        coverage_start=date(1900, 1, 1),
        coverage_end=date(2026, 10, 6),
        geography="NL",
        geography_version="Netherlands RDW register",
        measure=Measure.ACTIVE_FLEET,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="nl-rdw-vehicle-register",
        terms_url="https://opendata.rdw.nl",
        permitted_local_use="Public Domain / CC0",
        artifact_path="artifact.csv",
        artifact_bytes=artifact.stat().st_size,
        sha256=sha256_file(artifact),
        parser_name="rdw_registered_cars_csv_v1",
        parser_version="v1",
        expected_schema="rdw-test",
        raw_record_count=raw,
        accepted_record_count=accepted,
        rejected_record_count=rejected,
        quarantined_record_count=0,
    )
    manifest = replace(manifest, **changes)
    release = StoredRelease(
        manifest.source_id, manifest.release_id, artifact, root / "manifest.json", manifest
    )
    repository = SQLiteEvidenceRepository(root / "rdw.sqlite3", writable=True)
    repository.add_release(manifest)
    return release, repository


def test_loader_writes_active_fleet_cohort_observations(tmp_path: Path) -> None:
    body = (
        "VOLKSWAGEN,GOLF  VARIANT,2015,10\n"
        "AUDI,A3,2020,7\n"
        "AUDI,A3,,3\n"
        "AUDI,A3,2027,3\n"
        "AUDI,A3,1899,3\n"
    )
    release, repository = _release(tmp_path, body, raw=5, accepted=2, rejected=3)

    RDWRegisteredCarsLoader().load((release,), repository)

    observations = sorted(repository.list_observations(), key=lambda row: row.value)
    assert [row.value for row in observations] == [Decimal("7"), Decimal("10")]
    golf = observations[1]
    assert golf.geography == "NL"
    assert golf.measure is Measure.ACTIVE_FLEET
    assert golf.period_start == golf.period_end == date(2026, 10, 6)
    assert golf.period_precision is PeriodPrecision.DAY
    assert golf.unit == "vehicles"
    assert golf.original_make == "VOLKSWAGEN"
    assert golf.original_model == "GOLF  VARIANT"
    assert golf.normalized_make == "volkswagen"
    assert golf.normalized_model == "golf variant"
    assert golf.registration_cohort_year == 2015
    assert golf.canonical_vehicle_id is None
    assert golf.mapping_status is MappingStatus.UNRESOLVED
    assert golf.publication_status is PublicationStatus.FINAL
    assert golf.transformation_notes
    assert golf.evidence_confidence.reasons
    assert len({row.observation_id for row in observations}) == 2


def test_loader_accepts_cohort_in_stock_year_and_rejects_later(tmp_path: Path) -> None:
    release, repository = _release(
        tmp_path, "A,B,2026,1\nA,B,2027,1\n", raw=2, accepted=1, rejected=1
    )

    RDWRegisteredCarsLoader().load((release,), repository)

    assert [row.registration_cohort_year for row in repository.list_observations()] == [2026]


@pytest.mark.parametrize(("raw", "accepted", "rejected"), [(3, 2, 1), (2, 2, 0), (2, 0, 2)])
def test_loader_rejects_manifest_count_mismatch(
    tmp_path: Path, raw: int, accepted: int, rejected: int
) -> None:
    release, repository = _release(
        tmp_path, "A,B,2026,1\nA,B,2027,1\n", raw=raw, accepted=accepted, rejected=rejected
    )

    with pytest.raises(ValueError, match="counts do not match"):
        RDWRegisteredCarsLoader().load((release,), repository)


def test_loader_rejects_duplicate_groups(tmp_path: Path) -> None:
    release, repository = _release(
        tmp_path, "A,B,2020,1\nA,B,2020,2\n", raw=2, accepted=2, rejected=0
    )

    with pytest.raises(ValueError, match="duplicate"):
        RDWRegisteredCarsLoader().load((release,), repository)


def test_loader_rejects_unreviewed_manifest_contract(tmp_path: Path) -> None:
    release, repository = _release(
        tmp_path / "parser", "A,B,2020,1\n", raw=1, accepted=1, rejected=0, parser_name="other_v1"
    )
    with pytest.raises(ValueError, match="parser"):
        RDWRegisteredCarsLoader().load((release,), repository)

    release, repository = _release(
        tmp_path / "status",
        "A,B,2020,1\n",
        raw=1,
        accepted=1,
        rejected=0,
        publication_status=PublicationStatus.PROVISIONAL,
    )
    with pytest.raises(ValueError, match="final"):
        RDWRegisteredCarsLoader().load((release,), repository)

    release, repository = _release(
        tmp_path / "geo", "A,B,2020,1\n", raw=1, accepted=1, rejected=0, geography="DE"
    )
    with pytest.raises(ValueError, match="geography"):
        RDWRegisteredCarsLoader().load((release,), repository)


def test_loader_rejects_unexpected_schema(tmp_path: Path) -> None:
    release, repository = _release(
        tmp_path, "A,B\n", raw=1, accepted=1, rejected=0, header="make,model\n"
    )

    with pytest.raises(ValueError, match="schema"):
        RDWRegisteredCarsLoader().load((release,), repository)

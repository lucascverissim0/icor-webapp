from __future__ import annotations

import zipfile
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
from icor.evidence.sources.traficom import TraficomVehicleLoader
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

COLUMNS = (
    "ajoneuvoluokka",
    "ensirekisterointipvm",
    "kayttoonottopvm",
    "merkkiSelvakielinen",
    "mallimerkinta",
    "kaupallinenNimi",
)


def _row(klass: str, first_use: str, make: str, model: str, name: str) -> str:
    return ";".join((klass, "01.01.2000", first_use, make, model, name))


def _release(
    root: Path,
    rows: list[str],
    *,
    raw: int,
    accepted: int,
    rejected: int,
    columns: tuple[str, ...] = COLUMNS,
    member: str = "TieliikenneAvoinData_30_06_2026.csv",
    **changes: object,
) -> tuple[StoredRelease, SQLiteEvidenceRepository]:
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / "artifact.zip"
    text = "\n".join((";".join(columns), *rows)) + "\n"
    with zipfile.ZipFile(artifact, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text.encode("iso-8859-1"))
    manifest = ReleaseManifest(
        release_id="traficom-fi-vehicles-test",
        source_id="traficom-open-vehicle-data",
        publisher="Traficom (Finnish Transport and Communications Agency)",
        source_url="https://opendata.traficom.fi/Content/Ajoneuvorekisteri.zip",
        retrieved_at=datetime(2026, 10, 6, tzinfo=UTC),
        published_at=datetime(2026, 7, 27, tzinfo=UTC),
        coverage_start=date(1900, 1, 1),
        coverage_end=date(2026, 6, 30),
        geography="FI",
        geography_version="Finland Traficom vehicle register 2026-06-30",
        measure=Measure.ACTIVE_FLEET,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="fi-traficom-vehicle-register",
        terms_url="https://www.avoindata.fi/data/en_GB/dataset/test",
        permitted_local_use="CC BY 4.0",
        artifact_path="artifact.zip",
        artifact_bytes=artifact.stat().st_size,
        sha256=sha256_file(artifact),
        parser_name="traficom_vehicles_zip_v1",
        parser_version="v1",
        expected_schema="traficom-test",
        raw_record_count=raw,
        accepted_record_count=accepted,
        rejected_record_count=rejected,
        quarantined_record_count=0,
    )
    manifest = replace(manifest, **changes)
    release = StoredRelease(
        manifest.source_id, manifest.release_id, artifact, root / "manifest.json", manifest
    )
    repository = SQLiteEvidenceRepository(root / "traficom.sqlite3", writable=True)
    repository.add_release(manifest)
    return release, repository


def test_loader_aggregates_passenger_cars_by_make_name_and_first_use_year(
    tmp_path: Path,
) -> None:
    rows = [
        _row("M1", "20150310", "Volkswagen", "GOLF-TYPE", "Golf"),
        _row("M1", "20150000", "Volkswagen", "GOLF-TYPE", "Golf"),
        _row("M1G", "20150611", "Volkswagen", "GOLF-TYPE", "Golf"),
        _row("M1", "20200101", "Audi", "8Y", "A3"),
        _row("M1", "20200101", "Audi", "8Y-LONG", ""),
        _row("M1", "20200101", "Audi", "", ""),
        _row("M1", "20200101", "", "X", "Y"),
        _row("M1", "00000000", "Audi", "8Y", "A3"),
        _row("M1", "20270101", "Audi", "8Y", "A3"),
        _row("M1", "18990101", "Audi", "8Y", "A3"),
        _row("M1", "bad", "Audi", "8Y", "A3"),
        _row("N1", "20200101", "Ford", "Transit", "Transit"),
        _row("L3e", "20200101", "Kymco", "Zing", "Zing"),
    ]
    # A blank commercial name is rejected, not filled from the type designation,
    # which holds body-type strings such as "2DSEDAN 1300-11/2400".
    release, repository = _release(tmp_path, rows, raw=13, accepted=4, rejected=9)

    TraficomVehicleLoader().load((release,), repository)

    observations = {
        (row.original_make, row.original_model, row.registration_cohort_year): row
        for row in repository.list_observations()
    }
    assert set(observations) == {
        ("Volkswagen", "Golf", 2015),
        ("Audi", "A3", 2020),
    }
    golf = observations[("Volkswagen", "Golf", 2015)]
    assert golf.value == Decimal("3")
    assert observations[("Audi", "A3", 2020)].value == Decimal("1")
    assert golf.geography == "FI"
    assert golf.measure is Measure.ACTIVE_FLEET
    assert golf.period_start == golf.period_end == date(2026, 6, 30)
    assert golf.period_precision is PeriodPrecision.DAY
    assert golf.unit == "vehicles"
    assert golf.normalized_make == "volkswagen"
    assert golf.normalized_model == "golf"
    assert golf.canonical_vehicle_id is None
    assert golf.mapping_status is MappingStatus.UNRESOLVED
    assert golf.publication_status is PublicationStatus.FINAL
    assert golf.transformation_notes
    assert golf.evidence_confidence.reasons
    assert len({row.observation_id for row in observations.values()}) == 2


def test_loader_reads_the_semicolon_iso_8859_1_zip_member(tmp_path: Path) -> None:
    rows = [_row("M1", "20100101", "Citroën", "C3", "Citroën Käyttö")]
    release, repository = _release(tmp_path, rows, raw=1, accepted=1, rejected=0)

    TraficomVehicleLoader().load((release,), repository)

    [observation] = repository.list_observations()
    assert observation.original_make == "Citroën"
    assert observation.original_model == "Citroën Käyttö"


def test_loader_accepts_cohort_in_stock_year(tmp_path: Path) -> None:
    rows = [_row("M1", "20260630", "A", "B", "C"), _row("M1", "20270101", "A", "B", "C")]
    release, repository = _release(tmp_path, rows, raw=2, accepted=1, rejected=1)

    TraficomVehicleLoader().load((release,), repository)

    assert [row.registration_cohort_year for row in repository.list_observations()] == [2026]


@pytest.mark.parametrize(("raw", "accepted", "rejected"), [(3, 2, 1), (2, 2, 0), (2, 0, 2)])
def test_loader_rejects_manifest_count_mismatch(
    tmp_path: Path, raw: int, accepted: int, rejected: int
) -> None:
    rows = [_row("M1", "20200101", "A", "B", "C"), _row("N1", "20200101", "A", "B", "C")]
    release, repository = _release(tmp_path, rows, raw=raw, accepted=accepted, rejected=rejected)

    with pytest.raises(ValueError, match="counts do not match"):
        TraficomVehicleLoader().load((release,), repository)


def test_loader_rejects_unreviewed_manifest_contract(tmp_path: Path) -> None:
    rows = [_row("M1", "20200101", "A", "B", "C")]
    release, repository = _release(
        tmp_path / "parser", rows, raw=1, accepted=1, rejected=0, parser_name="other_v1"
    )
    with pytest.raises(ValueError, match="parser"):
        TraficomVehicleLoader().load((release,), repository)

    release, repository = _release(
        tmp_path / "status",
        rows,
        raw=1,
        accepted=1,
        rejected=0,
        publication_status=PublicationStatus.PROVISIONAL,
    )
    with pytest.raises(ValueError, match="final"):
        TraficomVehicleLoader().load((release,), repository)

    release, repository = _release(
        tmp_path / "geo", rows, raw=1, accepted=1, rejected=0, geography="NL"
    )
    with pytest.raises(ValueError, match="geography"):
        TraficomVehicleLoader().load((release,), repository)


def test_loader_rejects_unexpected_schema(tmp_path: Path) -> None:
    release, repository = _release(
        tmp_path, ["M1;x"], raw=1, accepted=0, rejected=1, columns=("ajoneuvoluokka", "x")
    )

    with pytest.raises(ValueError, match="schema"):
        TraficomVehicleLoader().load((release,), repository)


def test_loader_rejects_archive_without_exactly_one_csv(tmp_path: Path) -> None:
    release, repository = _release(
        tmp_path, [_row("M1", "20200101", "A", "B", "C")], raw=1, accepted=1, rejected=0,
        member="notes.txt",
    )

    with pytest.raises(ValueError, match="CSV"):
        TraficomVehicleLoader().load((release,), repository)

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from icor.domain.evidence import (
    STOCK_LOWER_BOUND_FLAG,
    MappingStatus,
    Measure,
    PeriodPrecision,
    PublicationStatus,
    ReleaseManifest,
)
from icor.evidence.serialization import sha256_file
from icor.evidence.sources.dgt import DGTParqueLoader
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

HEADER = "marca,modelo,cohort,vehicles\n"


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
        release_id="dgt-es-parque-test",
        source_id="dgt-parque-vehiculos-anual",
        publisher="DGT",
        source_url="https://www.dgt.es/microdatos/Parque/2025/parque_consolidado_2025.zip",
        retrieved_at=datetime(2026, 10, 21, tzinfo=UTC),
        published_at=datetime(2026, 4, 10, tzinfo=UTC),
        coverage_start=date(1900, 1, 1),
        coverage_end=date(2025, 12, 31),
        geography="ES",
        geography_version="Spain DGT parque 2025",
        measure=Measure.ACTIVE_FLEET,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="es-dgt-vehicle-register",
        terms_url="https://creativecommons.org/licenses/by/4.0/",
        permitted_local_use="CC BY 4.0",
        artifact_path="artifact.csv",
        artifact_bytes=artifact.stat().st_size,
        sha256=sha256_file(artifact),
        parser_name="dgt_parque_cohorts_csv_v1",
        parser_version="v1",
        expected_schema="dgt-test",
        raw_record_count=raw,
        accepted_record_count=accepted,
        rejected_record_count=rejected,
        quarantined_record_count=0,
    )
    manifest = replace(manifest, **changes)
    release = StoredRelease(
        manifest.source_id, manifest.release_id, artifact, root / "manifest.json", manifest
    )
    repository = SQLiteEvidenceRepository(root / "dgt.sqlite3", writable=True)
    repository.add_release(manifest)
    return release, repository


def test_loader_writes_lower_bound_stock_for_cohorts_before_2010(tmp_path: Path) -> None:
    body = (
        "MARCA A,MODEL ONE 1.6,2005,10\n"
        "MARCA A,MODEL ONE 1.6,2009,7\n"
        "MARCA A,MODEL ONE 1.6,2010,30\n"
        "MARCA A,¡,2005,99\n"
        "¡,¡,2005,5\n"
        "MARCA A,,2005,4\n"
        "MARCA A,MODEL ONE,1899,3\n"
        "MARCA A,MODEL ONE,,3\n"
    )
    release, repository = _release(tmp_path, body, raw=8, accepted=2, rejected=6)

    DGTParqueLoader().load((release,), repository)

    observations = sorted(repository.list_observations(), key=lambda row: row.value)
    assert [row.value for row in observations] == [Decimal("7"), Decimal("10")]
    first = observations[1]
    assert first.geography == "ES"
    assert first.measure is Measure.ACTIVE_FLEET
    assert first.period_start == first.period_end == date(2025, 12, 31)
    assert first.period_precision is PeriodPrecision.DAY
    assert first.original_make == "MARCA A"
    assert first.original_model == "MODEL ONE 1.6"
    assert first.normalized_make == "marca a"
    assert first.registration_cohort_year == 2005
    assert first.validation_flags == (STOCK_LOWER_BOUND_FLAG,)
    assert first.canonical_vehicle_id is None
    assert first.mapping_status is MappingStatus.UNRESOLVED
    assert first.publication_status is PublicationStatus.FINAL
    assert any("lower bound" in note for note in first.transformation_notes)
    assert first.evidence_confidence.coverage < 25


@pytest.mark.parametrize(("raw", "accepted", "rejected"), [(3, 2, 1), (2, 2, 0), (2, 0, 2)])
def test_loader_rejects_manifest_count_mismatch(
    tmp_path: Path, raw: int, accepted: int, rejected: int
) -> None:
    release, repository = _release(
        tmp_path, "A,B,2001,1\nA,B,2011,1\n", raw=raw, accepted=accepted, rejected=rejected
    )

    with pytest.raises(ValueError, match="counts do not match"):
        DGTParqueLoader().load((release,), repository)


def test_loader_rejects_duplicate_groups(tmp_path: Path) -> None:
    release, repository = _release(
        tmp_path, "A,B,2000,1\nA,B,2000,2\n", raw=2, accepted=2, rejected=0
    )

    with pytest.raises(ValueError, match="duplicate"):
        DGTParqueLoader().load((release,), repository)


def test_loader_rejects_unreviewed_manifest_contract(tmp_path: Path) -> None:
    cases = (
        ("parser", {"parser_name": "other_v1"}, "parser"),
        ("status", {"publication_status": PublicationStatus.PROVISIONAL}, "final"),
        ("geo", {"geography": "PT"}, "geography"),
        ("date", {"coverage_end": date(2025, 10, 6)}, "year end"),
    )
    for name, changes, message in cases:
        release, repository = _release(
            tmp_path / name, "A,B,2000,1\n", raw=1, accepted=1, rejected=0, **changes
        )
        with pytest.raises(ValueError, match=message):
            DGTParqueLoader().load((release,), repository)


def test_loader_rejects_unexpected_schema(tmp_path: Path) -> None:
    release, repository = _release(
        tmp_path, "A,B\n", raw=1, accepted=1, rejected=0, header="make,model\n"
    )

    with pytest.raises(ValueError, match="schema"):
        DGTParqueLoader().load((release,), repository)

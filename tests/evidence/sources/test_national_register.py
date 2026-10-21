from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from icor.domain.evidence import (
    STOCK_LOWER_BOUND_FLAG,
    Measure,
    PeriodPrecision,
    PublicationStatus,
    ReleaseManifest,
)
from icor.evidence.serialization import sha256_file
from icor.evidence.sources.national_register import (
    PARSER_NAME,
    NationalRegisterLoader,
    RegisterProfile,
)
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

HEADER = "make,model,cohort,vehicles\n"
_COMPLETE = RegisterProfile(
    source_id="xx-register",
    geography="XX",
    publisher="Fictional Register",
    definition="the year of first registration anywhere",
    masks=frozenset({"hidden"}),
)


def _release(
    root: Path, body: str, *, raw: int, accepted: int, rejected: int, **changes: object
) -> tuple[StoredRelease, SQLiteEvidenceRepository]:
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / "artifact.csv"
    artifact.write_text(HEADER + body, encoding="utf-8", newline="")
    manifest = ReleaseManifest(
        release_id="xx-registered-cars-20261001",
        source_id="xx-register",
        publisher="Fictional Register",
        source_url="https://register.example/fleet.csv",
        retrieved_at=datetime(2026, 10, 21, tzinfo=UTC),
        published_at=datetime(2026, 10, 1, tzinfo=UTC),
        coverage_start=date(1900, 1, 1),
        coverage_end=date(2026, 10, 1),
        geography="XX",
        geography_version="Fictional register 2026-10-01",
        measure=Measure.ACTIVE_FLEET,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="xx-vehicle-register",
        terms_url="https://register.example/terms",
        permitted_local_use="CC0",
        artifact_path="artifact.csv",
        artifact_bytes=artifact.stat().st_size,
        sha256=sha256_file(artifact),
        parser_name=PARSER_NAME,
        parser_version="v1",
        expected_schema="register-test",
        raw_record_count=raw,
        accepted_record_count=accepted,
        rejected_record_count=rejected,
        quarantined_record_count=0,
    )
    manifest = replace(manifest, **changes)
    release = StoredRelease(
        manifest.source_id, manifest.release_id, artifact, root / "manifest.json", manifest
    )
    repository = SQLiteEvidenceRepository(root / "register.sqlite3", writable=True)
    repository.add_release(manifest)
    return release, repository


def test_a_complete_register_gives_every_cohort_from_2006(tmp_path: Path) -> None:
    body = (
        "AURORA,Horizon,2005,4\n"
        "AURORA,Horizon,2006,5\n"
        "AURORA,Horizon,2024,30\n"
        "AURORA,,2024,2\n"
        "HIDDEN,HIDDEN,2020,3\n"
        "AURORA,Horizon,,6\n"
    )
    release, repository = _release(tmp_path, body, raw=6, accepted=2, rejected=4)

    NationalRegisterLoader({"xx-register": _COMPLETE}).load((release,), repository)

    rows = sorted(
        (row.registration_cohort_year, row.value, row.validation_flags, row.period_end)
        for row in repository.list_observations()
    )
    untaught = ("labels-do-not-teach-vocabulary",)
    assert rows == [
        (2006, Decimal(5), untaught, date(2026, 10, 1)),
        (2024, Decimal(30), untaught, date(2026, 10, 1)),
    ]
    observation = repository.list_observations()[0]
    assert observation.measure is Measure.ACTIVE_FLEET
    assert observation.period_precision is PeriodPrecision.DAY
    assert observation.geography == "XX"
    assert observation.normalized_model == "horizon"


def test_a_lower_bound_register_flags_every_cohort(tmp_path: Path) -> None:
    body = "AURORA,Horizon,2009,5\nAURORA,Horizon,2010,30\n"
    release, repository = _release(tmp_path, body, raw=2, accepted=2, rejected=0)
    profile = replace(_COMPLETE, lower_bound=True)

    NationalRegisterLoader({"xx-register": profile}).load((release,), repository)

    observations = repository.list_observations()
    assert sorted(item.registration_cohort_year for item in observations) == [2009, 2010]
    assert {item.validation_flags for item in observations} == {
        (STOCK_LOWER_BOUND_FLAG, "labels-do-not-teach-vocabulary")
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_id": "yy-register"}, "no reviewed register profile"),
        ({"geography": "YY"}, "geography or measure"),
        ({"publication_status": PublicationStatus.PROVISIONAL}, "final"),
        ({"accepted_record_count": 2, "rejected_record_count": 0}, "counts do not match"),
    ],
)
def test_an_unreviewed_or_inconsistent_release_is_refused(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    release, repository = _release(
        tmp_path,
        "AURORA,Horizon,2010,3\nAURORA,Horizon,2004,1\n",
        raw=2,
        accepted=1,
        rejected=1,
        **changes,
    )

    with pytest.raises(ValueError, match=message):
        NationalRegisterLoader({"xx-register": _COMPLETE}).load((release,), repository)


def test_a_duplicate_group_or_drifted_schema_is_refused(tmp_path: Path) -> None:
    release, repository = _release(
        tmp_path / "duplicate",
        "AURORA,Horizon,2010,3\nAURORA,Horizon,2010,1\n",
        raw=2,
        accepted=2,
        rejected=0,
    )
    with pytest.raises(ValueError, match="duplicate"):
        NationalRegisterLoader({"xx-register": _COMPLETE}).load((release,), repository)

    drifted, repository = _release(
        tmp_path / "drifted", "AURORA,Horizon,2010,3\n", raw=1, accepted=1, rejected=0
    )
    drifted.artifact_path.write_text(
        "marca,modelo,cohort,vehicles\nAURORA,Horizon,2010,3\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema"):
        NationalRegisterLoader({"xx-register": _COMPLETE}).load((drifted,), repository)

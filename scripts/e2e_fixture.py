"""Build a tiny deterministic sealed snapshot for clean-room browser tests."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

from icor.application.snapshot_build import SnapshotBuilder, SnapshotBuildRequest
from icor.domain.evidence import Measure, PublicationStatus, ReleaseManifest
from icor.evidence.serialization import sha256_file
from icor.evidence.source_registry import (
    OFFICIAL_SOURCE_VERSIONS,
    official_repository_transformer,
)
from icor.evidence.sources.eea import ANNUAL_AGGREGATE_SCHEMA, EEAAnnualAggregateLoader
from icor.infrastructure.release_store import ReleaseStore

_BUILD_AS_OF = datetime(2026, 8, 29, 12, tzinfo=UTC)
_RELEASE_ID = "eea-co2cars-2024-final-v30"
_VERSIONS = replace(
    OFFICIAL_SOURCE_VERSIONS,
    generation_registry="generation-registry-v0",
    generation_resolver="generation-resolver-v0",
)
_ROWS = (
    ("FR", "DACIA", "SANDERO", "300"),
    ("DE", "TESLA", "MODEL Y", "200"),
    ("IT", "ALFA ROMEO", "GIULIA", "100"),
)


def prepare_e2e_fixture(root: Path) -> Path:
    """Return an idempotently built candidate using production snapshot boundaries."""
    root = Path(root)
    inputs = root / "fixture-input"
    inputs.mkdir(parents=True, exist_ok=True)
    artifact = inputs / "eea-2024-final.csv"
    _write_artifact(artifact)
    manifest = _manifest(artifact)
    store = ReleaseStore(root / "releases")
    store.stage(artifact, manifest)
    result = SnapshotBuilder(
        root,
        store,
        EEAAnnualAggregateLoader(),
        repository_transformer=official_repository_transformer,
    ).build(
        SnapshotBuildRequest(
            release_ids=(_RELEASE_ID,),
            versions=_VERSIONS,
            deterministic_seed=20260829,
            build_as_of=_BUILD_AS_OF,
        )
    )
    if not result.validation_report.can_promote:
        raise RuntimeError("browser fixture candidate failed validation")
    return result.candidate_path


def _write_artifact(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, ANNUAL_AGGREGATE_SCHEMA, delimiter=";", lineterminator="\n")
        writer.writeheader()
        for country, make, model, registrations in _ROWS:
            writer.writerow(
                {
                    "Year": "2024",
                    "Status": "F",
                    "Version_file": "v30",
                    "MS": country,
                    "Mk": make,
                    "Cn": model,
                    "TAN": "",
                    "T": "",
                    "Va": "",
                    "Ve": "",
                    "Ft": "petrol",
                    "Registrations": registrations,
                    "SourceRows": "1",
                }
            )


def _manifest(artifact: Path) -> ReleaseManifest:
    return ReleaseManifest(
        release_id=_RELEASE_ID,
        source_id="eea-co2-monitoring",
        publisher="European Environment Agency / DG CLIMA browser fixture",
        source_url="https://example.test/eea-browser-fixture",
        retrieved_at=_BUILD_AS_OF,
        published_at=datetime(2025, 10, 31, tzinfo=UTC),
        coverage_start=date(2024, 1, 1),
        coverage_end=date(2024, 12, 31),
        geography="EEA reporting countries",
        geography_version="EEA browser fixture 2024",
        measure=Measure.NEW_REGISTRATIONS,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group="browser-fixture-2024",
        terms_url="https://creativecommons.org/licenses/by/4.0/",
        permitted_local_use="Synthetic browser fixture; not publisher evidence.",
        artifact_path="artifact.csv",
        artifact_bytes=artifact.stat().st_size,
        sha256=sha256_file(artifact),
        parser_name="eea_co2_cars_annual_aggregate_csv_v1",
        parser_version="v1",
        expected_schema="EEA 2024 browser fixture canonical aggregate",
        raw_record_count=3,
        accepted_record_count=3,
        rejected_record_count=0,
        quarantined_record_count=0,
    )

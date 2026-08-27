"""Pinned acquisition metadata for reviewed official vehicle-evidence releases."""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

from icor.domain.evidence import Measure, PublicationStatus, ReleaseManifest
from icor.infrastructure.release_store import ReleaseStore, StoredRelease


@dataclass(frozen=True, slots=True)
class OfficialSource:
    key: str
    release_id: str
    source_id: str
    publisher: str
    url: str
    published_at: datetime
    coverage_start: date
    coverage_end: date
    geography: str
    geography_version: str
    measure: Measure
    dependency_group: str
    terms_url: str
    permitted_local_use: str
    parser_name: str
    expected_schema: str
    suffix: str
    artifact_bytes: int
    sha256: str
    raw_count: int
    accepted_count: int
    rejected_count: int
    quarantined_count: int = 0


_UK_TERMS = "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
OFFICIAL_SOURCES = {
    "eea-2024-final": OfficialSource(
        key="eea-2024-final",
        release_id="eea-co2cars-2024-final-v30-r1",
        source_id="eea-co2-monitoring",
        publisher="European Environment Agency / European Commission DG CLIMA",
        url="https://discodata.eea.europa.eu/download/CO2Emission/latest/co2cars_2024Fv30",
        published_at=datetime(2026, 8, 7, 8, 49, 5, tzinfo=UTC),
        coverage_start=date(2024, 1, 1),
        coverage_end=date(2024, 12, 31),
        geography="EEA reporting countries",
        geography_version="EEA CO2 monitoring 2024 final v30",
        measure=Measure.NEW_REGISTRATIONS,
        dependency_group="european-passenger-car-registrations-2024",
        terms_url="https://creativecommons.org/licenses/by/4.0/",
        permitted_local_use="Reuse permitted with attribution under CC BY 4.0.",
        parser_name="eea_co2_cars_zip_v1",
        expected_schema="co2cars_2024fv30.csv exact 43-column schema",
        suffix=".zip",
        artifact_bytes=138_252_239,
        sha256="122dab33e931ea04d3ddb4bb2691dae85dc0da14428fc17873d3fb1f648b7b67",
        raw_count=10_782_314,
        accepted_count=10_781_686,
        rejected_count=628,
    ),
    "kba-fz10-2024": OfficialSource(
        key="kba-fz10-2024",
        release_id="kba-fz10-2024-12-final-v3",
        source_id="kba-fz10",
        publisher="Kraftfahrt-Bundesamt (KBA)",
        url="https://www.kba.de/SharedDocs/Downloads/DE/Statistik/Fahrzeuge/FZ10/fz10_2024_12.xlsx?__blob=publicationFile&v=3",
        published_at=datetime(2025, 5, 12, 13, 4, 21, tzinfo=UTC),
        coverage_start=date(2024, 1, 1),
        coverage_end=date(2024, 12, 31),
        geography="DE",
        geography_version="Germany national registration territory 2024",
        measure=Measure.NEW_REGISTRATIONS,
        dependency_group="european-passenger-car-registrations-2024",
        terms_url="https://www.govdata.de/dl-de/by-2-0",
        permitted_local_use="Reuse permitted with source attribution under DL-DE/BY-2.0.",
        parser_name="kba_fz10_xlsx_v1",
        expected_schema="FZ 10.1 December 2024 annual cumulative workbook",
        suffix=".xlsx",
        artifact_bytes=177_108,
        sha256="856b9afe515d51aa52bcb34d645dce2c5cdeaf47ef398b4e0a754c1bd5813dbf",
        raw_count=479,
        accepted_count=417,
        rejected_count=62,
    ),
    "uk-veh0160-gb": OfficialSource(
        key="uk-veh0160-gb",
        release_id="uk-dft-veh0160-gb-2025-final-20260713",
        source_id="uk-dft-veh0160",
        publisher="UK Department for Transport / DVLA",
        url="https://assets.publishing.service.gov.uk/media/6a54d2eea6586e258d371d72/df_VEH0160_GB.csv",
        published_at=datetime(2026, 4, 29, tzinfo=UTC),
        coverage_start=date(2001, 1, 1),
        coverage_end=date(2025, 12, 31),
        geography="GB",
        geography_version="Great Britain licensing geography",
        measure=Measure.NEW_REGISTRATIONS,
        dependency_group="uk-dvla-vehicle-register",
        terms_url=_UK_TERMS,
        permitted_local_use="Reuse permitted with Crown copyright attribution under OGL v3.0.",
        parser_name="uk_dft_veh0160_csv_v1",
        expected_schema="DfT VEH0160 wide CSV through finalized 2025 Q4",
        suffix=".csv",
        artifact_bytes=28_355_274,
        sha256="312d09ecabc0f0bcd85d5d2b10ddebf222ba39bb1f833ad60b725708f4f4f06c",
        raw_count=106_148,
        accepted_count=60_406,
        rejected_count=45_742,
    ),
    "uk-veh0120-gb": OfficialSource(
        key="uk-veh0120-gb",
        release_id="uk-dft-veh0120-gb-2025-final-20260713",
        source_id="uk-dft-veh0120",
        publisher="UK Department for Transport / DVLA",
        url="https://assets.publishing.service.gov.uk/media/6a54d2e39e9c95844ae64da0/df_VEH0120_GB.csv",
        published_at=datetime(2026, 4, 29, tzinfo=UTC),
        coverage_start=date(1994, 1, 1),
        coverage_end=date(2025, 12, 31),
        geography="GB",
        geography_version="Great Britain licensing geography",
        measure=Measure.ACTIVE_FLEET,
        dependency_group="uk-dvla-vehicle-register",
        terms_url=_UK_TERMS,
        permitted_local_use="Reuse permitted with Crown copyright attribution under OGL v3.0.",
        parser_name="uk_dft_veh0120_csv_v1",
        expected_schema="DfT VEH0120 wide CSV, Cars and Licensed, through finalized 2025 Q4",
        suffix=".csv",
        artifact_bytes=65_878_628,
        sha256="3bf96499b09fbb5a9710e1257a2dbc2a8a538190cf823430e2dee23709bb73d3",
        raw_count=245_043,
        accepted_count=77_299,
        rejected_count=167_744,
    ),
}


def validate_source_url(source: OfficialSource, candidate: str) -> None:
    expected = urlsplit(source.url)
    actual = urlsplit(candidate)
    if actual.scheme != "https" or actual.netloc != expected.netloc or candidate != source.url:
        raise ValueError("source URL is not allowlisted")


def build_manifest(
    source: OfficialSource, artifact: Path, *, retrieved_at: datetime
) -> ReleaseManifest:
    digest = hashlib.sha256()
    size = 0
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    if digest.hexdigest() != source.sha256:
        raise ValueError("official artifact checksum does not match the pinned release")
    if size != source.artifact_bytes:
        raise ValueError("official artifact size does not match the pinned release")
    return ReleaseManifest(
        release_id=source.release_id,
        source_id=source.source_id,
        publisher=source.publisher,
        source_url=source.url,
        retrieved_at=retrieved_at,
        published_at=source.published_at,
        coverage_start=source.coverage_start,
        coverage_end=source.coverage_end,
        geography=source.geography,
        geography_version=source.geography_version,
        measure=source.measure,
        unit="vehicles",
        publication_status=PublicationStatus.FINAL,
        dependency_group=source.dependency_group,
        terms_url=source.terms_url,
        permitted_local_use=source.permitted_local_use,
        artifact_path=f"artifact{source.suffix}",
        artifact_bytes=size,
        sha256=source.sha256,
        parser_name=source.parser_name,
        parser_version="v1",
        expected_schema=source.expected_schema,
        raw_record_count=source.raw_count,
        accepted_record_count=source.accepted_count,
        rejected_record_count=source.rejected_count,
        quarantined_record_count=source.quarantined_count,
    )


def download(source: OfficialSource, destination: Path) -> Path:
    """Download one exact allowlisted resource atomically and enforce its byte ceiling."""
    validate_source_url(source, source.url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        request = urllib.request.Request(source.url, headers={"User-Agent": "ICOR-evidence/1"})
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            validate_source_url(source, response.geturl())
            with NamedTemporaryFile("wb", delete=False, dir=destination.parent) as output:
                temporary = Path(output.name)
                copied = 0
                while chunk := response.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > source.artifact_bytes:
                        raise ValueError("official artifact exceeds the pinned byte limit")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        shutil.move(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def acquire(source: OfficialSource, root: Path, *, artifact: Path | None = None) -> StoredRelease:
    retrieved_at = datetime.now(UTC)
    local = artifact or download(source, root / "downloads" / f"{source.key}{source.suffix}")
    manifest = build_manifest(source, local, retrieved_at=retrieved_at)
    return ReleaseStore(root / "releases").stage(local, manifest)

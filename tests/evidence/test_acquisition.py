from datetime import UTC, datetime
from pathlib import Path

import pytest

from icor.evidence.acquisition import OFFICIAL_SOURCES, build_manifest, validate_source_url


def test_catalog_contains_only_reviewed_https_sources():
    expected = {
        "eea-2024-final",
        "eea-2025-provisional",
        "kba-fz10-2024",
        "uk-veh0160-gb",
        "uk-veh0120-gb",
        "uk-veh0124-am",
        "uk-veh0124-nz",
        "rdw-nl-fleet",
        "traficom-fi-fleet",
        "dgt-es-fleet",
    }
    expected.update(f"eea-{year}-final" for year in range(2010, 2024))
    assert set(OFFICIAL_SOURCES) == expected
    for source in OFFICIAL_SOURCES.values():
        validate_source_url(source, source.url)


def test_historical_sources_use_new_canonical_release_revisions() -> None:
    assert tuple(
        OFFICIAL_SOURCES[f"eea-{year}-final"].release_id
        for year in range(2010, 2024)
    ) == (
        "eea-co2cars-2010-final-v2-r1",
        "eea-co2cars-2011-final-v4-r1",
        "eea-co2cars-2012-final-v6-r1",
        "eea-co2cars-2013-final-v8-r1",
        "eea-co2cars-2014-final-v10-r1",
        "eea-co2cars-2015-final-v12-r1",
        "eea-co2cars-2016-final-v14-r1",
        "eea-co2cars-2017-final-v16-r1",
        "eea-co2cars-2018-final-v18-r1",
        "eea-co2cars-2019-final-v20-r1",
        "eea-co2cars-2020-final-v22-r1",
        "eea-co2cars-2021-final-v24-r1",
        "eea-co2cars-2022-final-v26-r1",
        "eea-co2cars-2023-final-v28-r1",
    )


def test_eea_2024_uses_verified_versioned_blob_without_revising_release() -> None:
    source = OFFICIAL_SOURCES["eea-2024-final"]

    assert source.url == (
        "https://dis2datalake.blob.core.windows.net/discodata/co2emission/v7r2/co2cars_2024fv30.zip"
    )
    assert source.release_id == "eea-co2cars-2024-final-v30-r1"
    assert source.artifact_bytes == 138_252_239
    assert source.sha256 == ("122dab33e931ea04d3ddb4bb2691dae85dc0da14428fc17873d3fb1f648b7b67")
    validate_source_url(source, source.url)


def test_eea_2025_uses_replay_verified_provisional_aggregate() -> None:
    source = OFFICIAL_SOURCES["eea-2025-provisional"]

    assert source.publication_status.value == "provisional"
    assert source.release_id == "eea-co2cars-2025-provisional-v31-r1"
    assert source.raw_count == 10_833_597
    assert source.artifact_bytes == 1_237_363
    assert source.sha256 == "46f4cbb5408d48797269b783bbb560968b2bb98c54569240df1734b21fcf0eea"


def test_rdw_fleet_is_a_dated_final_nl_active_fleet_snapshot() -> None:
    source = OFFICIAL_SOURCES["rdw-nl-fleet"]

    assert source.release_id == "rdw-nl-registered-cars-20261021"
    assert source.source_id == "rdw-gekentekende-voertuigen"
    assert source.geography == "NL"
    assert source.measure.value == "active_fleet"
    assert source.publication_status.value == "final"
    assert source.direct_download is False
    assert source.parser_name == "rdw_registered_cars_csv_v1"
    assert source.dependency_group == "nl-rdw-vehicle-register"
    assert source.raw_count == source.accepted_count + source.rejected_count
    assert "CC0" in source.permitted_local_use


def test_traficom_fleet_is_a_pinned_direct_download_fi_active_fleet_zip() -> None:
    source = OFFICIAL_SOURCES["traficom-fi-fleet"]

    assert source.release_id == "traficom-fi-vehicles-20260630"
    assert source.source_id == "traficom-open-vehicle-data"
    assert source.publisher == "Traficom (Finnish Transport and Communications Agency)"
    assert source.geography == "FI"
    assert source.measure.value == "active_fleet"
    assert source.publication_status.value == "final"
    assert source.direct_download is True
    assert source.suffix == ".zip"
    assert source.coverage_end.isoformat() == "2026-06-30"
    assert source.parser_name == "traficom_vehicles_zip_v1"
    assert source.dependency_group == "fi-traficom-vehicle-register"
    assert source.raw_count == source.accepted_count + source.rejected_count
    assert "Attribution 4.0" in source.permitted_local_use
    assert "Traficom" in source.permitted_local_use
    assert len(source.sha256) == 64
    validate_source_url(source, "https://opendata.traficom.fi/Content/Ajoneuvorekisteri.zip")
    with pytest.raises(ValueError, match="allowlisted"):
        validate_source_url(source, "https://example.com/Content/Ajoneuvorekisteri.zip")


def test_source_url_rejects_changed_host_or_path():
    source = OFFICIAL_SOURCES["eea-2024-final"]
    with pytest.raises(ValueError, match="allowlisted"):
        validate_source_url(source, "https://example.test/payload.zip")
    with pytest.raises(ValueError, match="allowlisted"):
        validate_source_url(source, source.url + "/changed")


def test_manifest_requires_the_pinned_artifact(tmp_path: Path):
    artifact = tmp_path / "release.zip"
    artifact.write_bytes(b"not the official release")
    with pytest.raises(ValueError, match="checksum"):
        build_manifest(
            OFFICIAL_SOURCES["eea-2024-final"],
            artifact,
            retrieved_at=datetime(2026, 8, 27, tzinfo=UTC),
        )

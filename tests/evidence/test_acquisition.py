from datetime import UTC, datetime
from pathlib import Path

import pytest

from icor.evidence.acquisition import OFFICIAL_SOURCES, build_manifest, validate_source_url


def test_catalog_contains_only_reviewed_https_sources():
    expected = {
        "eea-2024-final",
        "kba-fz10-2024",
        "uk-veh0160-gb",
        "uk-veh0120-gb",
        "uk-veh0124-am",
        "uk-veh0124-nz",
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

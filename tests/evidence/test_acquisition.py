from datetime import UTC, datetime
from pathlib import Path

import pytest

from icor.evidence.acquisition import OFFICIAL_SOURCES, build_manifest, validate_source_url


def test_catalog_contains_only_reviewed_https_sources():
    assert set(OFFICIAL_SOURCES) == {
        "eea-2024-final",
        "kba-fz10-2024",
        "uk-veh0160-gb",
        "uk-veh0120-gb",
    }
    for source in OFFICIAL_SOURCES.values():
        validate_source_url(source, source.url)


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

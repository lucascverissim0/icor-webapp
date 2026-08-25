from pathlib import Path

import pytest

from icor.config import ConfigurationError, load_settings


def test_load_settings_uses_repository_relative_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path, {})
    assert settings.environment == "local"
    assert settings.data_dir == tmp_path / "data"
    assert settings.output_dir == tmp_path / ".local" / "outputs"
    assert settings.external_network_enabled is False


def test_load_settings_rejects_unknown_environment(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="ICOR_ENVIRONMENT"):
        load_settings(tmp_path, {"ICOR_ENVIRONMENT": "mystery"})


def test_network_requires_explicit_true(tmp_path: Path) -> None:
    settings = load_settings(tmp_path, {"ICOR_EXTERNAL_NETWORK": "true"})
    assert settings.external_network_enabled is True

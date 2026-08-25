from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when a non-secret runtime setting is invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    environment: str
    data_dir: Path
    output_dir: Path
    external_network_enabled: bool


def load_settings(root: Path, environ: Mapping[str, str] | None = None) -> Settings:
    values = os.environ if environ is None else environ
    environment = values.get("ICOR_ENVIRONMENT", "local").strip().lower()
    if environment not in {"local", "test", "production"}:
        raise ConfigurationError("ICOR_ENVIRONMENT must be one of: local, test, production")

    network_value = values.get("ICOR_EXTERNAL_NETWORK", "false").strip().lower()
    if network_value not in {"true", "false"}:
        raise ConfigurationError("ICOR_EXTERNAL_NETWORK must be true or false")

    resolved_root = root.resolve()
    return Settings(
        environment=environment,
        data_dir=resolved_root / "data",
        output_dir=resolved_root / ".local" / "outputs",
        external_network_enabled=network_value == "true",
    )

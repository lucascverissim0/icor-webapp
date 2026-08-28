"""Fail-closed security configuration for the authenticated ICOR preview."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass

USERS_VARIABLE = "ICOR_PREVIEW_USERS"
SESSION_SECRET_VARIABLE = "ICOR_PREVIEW_SESSION_SECRET"
SESSION_TTL_VARIABLE = "ICOR_PREVIEW_SESSION_TTL_SECONDS"
DEFAULT_SESSION_TTL_SECONDS = 3_600
MINIMUM_SESSION_TTL_SECONDS = 300
MAXIMUM_SESSION_TTL_SECONDS = 43_200
MINIMUM_SESSION_SECRET_BYTES = 32


class ConfigurationError(ValueError):
    """Preview configuration is absent, malformed, or unsafe."""


@dataclass(frozen=True, slots=True)
class PreviewUser:
    """One named preview user and its Argon2id verifier."""

    username: str
    password_hash: str


@dataclass(frozen=True, slots=True)
class PreviewSettings:
    """Validated preview-only authentication settings."""

    users: tuple[PreviewUser, ...]
    session_secret: bytes
    session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> PreviewSettings:
        users = _users(environment.get(USERS_VARIABLE))
        secret = _session_secret(environment.get(SESSION_SECRET_VARIABLE))
        ttl = _session_ttl(environment.get(SESSION_TTL_VARIABLE))
        return cls(users=users, session_secret=secret, session_ttl_seconds=ttl)


def _users(value: str | None) -> tuple[PreviewUser, ...]:
    if not value:
        raise ConfigurationError("preview users configuration is required")
    try:
        payload = json.loads(value)
    except (json.JSONDecodeError, TypeError) as error:
        raise ConfigurationError("preview users configuration is invalid") from error
    if not isinstance(payload, dict) or not payload:
        raise ConfigurationError("preview users configuration is invalid")

    users: list[PreviewUser] = []
    normalized_names: set[str] = set()
    for raw_username, password_hash in payload.items():
        if not isinstance(raw_username, str) or not isinstance(password_hash, str):
            raise ConfigurationError("preview users configuration is invalid")
        username = raw_username.strip()
        normalized = username.casefold()
        if (
            not username
            or len(username) > 64
            or any(character.isspace() and character != " " for character in username)
            or normalized in normalized_names
            or not password_hash.startswith("$argon2id$")
        ):
            raise ConfigurationError("preview users configuration is invalid")
        normalized_names.add(normalized)
        users.append(PreviewUser(username=username, password_hash=password_hash))
    return tuple(users)


def _session_secret(value: str | None) -> bytes:
    if not value:
        raise ConfigurationError("preview session secret is required")
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.b64decode(
            (value + padding).encode("ascii"), altchars=b"-_", validate=True
        )
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ConfigurationError("preview session secret is invalid") from error
    if len(decoded) < MINIMUM_SESSION_SECRET_BYTES:
        raise ConfigurationError("preview session secret is too weak")
    return decoded


def _session_ttl(value: str | None) -> int:
    if value is None:
        return DEFAULT_SESSION_TTL_SECONDS
    try:
        ttl = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError("preview session lifetime is invalid") from error
    if not MINIMUM_SESSION_TTL_SECONDS <= ttl <= MAXIMUM_SESSION_TTL_SECONDS:
        raise ConfigurationError("preview session lifetime is outside the allowed range")
    return ttl

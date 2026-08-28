from __future__ import annotations

import base64

import pytest

from icor.preview.config import ConfigurationError, PreviewSettings

USER_HASH = "$argon2id$v=19$m=65536,t=3,p=4$c2FsdA$dmVyaWZpZXI"
MANAGER_HASH = "$argon2id$v=19$m=65536,t=3,p=4$bWFuYWdlcg$dmVyaWZpZXI"
SESSION_SECRET = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"


def _environment(**overrides: str) -> dict[str, str]:
    environment = {
        "ICOR_PREVIEW_USERS": (
            '{" Lucas ":"' + USER_HASH + '","manager":"' + MANAGER_HASH + '"}'
        ),
        "ICOR_PREVIEW_SESSION_SECRET": SESSION_SECRET,
    }
    environment.update(overrides)
    return environment


def test_valid_settings_normalize_users_and_decode_secret() -> None:
    settings = PreviewSettings.from_environment(
        _environment(ICOR_PREVIEW_SESSION_TTL_SECONDS="7200")
    )

    assert tuple(user.username for user in settings.users) == ("Lucas", "manager")
    assert tuple(user.password_hash for user in settings.users) == (
        USER_HASH,
        MANAGER_HASH,
    )
    assert settings.session_secret == bytes(range(32))
    assert settings.session_ttl_seconds == 7200


def test_settings_default_session_ttl_is_one_hour() -> None:
    assert PreviewSettings.from_environment(_environment()).session_ttl_seconds == 3600


@pytest.mark.parametrize(
    ("environment", "secret_marker"),
    (
        ({}, None),
        (
            {
                "ICOR_PREVIEW_USERS": "not-json",
                "ICOR_PREVIEW_SESSION_SECRET": SESSION_SECRET,
            },
            "not-json",
        ),
        (
            {
                "ICOR_PREVIEW_USERS": "[]",
                "ICOR_PREVIEW_SESSION_SECRET": SESSION_SECRET,
            },
            "[]",
        ),
        (_environment(ICOR_PREVIEW_USERS='{"":"' + USER_HASH + '"}'), USER_HASH),
        (
            _environment(
                ICOR_PREVIEW_USERS=(
                    '{"Lucas":"' + USER_HASH + '","lucas":"' + MANAGER_HASH + '"}'
                )
            ),
            MANAGER_HASH,
        ),
        (_environment(ICOR_PREVIEW_USERS='{"lucas":"plaintext-password"}'), "plaintext-password"),
        (_environment(ICOR_PREVIEW_SESSION_SECRET="weak-secret"), "weak-secret"),
        (_environment(ICOR_PREVIEW_SESSION_SECRET="not_base64!!"), "not_base64!!"),
        (_environment(ICOR_PREVIEW_SESSION_TTL_SECONDS="299"), "299"),
        (_environment(ICOR_PREVIEW_SESSION_TTL_SECONDS="43201"), "43201"),
        (_environment(ICOR_PREVIEW_SESSION_TTL_SECONDS="one-hour"), "one-hour"),
    ),
)
def test_invalid_settings_fail_closed_without_secret_leakage(
    environment: dict[str, str], secret_marker: str | None
) -> None:
    with pytest.raises(ConfigurationError) as captured:
        PreviewSettings.from_environment(environment)

    message = str(captured.value)
    if secret_marker:
        assert secret_marker not in message
    for value in environment.values():
        if value and len(value) > 8:
            assert value not in message


def test_session_secret_requires_at_least_32_decoded_bytes() -> None:
    short_secret = base64.urlsafe_b64encode(bytes(range(31))).decode().rstrip("=")

    with pytest.raises(ConfigurationError, match="session secret"):
        PreviewSettings.from_environment(
            _environment(ICOR_PREVIEW_SESSION_SECRET=short_secret)
        )

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest
from argon2 import PasswordHasher
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from icor.preview.auth import LoginThrottle, PreviewAuthenticator, SessionCodec
from icor.preview.config import PreviewSettings, PreviewUser
from icor.preview.security import PreviewSecurityMiddleware, SecurityHeadersMiddleware

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])
NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
SECRET = bytes(range(32))


def _hasher() -> PasswordHasher:
    return PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1)


def _settings() -> tuple[PreviewSettings, PasswordHasher]:
    hasher = _hasher()
    return (
        PreviewSettings(
            users=(
                PreviewUser("Lucas", hasher.hash("lucas-password")),
                PreviewUser("manager", hasher.hash("manager-password")),
            ),
            session_secret=SECRET,
            session_ttl_seconds=3600,
        ),
        hasher,
    )


def test_session_round_trip_and_key_rotation() -> None:
    codec = SessionCodec(SECRET, 3600, nonce_factory=lambda size: b"n" * size)
    token = codec.issue("Lucas", NOW)

    assert codec.verify(token, NOW + timedelta(minutes=30)) == "Lucas"
    assert SessionCodec(b"x" * 32, 3600).verify(token, NOW) is None


@pytest.mark.parametrize("token", ("", "not-a-token", "a.b.c", "!.!"))
def test_session_rejects_malformed_tokens(token: str) -> None:
    assert SessionCodec(SECRET, 3600).verify(token, NOW) is None


def test_session_rejects_expired_and_tampered_tokens() -> None:
    codec = SessionCodec(SECRET, 300, nonce_factory=lambda size: b"n" * size)
    token = codec.issue("Lucas", NOW)
    payload, signature = token.split(".")
    replacement = "A" if payload[-1] != "A" else "B"
    tampered = f"{payload[:-1]}{replacement}.{signature}"

    assert codec.verify(token, NOW + timedelta(seconds=301)) is None
    assert codec.verify(tampered, NOW) is None


def test_authenticator_accepts_each_named_user_and_rejects_generic_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings, hasher = _settings()
    authenticator = PreviewAuthenticator(settings, password_hasher=hasher)
    caplog.set_level(logging.DEBUG)

    assert authenticator.verify(" lucas ", "lucas-password")
    assert authenticator.verify("MANAGER", "manager-password")
    assert not authenticator.verify("missing", "submitted-secret")
    assert not authenticator.verify("Lucas", "wrong-secret")
    captured = caplog.text
    for secret in ("submitted-secret", "wrong-secret", settings.users[0].password_hash):
        assert secret not in captured


def test_throttle_limits_failures_resets_and_evicts_stale_buckets() -> None:
    throttle = LoginThrottle(SECRET, max_attempts=5, window_seconds=900, max_buckets=2)
    key = throttle.key("Lucas", "203.0.113.10")

    for second in range(5):
        assert throttle.allow(key, float(second))
        throttle.record_failure(key, float(second))
    assert not throttle.allow(key, 5.0)
    throttle.reset(key)
    assert throttle.allow(key, 6.0)

    stale = throttle.key("stale", "203.0.113.11")
    throttle.record_failure(stale, 0.0)
    current = throttle.key("current", "203.0.113.12")
    throttle.record_failure(current, 901.0)
    assert throttle.bucket_count == 1


def _protected_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    codec = SessionCodec(SECRET, 3600, nonce_factory=lambda size: b"n" * size)

    @app.get("/{path:path}")
    def route(request: Request, path: str) -> dict[str, object]:
        return {
            "path": request.url.path,
            "username": getattr(request.state, "preview_username", None),
        }

    app.add_middleware(PreviewSecurityMiddleware, session_codec=codec)
    app.add_middleware(SecurityHeadersMiddleware)
    app.state.codec = codec
    return app


@pytest.mark.parametrize("path", ("/healthz", "/auth/login"))
def test_middleware_allows_only_anonymous_boundaries(path: str) -> None:
    with TestClient(_protected_app()) as client:
        assert client.get(path).status_code == 200


@pytest.mark.parametrize(
    "path",
    ("/", "/assets/app.js", "/api/data", "/docs", "/openapi.json", "/api/exports/ml.csv"),
)
def test_middleware_protects_application_surfaces(path: str) -> None:
    with TestClient(_protected_app()) as client:
        response = client.get(path)

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


def test_middleware_attaches_authenticated_user_and_security_headers() -> None:
    app = _protected_app()
    token = app.state.codec.issue("Lucas", datetime.now(UTC))
    with TestClient(app) as client:
        client.cookies.set("icor_preview_session", token)
        response = client.get("/planner")

    assert response.status_code == 200
    assert response.json()["username"] == "Lucas"
    assert response.headers["content-security-policy"] == (
        "default-src 'self'; object-src 'none'; frame-ancestors 'none'"
    )
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"

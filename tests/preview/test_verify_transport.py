"""The release verifier's transport seam.

`scripts/verify_client_release.py` asserts the client-release gates against a
deployed URL. Every check after sign-in depends on the session cookie coming
back, and that cookie is `Secure`, so none of them can run over local plain
HTTP -- which meant the gates were first exercised on a paid deployment.

Driving the same checks over an in-process ASGI transport with an `https`
scope fixes that: `_is_tls` is true, so HSTS is emitted and the cookie is
returned. These tests pin the seam itself -- status, header casing, cookie
round-trip, redirect suppression -- against a stub API, not the 1.8 GB
artifact. The real snapshot is exercised by running the script with --local.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from argon2 import PasswordHasher

from icor.preview.config import PreviewSettings, PreviewUser
from scripts.verify_client_release import (
    LOCAL_DEFERRED,
    Checker,
    _AsgiTransport,
    _check_transport,
    _sign_in,
    _verdict,
)
from tests.preview.test_app import _preview

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])

PASSWORD = "correct horse battery staple"


@pytest.fixture
def settings() -> PreviewSettings:
    hasher = PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1)
    return PreviewSettings(
        users=(PreviewUser("client-reviewer", hasher.hash(PASSWORD)),),
        session_secret=b"s" * 32,
        session_ttl_seconds=3600,
    )


@pytest.fixture
def assets(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<main>ICOR application</main>", encoding="utf-8")
    (root / "assets" / "app-123.js").write_text("export {};", encoding="utf-8")
    return root


@pytest.fixture
def checker(
    monkeypatch: pytest.MonkeyPatch, settings: PreviewSettings, assets: Path
) -> Checker:
    """An internal preview, where a served API path answers 200 once signed in."""

    app = _preview(monkeypatch, settings, assets, client_release=False)
    with _AsgiTransport(app) as transport:
        yield Checker(transport.base_url, transport=transport)


@pytest.fixture
def client_checker(
    monkeypatch: pytest.MonkeyPatch, settings: PreviewSettings, assets: Path
) -> Checker:
    app = _preview(monkeypatch, settings, assets, client_release=True)
    with _AsgiTransport(app) as transport:
        yield Checker(transport.base_url, transport=transport)


def test_the_transport_reports_status_lowercased_headers_and_bytes(
    checker: Checker,
) -> None:
    status, headers, body = checker.request("/auth/login")

    assert status == 200
    assert isinstance(body, bytes)
    # The checks index headers case-insensitively; both transports must agree.
    assert all(key == key.lower() for key in headers)
    assert "content-security-policy" in headers


def test_an_https_scope_emits_hsts_so_gate5_can_be_exercised(
    checker: Checker,
) -> None:
    _check_transport(checker)

    recorded = {item["check"]: item["passed"] for item in checker.results}
    assert recorded["gate5:hsts-present"] is True
    assert recorded["gate5:noindex"] is True
    assert recorded["gate5:csp-pins-base-and-form"] is True


def test_sign_in_is_observed_as_a_redirect_and_not_followed(
    checker: Checker,
) -> None:
    status, _, _ = checker.request(
        "/auth/login",
        method="POST",
        data=b"username=client-reviewer&password=correct+horse+battery+staple",
    )

    # 200 here would mean the transport followed the redirect and the check
    # `sign-in` would silently pass for the wrong reason.
    assert status == 303


def test_the_secure_session_cookie_survives_the_round_trip(
    checker: Checker,
) -> None:
    """The whole point: authenticated checks are impossible over plain HTTP."""

    before, _, _ = checker.request("/api/example")
    assert before == 401

    assert _sign_in(checker, "client-reviewer", PASSWORD) is True

    after, _, _ = checker.request("/api/example")
    assert after == 200


def test_the_blocked_surface_404s_only_once_authenticated(
    client_checker: Checker,
) -> None:
    """Smoke item 6, which is worded "after signing in" for a reason.

    Anonymously the path is indistinguishable from any other (401); it is the
    authenticated response that must be 404. Local plain HTTP can only ever
    show the 401 half, which is why this check needed the https scope.
    """

    before, _, _ = client_checker.request("/api/example")
    assert before == 401

    assert _sign_in(client_checker, "client-reviewer", PASSWORD) is True

    after, _, _ = client_checker.request("/api/example")
    assert after == 404


def test_a_local_verdict_is_never_the_deployed_verdict(checker: Checker) -> None:
    """A green laptop run must not be fileable as the release evidence.

    Both gate 5 checks pass in an https ASGI scope without a byte of TLS, so
    `failed: 0` alone is indistinguishable between the two modes. The mode,
    the verdict word and the per-check evidence are what separate them.
    """

    _check_transport(checker)

    local = _verdict(checker, "local", {"snapshot_id": "snapshot-test"})

    assert local["failed"] == 0
    assert local["verdict"] == "local-preflight-passed"
    assert local["proves_tls"] is False
    assert local["not_proven_locally"] == list(LOCAL_DEFERRED)
    asserted = {
        item["check"] for item in local["results"] if item["evidence"] == "asserted"
    }
    assert asserted == set(LOCAL_DEFERRED)


def test_a_deployed_verdict_observes_every_check(checker: Checker) -> None:
    _check_transport(checker)

    deployed = _verdict(checker, "deployed")

    assert deployed["verdict"] == "deployed-release-verified"
    assert deployed["proves_tls"] is True
    assert deployed["asserted"] == 0
    assert deployed["not_proven_locally"] == []
    assert all(item["evidence"] == "observed" for item in deployed["results"])


def test_a_failing_check_is_never_reported_as_verified(checker: Checker) -> None:
    checker.record("invented:failure", False, "")

    assert _verdict(checker, "deployed")["verdict"] == "failed"
    assert _verdict(checker, "local")["verdict"] == "failed"


def test_sign_out_revokes_the_session(checker: Checker) -> None:
    assert _sign_in(checker, "client-reviewer", PASSWORD) is True

    checker.request("/auth/logout", method="POST")

    status, _, _ = checker.request("/api/example")
    assert status == 401

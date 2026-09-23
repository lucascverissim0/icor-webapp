#!/usr/bin/env python3
"""Check a client preview against the release gates and smoke test.

Takes the URL and a credential on stdin, signs in once, and asserts the gates in
docs/CLIENT_RELEASE.md that can be checked mechanically, plus the seven smoke
test steps. Prints one JSON verdict, so the same check is repeatable after every
redeploy and its output can be pasted into the handoff as evidence.

The password is read from stdin and never appears in a command line, a process
listing, a shell history or this file.

With --local the same checks run in process against the real client artifacts,
over an ASGI scope that declares https. That is what makes the authenticated
half checkable before a deploy: the session cookie is Secure, so it is never
returned over plain HTTP. It declares TLS rather than negotiating it, so it is
a pre-flight and says nothing about gate 5 -- the verdict names which it is.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any

BLOCKED_PATHS = (
    "/evidence",
    "/registrations",
    "/completeness",
    "/exports",
    "/docs",
    "/openapi.json",
)


FORM_TYPE = "application/x-www-form-urlencoded"

# The two gate 5 checks an in-process run cannot prove. An ASGI scope can
# declare https without a certificate, so both are structurally true locally
# and say nothing about the deployment's real TLS or the fly-proxy path.
LOCAL_DEFERRED = ("gate5:url-is-https", "gate5:hsts-present")


def _lowered(headers: Any) -> dict[str, str]:
    """Both transports hand the checks the same case-insensitive mapping."""

    return {str(key).lower(): str(value) for key, value in dict(headers).items()}


class _UrllibTransport:
    """Real HTTP against a deployed URL."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.jar = CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar),
            _NoRedirect(),
        )

    def open(
        self, path: str, method: str, data: bytes | None
    ) -> tuple[int, dict[str, str], bytes]:
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method
        )
        if data is not None:
            request.add_header("Content-Type", FORM_TYPE)
        try:
            with self.opener.open(request, timeout=30) as response:
                return response.status, _lowered(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, _lowered(error.headers), error.read()


class _AsgiTransport:
    """The same checks, in process, over an https scope.

    The session cookie is `Secure` and HSTS is emitted only when the request
    scheme is https (src/icor/preview/security.py:48-52, :85-86), so a local
    plain-HTTP run cannot exercise anything after sign-in. An ASGI scope can
    declare https without a certificate, which is what makes a local
    pre-flight of the authenticated gates possible at all.
    """

    base_url = "https://local.verify"

    def __init__(self, app: Any) -> None:
        from fastapi.testclient import TestClient

        self._client = TestClient(app, base_url=self.base_url, follow_redirects=False)

    def __enter__(self) -> _AsgiTransport:
        self._client.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._client.__exit__(*exc)

    def open(
        self, path: str, method: str, data: bytes | None
    ) -> tuple[int, dict[str, str], bytes]:
        response = self._client.request(
            method,
            path,
            content=data,
            headers={"Content-Type": FORM_TYPE} if data is not None else None,
        )
        return response.status_code, _lowered(response.headers), response.content


class Checker:
    def __init__(self, base_url: str, *, transport: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _UrllibTransport(base_url)
        self.results: list[dict[str, Any]] = []

    def record(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append({"check": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    def request(
        self, path: str, *, method: str = "GET", data: bytes | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        return self.transport.open(path, method, data)

    def json(self, path: str) -> tuple[int, Any]:
        status, _, body = self.request(path)
        try:
            return status, json.loads(body)
        except ValueError:
            return status, None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _check_transport(checker: Checker) -> None:
    """Gate 5: HTTPS only, and the security headers that make it stick."""

    checker.record(
        "gate5:url-is-https", checker.base_url.startswith("https://"), checker.base_url
    )
    status, headers, _ = checker.request("/auth/login")
    checker.record(
        "gate5:hsts-present",
        "strict-transport-security" in {k.lower() for k in headers},
        "",
    )
    lowered = {k.lower(): v for k, v in headers.items()}
    checker.record(
        "gate5:noindex",
        lowered.get("x-robots-tag", "").startswith("noindex"),
        lowered.get("x-robots-tag", ""),
    )
    policy = lowered.get("content-security-policy", "")
    checker.record(
        "gate5:csp-pins-base-and-form",
        "base-uri 'none'" in policy and "form-action 'self'" in policy,
        policy[:120],
    )
    checker.record("login-form-reachable", status == 200, str(status))


def _check_unauthenticated(checker: Checker) -> None:
    """Nothing but the login page and health may answer before sign-in."""

    status, _, _ = checker.request("/api/v1/opportunities")
    checker.record("anonymous:ranking-is-401", status == 401, str(status))
    blocked, _, _ = checker.request("/api/v1/registrations/ranking")
    checker.record(
        "anonymous:surface-not-enumerable",
        blocked == status,
        f"blocked={blocked} allowed={status}",
    )


def _sign_in(checker: Checker, username: str, password: str) -> bool:
    payload = urllib.parse.urlencode(
        {"username": username, "password": password}
    ).encode()
    status, _, _ = checker.request("/auth/login", method="POST", data=payload)
    return checker.record("sign-in", status in {303, 302}, str(status))


def _check_smoke(checker: Checker) -> None:
    status, health = checker.json("/api/health")
    checker.record("gate2:health-ready", status == 200 and bool(health), str(status))

    status, summary = checker.json("/api/v1/registrations/summary")
    if status == 200 and isinstance(summary, dict):
        versions = summary.get("versions") or {}
        checker.record(
            "gate7:uncertainty-method-exposed",
            "uncertainty_method" in versions,
            ",".join(sorted(versions)),
        )
        checker.record(
            "gate2:snapshot-id-reported",
            bool(summary.get("snapshot_id") or summary.get("data_version")),
            str(summary.get("snapshot_id") or summary.get("data_version")),
        )
    else:
        checker.record("gate7:uncertainty-method-exposed", False, str(status))

    status, page = checker.json(
        "/api/v1/opportunities?group_by=model_year&page_size=25"
    )
    if status == 200 and isinstance(page, dict):
        items = page.get("items") or []
        checker.record("smoke2:paginates", int(page.get("pages", 0)) > 1, str(page.get("pages")))
        distinct = {(item.get("brand"), item.get("model")) for item in items}
        checker.record("smoke2:multiple-models", len(distinct) > 1, str(len(distinct)))
        checker.record(
            "smoke3:no-estimated-generation-labels",
            "estimated-generation-" not in json.dumps(page),
            "",
        )
    else:
        checker.record("smoke2:paginates", False, str(status))

    # Gate 4: a mode assertion no other configuration can fake.
    status, _ = checker.json("/api/v1/opportunities?group_by=brand")
    checker.record("gate4:client-release-scope-enforced", status == 422, str(status))

    for path in BLOCKED_PATHS:
        code, _, _ = checker.request(path)
        checker.record(f"smoke6:blocked{path}", code == 404, str(code))


def _check_sign_out(checker: Checker) -> None:
    checker.request("/auth/logout", method="POST")
    status, _, _ = checker.request("/api/v1/opportunities")
    checker.record("smoke7:sign-out-revokes-access", status == 401, str(status))


def _local_app(username: str) -> tuple[Any, str, dict[str, Any]]:
    """The real client preview, built in process over the real artifacts.

    Generates its own throwaway credential, so a local pre-flight needs no
    stored secret and no deployed environment, and reads nothing from stdin.
    Runs the same `validate_runner` preflight the container entrypoint runs --
    building the app directly would skip it, and skipping it is how a local
    run starts proving less than it appears to.
    """

    import secrets
    import time

    from argon2 import PasswordHasher

    from icor.api.app import ROOT
    from icor.preview.app import create_preview_app
    from icor.preview.runner import validate_runner

    try:
        from scripts.generate_preview_credentials import session_secret
    except ModuleNotFoundError:  # run as a script rather than imported as a package
        from generate_preview_credentials import session_secret

    asset_root = ROOT / ".local" / "client-release"
    snapshot_root = ROOT / ".local" / "client-evidence"
    coverage_db = Path(
        os.environ.get(
            "ICOR_COVERAGE_DB", str(ROOT / ".local" / "production-coverage.sqlite3")
        )
    )

    password = secrets.token_urlsafe(32)
    environment = {
        **os.environ,
        "ICOR_PREVIEW_HOST_MODE": "container",
        "ICOR_CLIENT_RELEASE_MODE": "verified",
        "ICOR_PREVIEW_PUBLIC_ORIGIN": os.environ.get(
            "ICOR_PREVIEW_PUBLIC_ORIGIN", "https://icor-client-preview.fly.dev"
        ),
        "ICOR_PREVIEW_TRUSTED_PROXIES": "127.0.0.1",
        "ICOR_PREVIEW_USERS": json.dumps({username: PasswordHasher().hash(password)}),
        "ICOR_PREVIEW_SESSION_SECRET": session_secret(),
        "ICOR_EXPORT_TOKEN": session_secret(),
    }
    settings = validate_runner(
        environment,
        asset_root=asset_root,
        snapshot_root=snapshot_root,
        coverage_db=coverage_db,
    )
    os.environ["ICOR_COVERAGE_DB"] = str(coverage_db)

    started = time.monotonic()
    app = create_preview_app(
        settings,
        asset_root=asset_root,
        snapshot_root=snapshot_root,
        client_release=True,
    )
    # Opening the snapshot and warming the model-year population are what the
    # health check's grace period has to cover, so report them, not just pass.
    boot_seconds = round(time.monotonic() - started, 1)
    manifest = getattr(app.state, "snapshot_manifest", None)
    target = {
        "asset_root": str(asset_root),
        "snapshot_root": str(snapshot_root),
        "snapshot_id": getattr(manifest, "snapshot_id", None),
        "snapshot_scope": getattr(manifest, "scope", None),
        "public_origin_declared": environment["ICOR_PREVIEW_PUBLIC_ORIGIN"],
        "trust_baked_snapshot": False,
        "boot_seconds": boot_seconds,
    }
    return app, password, target


def _run(checker: Checker, username: str, password: str) -> None:
    _check_transport(checker)
    _check_unauthenticated(checker)
    if _sign_in(checker, username, password):
        _check_smoke(checker)
        _check_sign_out(checker)


def _verdict(
    checker: Checker, mode: str, target: dict[str, Any] | None = None
) -> dict[str, Any]:
    local = mode == "local"
    results = checker.results
    for item in results:
        # An in-process scope declares https; it does not negotiate it. Every
        # other check ran against real middleware and is observed.
        item["evidence"] = (
            "asserted" if local and item["check"] in LOCAL_DEFERRED else "observed"
        )
    failed = [item for item in results if not item["passed"]]
    verdict: dict[str, Any] = {
        "url": checker.base_url,
        "mode": mode,
        # One word that cannot be mistaken for the other, so a green laptop run
        # can never be filed as the release evidence.
        "verdict": (
            ("local-preflight-passed" if local else "deployed-release-verified")
            if not failed
            else "failed"
        ),
        "proves_tls": not local,
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "asserted": sum(1 for item in results if item["evidence"] == "asserted"),
        "not_proven_locally": list(LOCAL_DEFERRED) if local else [],
        "results": results,
    }
    if target is not None:
        verdict["target"] = target
    if local:
        verdict["note"] = (
            "In-process pre-flight against the real client artifacts. The https "
            "scheme was declared by the transport, not negotiated, so HSTS was "
            "emitted because of that declaration: gate 5 is proven only by a "
            "deployed run. Smoke points 1, 4 and 5 are visual and are not "
            "covered here."
        )
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", help="verify a deployed preview over real HTTPS")
    target.add_argument(
        "--local",
        action="store_true",
        help="pre-flight the real local artifacts in process (not a substitute)",
    )
    parser.add_argument("--username", required=True)
    args = parser.parse_args(argv)

    if args.local:
        app, password, target = _local_app(args.username)
        with _AsgiTransport(app) as transport:
            checker = Checker(transport.base_url, transport=transport)
            _run(checker, args.username, password)
            verdict = _verdict(checker, "local", target)
        print(verdict["note"], file=sys.stderr)
    else:
        password = sys.stdin.readline().rstrip("\n")
        if not password:
            print("no password on stdin", file=sys.stderr)
            return 2
        checker = Checker(args.url)
        _run(checker, args.username, password)
        verdict = _verdict(checker, "deployed")

    print(json.dumps(verdict, indent=2))
    return 1 if verdict["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Check a deployed client preview against the release gates and smoke test.

Takes the URL and a credential on stdin, signs in once, and asserts the gates in
docs/CLIENT_RELEASE.md that can be checked mechanically, plus the seven smoke
test steps. Prints one JSON verdict, so the same check is repeatable after every
redeploy and its output can be pasted into the handoff as evidence.

The password is read from stdin and never appears in a command line, a process
listing, a shell history or this file.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from typing import Any

BLOCKED_PATHS = (
    "/evidence",
    "/registrations",
    "/completeness",
    "/exports",
    "/docs",
    "/openapi.json",
)


class Checker:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.jar = CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar),
            _NoRedirect(),
        )
        self.results: list[dict[str, Any]] = []

    def record(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append({"check": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    def request(
        self, path: str, *, method: str = "GET", data: bytes | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method
        )
        if data is not None:
            request.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with self.opener.open(request, timeout=30) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--username", required=True)
    args = parser.parse_args(argv)

    password = sys.stdin.readline().rstrip("\n")
    if not password:
        print("no password on stdin", file=sys.stderr)
        return 2

    checker = Checker(args.url)
    _check_transport(checker)
    _check_unauthenticated(checker)
    if _sign_in(checker, args.username, password):
        _check_smoke(checker)
        _check_sign_out(checker)

    failed = [item for item in checker.results if not item["passed"]]
    print(
        json.dumps(
            {
                "url": checker.base_url,
                "passed": len(checker.results) - len(failed),
                "failed": len(failed),
                "results": checker.results,
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

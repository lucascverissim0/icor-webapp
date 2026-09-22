"""Authenticated, same-origin FastAPI composition for GitHub Codespaces preview."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from icor.api.app import DEFAULT_EVIDENCE_ROOT, ROOT, create_app
from icor.domain.snapshots import CLIENT_RELEASE_SCOPE, FULL_SCOPE
from icor.preview.auth import LoginThrottle, PreviewAuthenticator, SessionCodec
from icor.preview.config import ConfigurationError, PreviewSettings
from icor.preview.security import (
    SESSION_COOKIE,
    ClientReleaseMiddleware,
    PreviewSecurityMiddleware,
    SecurityHeadersMiddleware,
)
from icor.preview.static import resolve_asset

DEFAULT_ASSET_ROOT = ROOT / "web" / "dist"
ASSET_ROOT_VARIABLE = "ICOR_PREVIEW_ASSET_ROOT"
MAX_LOGIN_BODY_BYTES = 8_192
LOGIN_FORM = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ICOR preview sign in</title></head><body><main><h1>ICOR preview</h1>
<form method="post" action="/auth/login"><label>Username
<input name="username" autocomplete="username" required></label>
<label>Password
<input name="password" type="password" autocomplete="current-password" required></label>
<button type="submit">Sign in</button></form></main></body></html>"""


def create_preview_app(
    settings: PreviewSettings | None = None,
    *,
    asset_root: Path | None = None,
    snapshot_root: Path | None = None,
    client_release: bool | None = None,
) -> FastAPI:
    """Build the fail-closed Codespaces composition from validated runtime state."""
    selected_settings = settings or PreviewSettings.from_environment(os.environ)
    selected_snapshot_root = snapshot_root or Path(
        os.environ.get("ICOR_EVIDENCE_ACTIVE_ROOT", str(DEFAULT_EVIDENCE_ROOT))
    )
    selected_client_release = (
        _client_release_from_environment()
        if client_release is None
        else client_release
    )
    core = create_app(
        snapshot_root=selected_snapshot_root,
        client_release=selected_client_release,
    )
    manifest = getattr(core.state, "snapshot_manifest", None)
    if manifest is None:
        raise ConfigurationError("preview active snapshot is unavailable")
    # The mechanical guarantee that a client preview cannot serve the full
    # evidence corpus, and that an internal build cannot silently serve the
    # pruned one and report missing evidence as absent.
    expected_scope = CLIENT_RELEASE_SCOPE if selected_client_release else FULL_SCOPE
    if manifest.scope != expected_scope:
        raise ConfigurationError("preview snapshot scope does not match the release mode")
    selected_asset_root = asset_root or Path(
        os.environ.get(ASSET_ROOT_VARIABLE, str(DEFAULT_ASSET_ROOT))
    )
    return _compose(
        core,
        selected_settings,
        selected_asset_root,
        client_release=selected_client_release,
    )


def _compose(
    app: FastAPI,
    settings: PreviewSettings,
    asset_root: Path,
    *,
    client_release: bool = False,
) -> FastAPI:
    try:
        resolved_assets = asset_root.resolve(strict=True)
        index = (resolved_assets / "index.html").resolve(strict=True)
    except OSError as error:
        raise ConfigurationError("preview compiled frontend is unavailable") from error
    if not index.is_file() or not index.is_relative_to(resolved_assets):
        raise ConfigurationError("preview compiled frontend is unavailable")

    session_codec = SessionCodec(
        settings.session_secret, settings.session_ttl_seconds
    )
    authenticator = PreviewAuthenticator(settings)
    throttle = LoginThrottle(settings.session_secret)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/auth/login", include_in_schema=False)
    def login_form() -> HTMLResponse:
        return HTMLResponse(LOGIN_FORM, headers={"Cache-Control": "no-store"})

    @app.post("/auth/login", include_in_schema=False)
    async def login(request: Request):  # type: ignore[no-untyped-def]
        body = await request.body()
        if len(body) > MAX_LOGIN_BODY_BYTES:
            return JSONResponse(
                {"detail": "Login request is too large"},
                status_code=413,
                headers={"Cache-Control": "no-store"},
            )
        if request.headers.get("content-type", "").split(";", 1)[0].strip().casefold() != (
            "application/x-www-form-urlencoded"
        ):
            return _login_failure(400)
        try:
            fields = parse_qs(
                body.decode("utf-8"),
                keep_blank_values=True,
                max_num_fields=4,
                strict_parsing=True,
            )
            username = _single(fields, "username")
            password = _single(fields, "password")
        except (UnicodeDecodeError, ValueError):
            return _login_failure(400)

        address = request.client.host if request.client is not None else "unknown"
        throttle_key = throttle.key(username, address)
        address_key = throttle.address_key(address)
        now = time.monotonic()
        # Both buckets are checked before any hashing happens. The
        # address bucket is what an attacker rotating usernames hits.
        if not throttle.allow(throttle_key, now) or not throttle.allow(
            address_key, now
        ):
            return JSONResponse(
                {"detail": "Login temporarily unavailable"},
                status_code=429,
                headers={"Cache-Control": "no-store"},
            )
        # Argon2id is deliberately expensive: about 64 MiB and tens of
        # milliseconds per call, and an unknown user still pays it against
        # the dummy hash. Run on the event loop it is a remote stall.
        verified = await run_in_threadpool(
            authenticator.verify, username, password
        )
        if not verified:
            throttle.record_failure(throttle_key, now)
            throttle.record_failure(address_key, now)
            return _login_failure(401)

        throttle.reset(throttle_key)
        throttle.reset(address_key)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            session_codec.issue(username.strip(), datetime.now(UTC)),
            max_age=settings.session_ttl_seconds,
            secure=True,
            httponly=True,
            samesite="strict",
            path="/",
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/auth/logout", include_in_schema=False)
    def logout() -> RedirectResponse:
        response = RedirectResponse("/auth/login", status_code=303)
        response.delete_cookie(
            SESSION_COOKIE,
            path="/",
            secure=True,
            httponly=True,
            samesite="strict",
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.api_route("/{request_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def compiled_application(request_path: str):  # type: ignore[no-untyped-def]
        if request_path.startswith("api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        requested = resolve_asset(resolved_assets, request_path) if request_path else None
        if requested is not None:
            return FileResponse(requested)
        if request_path and PurePosixPath(request_path).suffix:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(index, media_type="text/html")

    app.state.preview_settings = settings
    # Starlette prepends, so the execution order is the reverse of this:
    # headers, then authentication, then the client-release path policy.
    # Authentication must run before the policy, or its 404s answer
    # unauthenticated callers and enumerate the client surface.
    if client_release:
        app.add_middleware(ClientReleaseMiddleware)
    app.add_middleware(PreviewSecurityMiddleware, session_codec=session_codec)
    app.add_middleware(SecurityHeadersMiddleware)
    return app


def _client_release_from_environment() -> bool:
    value = os.environ.get("ICOR_CLIENT_RELEASE_MODE", "").strip().casefold()
    if value not in {"", "verified"}:
        raise ConfigurationError("client release mode is invalid")
    return value == "verified"


def _single(fields: dict[str, list[str]], name: str) -> str:
    values = fields.get(name)
    if values is None or len(values) != 1:
        raise ValueError("invalid login form")
    return values[0]


def _login_failure(status_code: int) -> JSONResponse:
    return JSONResponse(
        {"detail": "Invalid username or password"},
        status_code=status_code,
        headers={"Cache-Control": "no-store"},
    )

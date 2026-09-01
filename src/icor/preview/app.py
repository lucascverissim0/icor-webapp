"""Authenticated, same-origin FastAPI composition for GitHub Codespaces preview."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from icor.api.app import DEFAULT_EVIDENCE_ROOT, ROOT, create_app
from icor.preview.auth import LoginThrottle, PreviewAuthenticator, SessionCodec
from icor.preview.config import ConfigurationError, PreviewSettings
from icor.preview.security import (
    SESSION_COOKIE,
    PreviewSecurityMiddleware,
    SecurityHeadersMiddleware,
)
from icor.preview.static import resolve_asset

DEFAULT_ASSET_ROOT = ROOT / "web" / "dist"
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
) -> FastAPI:
    """Build the fail-closed Codespaces composition from validated runtime state."""
    selected_settings = settings or PreviewSettings.from_environment(os.environ)
    selected_snapshot_root = snapshot_root or Path(
        os.environ.get("ICOR_EVIDENCE_ACTIVE_ROOT", str(DEFAULT_EVIDENCE_ROOT))
    )
    core = create_app(snapshot_root=selected_snapshot_root)
    if getattr(core.state, "snapshot_manifest", None) is None:
        raise ConfigurationError("preview active snapshot is unavailable")
    return _compose(core, selected_settings, asset_root or DEFAULT_ASSET_ROOT)


def _compose(
    app: FastAPI, settings: PreviewSettings, asset_root: Path
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
        now = time.monotonic()
        if not throttle.allow(throttle_key, now):
            return JSONResponse(
                {"detail": "Login temporarily unavailable"},
                status_code=429,
                headers={"Cache-Control": "no-store"},
            )
        if not authenticator.verify(username, password):
            throttle.record_failure(throttle_key, now)
            return _login_failure(401)

        throttle.reset(throttle_key)
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
    app.add_middleware(PreviewSecurityMiddleware, session_codec=session_codec)
    app.add_middleware(SecurityHeadersMiddleware)
    return app


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

"""ASGI authentication enforcement and preview security headers."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from icor.preview.auth import SessionCodec

SESSION_COOKIE = "icor_preview_session"
_ANONYMOUS_PATHS = frozenset({"/healthz", "/auth/login"})
# `base-uri` and `form-action` do not fall back to `default-src`, so without them
# the policy does nothing about a <base> injection retargeting every relative asset
# URL, or about the login form being pointed somewhere else. The login form posts
# credentials, which makes `form-action 'self'` the most valuable line here.
#
# The explicit directives are safe against the shipped bundle: it has no inline
# <script> and no inline style attribute. `test_the_compiled_bundle_stays_within_the
# _policy` keeps that true as the frontend changes.
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
        "font-src 'self'; connect-src 'self'; manifest-src 'self'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'; "
        "object-src 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "X-Robots-Tag": "noindex, nofollow, noarchive, nosnippet, noimageindex",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": (
        "accelerometer=(), camera=(), display-capture=(), geolocation=(), "
        "gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()"
    ),
}

# One year, no `preload`. Preload is a browser-vendor list submission that is
# effectively irreversible for a year, which is wrong for a temporary preview on a
# hostname the project does not own.
_STRICT_TRANSPORT_SECURITY = "max-age=31536000; includeSubDomains"


def _is_tls(request: Request) -> bool:
    """HSTS over plain HTTP is meaningless, and misleading in local runs."""

    forwarded = request.headers.get("x-forwarded-proto", "")
    return request.url.scheme == "https" or forwarded.split(",")[0].strip() == "https"


class PreviewSecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, session_codec: SessionCodec) -> None:
        super().__init__(app)
        self._session_codec = session_codec

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in _ANONYMOUS_PATHS:
            return await call_next(request)
        username = self._session_codec.verify(
            request.cookies.get(SESSION_COOKIE, ""), datetime.now(UTC)
        )
        if username is None:
            return JSONResponse(
                {"detail": "Authentication required"},
                status_code=401,
                headers={"Cache-Control": "no-store"},
            )
        request.state.preview_username = username
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        for name, value in _SECURITY_HEADERS.items():
            response.headers[name] = value
        if _is_tls(request):
            response.headers["Strict-Transport-Security"] = _STRICT_TRANSPORT_SECURITY
        path = request.url.path
        if path.startswith("/auth/") or path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response


class ClientReleaseMiddleware(BaseHTTPMiddleware):
    """Deny internal and mutable surfaces in the verified client preview."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path
        if path in {"/healthz", "/auth/login", "/auth/logout"}:
            return await call_next(request)
        if request.method not in {"GET", "HEAD"}:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        if path in {"/", "/opportunities", "/planner", "/api/health"}:
            return await call_next(request)
        if path.startswith("/opportunities/"):
            return await call_next(request)
        if path.startswith("/assets/"):
            return await call_next(request)
        if path.startswith("/api/v1/opportunities"):
            return await call_next(request)
        if path.startswith("/api/v1/vehicle-forecasts"):
            return await call_next(request)
        if path == "/api/v1/registrations/summary":
            return await call_next(request)
        return JSONResponse({"detail": "Not Found"}, status_code=404)

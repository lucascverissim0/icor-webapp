"""ASGI authentication enforcement and preview security headers."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from icor.preview.auth import SessionCodec

SESSION_COOKIE = "icor_preview_session"
_ANONYMOUS_PATHS = frozenset({"/healthz", "/auth/login"})
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; object-src 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


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
        if request.url.path.startswith("/auth/"):
            response.headers["Cache-Control"] = "no-store"
        return response

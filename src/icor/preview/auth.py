"""Password verification, signed sessions, and bounded login throttling."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from collections.abc import Callable
from datetime import UTC, datetime

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

from icor.preview.config import PreviewSettings


class SessionCodec:
    def __init__(
        self,
        secret: bytes,
        ttl_seconds: int,
        *,
        nonce_factory: Callable[[int], bytes] = secrets.token_bytes,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("session signing key must contain at least 32 bytes")
        if not 300 <= ttl_seconds <= 43_200:
            raise ValueError("session lifetime is outside the safe range")
        self._secret = secret
        self._ttl_seconds = ttl_seconds
        self._nonce_factory = nonce_factory

    def issue(self, username: str, now: datetime) -> str:
        timestamp = _utc_timestamp(now)
        payload = {
            "exp": timestamp + self._ttl_seconds,
            "iat": timestamp,
            "nonce": _encode(self._nonce_factory(16)),
            "username": username,
            "version": 1,
        }
        encoded = _encode(
            json.dumps(
                payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        )
        signature = _encode(
            hmac.digest(self._secret, encoded.encode("ascii"), hashlib.sha256)
        )
        return f"{encoded}.{signature}"

    def verify(self, token: str, now: datetime) -> str | None:
        try:
            encoded, supplied_signature = token.split(".")
            expected_signature = _encode(
                hmac.digest(self._secret, encoded.encode("ascii"), hashlib.sha256)
            )
            if not hmac.compare_digest(supplied_signature, expected_signature):
                return None
            payload = json.loads(_decode(encoded))
            if not isinstance(payload, dict) or set(payload) != {
                "exp",
                "iat",
                "nonce",
                "username",
                "version",
            }:
                return None
            issued_at = payload["iat"]
            expires_at = payload["exp"]
            username = payload["username"]
            if (
                payload["version"] != 1
                or type(issued_at) is not int
                or type(expires_at) is not int
                or not isinstance(username, str)
                or not username
                or not isinstance(payload["nonce"], str)
                or len(_decode(payload["nonce"])) != 16
            ):
                return None
            timestamp = _utc_timestamp(now)
            if issued_at > timestamp or expires_at <= timestamp or expires_at <= issued_at:
                return None
            return username
        except (
            UnicodeError,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
            binascii.Error,
        ):
            return None


class PreviewAuthenticator:
    def __init__(
        self,
        settings: PreviewSettings,
        *,
        password_hasher: PasswordHasher | None = None,
    ) -> None:
        self._password_hasher = password_hasher or PasswordHasher()
        self._users = {
            user.username.casefold(): user.password_hash for user in settings.users
        }
        self._dummy_hash = self._password_hasher.hash(
            "icor-preview-unknown-user-verification"
        )

    def verify(self, username: str, password: str) -> bool:
        identifier = username.strip().casefold()
        known_hash = self._users.get(identifier)
        selected_hash = known_hash or self._dummy_hash
        try:
            verified = self._password_hasher.verify(selected_hash, password)
        except VerificationError:
            verified = False
        return known_hash is not None and verified


class LoginThrottle:
    def __init__(
        self,
        digest_key: bytes,
        *,
        max_attempts: int = 5,
        window_seconds: int = 900,
        max_buckets: int = 10_000,
    ) -> None:
        if len(digest_key) < 32 or min(max_attempts, window_seconds, max_buckets) < 1:
            raise ValueError("login throttle configuration is invalid")
        self._digest_key = digest_key
        self._max_attempts = max_attempts
        self._window_seconds = window_seconds
        self._max_buckets = max_buckets
        self._failures: dict[str, list[float]] = {}

    @property
    def bucket_count(self) -> int:
        return len(self._failures)

    def key(self, username: str, client_address: str) -> str:
        value = f"{username.strip().casefold()}\0{client_address}".encode()
        return hmac.digest(self._digest_key, value, hashlib.sha256).hex()

    def allow(self, key: str, now: float) -> bool:
        self._evict(now)
        return len(self._failures.get(key, ())) < self._max_attempts

    def record_failure(self, key: str, now: float) -> None:
        self._evict(now)
        if key not in self._failures and len(self._failures) >= self._max_buckets:
            oldest = min(
                self._failures,
                key=lambda candidate: max(self._failures[candidate], default=float("-inf")),
            )
            self._failures.pop(oldest, None)
        self._failures.setdefault(key, []).append(now)

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)

    def _evict(self, now: float) -> None:
        threshold = now - self._window_seconds
        for key in tuple(self._failures):
            current = [value for value in self._failures[key] if value > threshold]
            if current:
                self._failures[key] = current
            else:
                self._failures.pop(key, None)


def _utc_timestamp(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("session time must be timezone-aware")
    return int(value.astimezone(UTC).timestamp())


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    if not value or "=" in value or len(value) % 4 == 1:
        raise ValueError("invalid base64url value")
    return base64.b64decode(
        (value + "=" * (-len(value) % 4)).encode("ascii"),
        altchars=b"-_",
        validate=True,
    )

from __future__ import annotations

import os
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.testclient import TestClient

from icor.preview.config import ConfigurationError, PreviewSettings, PreviewUser

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])


@pytest.fixture
def settings() -> PreviewSettings:
    hasher = PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1)
    return PreviewSettings(
        users=(PreviewUser("Lucas", hasher.hash("correct horse battery staple")),),
        session_secret=b"s" * 32,
        session_ttl_seconds=3600,
    )


@pytest.fixture
def assets(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<main>ICOR application</main>", encoding="utf-8")
    (root / "assets" / "app-123.js").write_text("export {};", encoding="utf-8")
    (root / "assets" / "app-123.css").write_text("body{}", encoding="utf-8")
    return root


def _stub_api() -> FastAPI:
    app = FastAPI()
    app.state.snapshot_manifest = object()

    @app.get("/api/example")
    def example() -> dict[str, bool]:
        return {"ready": True}

    return app


def _preview(monkeypatch: pytest.MonkeyPatch, settings: PreviewSettings, assets: Path):
    from icor.preview import app as preview_module

    monkeypatch.setattr(preview_module, "create_app", lambda **kwargs: _stub_api())
    return preview_module.create_preview_app(settings, asset_root=assets)


def test_factory_fails_closed_without_configuration(
    monkeypatch: pytest.MonkeyPatch, assets: Path
) -> None:
    from icor.preview.app import create_preview_app

    with pytest.raises(ConfigurationError, match="configuration"):
        create_preview_app(asset_root=assets, snapshot_root=assets / "missing")


def test_factory_fails_closed_without_active_snapshot(
    settings: PreviewSettings, assets: Path
) -> None:
    from icor.preview.app import create_preview_app

    with pytest.raises(ConfigurationError, match="snapshot"):
        create_preview_app(settings, asset_root=assets, snapshot_root=assets / "missing")


def test_factory_fails_closed_without_frontend(
    monkeypatch: pytest.MonkeyPatch, settings: PreviewSettings, tmp_path: Path
) -> None:
    from icor.preview import app as preview_module

    monkeypatch.setattr(preview_module, "create_app", lambda **kwargs: _stub_api())
    with pytest.raises(ConfigurationError, match="frontend"):
        preview_module.create_preview_app(settings, asset_root=tmp_path / "missing")


def test_factory_uses_configured_active_snapshot_root(
    monkeypatch: pytest.MonkeyPatch,
    settings: PreviewSettings,
    assets: Path,
    tmp_path: Path,
) -> None:
    from icor.preview import app as preview_module

    selected_root = tmp_path / "evidence"
    captured: dict[str, object] = {}

    def create_core(**kwargs: object) -> FastAPI:
        captured.update(kwargs)
        return _stub_api()

    monkeypatch.setenv("ICOR_EVIDENCE_ACTIVE_ROOT", str(selected_root))
    monkeypatch.setattr(preview_module, "create_app", create_core)

    preview_module.create_preview_app(settings, asset_root=assets)

    assert captured["snapshot_root"] == selected_root

    explicit_root = tmp_path / "explicit-evidence"
    captured.clear()
    preview_module.create_preview_app(
        settings, asset_root=assets, snapshot_root=explicit_root
    )
    assert captured["snapshot_root"] == explicit_root


def test_login_logout_and_protected_navigation(
    monkeypatch: pytest.MonkeyPatch,
    settings: PreviewSettings,
    assets: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _preview(monkeypatch, settings, assets)
    with TestClient(app, base_url="https://preview.example") as client:
        assert client.get("/auth/login").status_code == 200
        denied = client.post(
            "/auth/login", data={"username": "Lucas", "password": "wrong-secret"}
        )
        assert denied.status_code == 401
        assert "wrong-secret" not in denied.text

        signed_in = client.post(
            "/auth/login",
            data={"username": "Lucas", "password": "correct horse battery staple"},
            follow_redirects=False,
        )
        cookie = signed_in.headers["set-cookie"]
        assert signed_in.status_code == 303
        assert "Secure" in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=strict" in cookie
        assert "Path=/" in cookie
        assert client.get("/planner").text == "<main>ICOR application</main>"
        assert client.get("/api/example").json() == {"ready": True}

        signed_out = client.post("/auth/logout", follow_redirects=False)
        assert signed_out.status_code == 303
        assert "Max-Age=0" in signed_out.headers["set-cookie"]
        assert client.get("/planner").status_code == 401

    assert "wrong-secret" not in caplog.text
    assert settings.users[0].password_hash not in caplog.text


def test_login_rejects_oversized_and_throttled_submissions(
    monkeypatch: pytest.MonkeyPatch, settings: PreviewSettings, assets: Path
) -> None:
    app = _preview(monkeypatch, settings, assets)
    with TestClient(app, base_url="https://preview.example") as client:
        oversized = client.post(
            "/auth/login",
            content=b"username=x&password=" + b"x" * 9000,
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        assert oversized.status_code == 413
        for _ in range(5):
            assert client.post(
                "/auth/login", data={"username": "Lucas", "password": "wrong"}
            ).status_code == 401
        limited = client.post(
            "/auth/login", data={"username": "Lucas", "password": "wrong"}
        )
        assert limited.status_code == 429
        assert limited.json() == {"detail": "Login temporarily unavailable"}


def test_static_assets_spa_and_api_404_are_distinct(
    monkeypatch: pytest.MonkeyPatch, settings: PreviewSettings, assets: Path
) -> None:
    app = _preview(monkeypatch, settings, assets)
    with TestClient(app, base_url="https://preview.example") as client:
        client.post(
            "/auth/login",
            data={"username": "Lucas", "password": "correct horse battery staple"},
        )
        script = client.get("/assets/app-123.js")
        stylesheet = client.get("/assets/app-123.css")
        assert script.status_code == 200
        assert script.headers["content-type"].startswith("text/javascript")
        assert stylesheet.headers["content-type"].startswith("text/css")
        assert client.get("/opportunities/market").text == "<main>ICOR application</main>"
        missing_api = client.get("/api/not-a-route")
        assert missing_api.status_code == 404
        assert missing_api.headers["content-type"].startswith("application/json")
        assert "ICOR application" not in missing_api.text


@pytest.mark.parametrize(
    "request_path",
    (
        "../outside.txt",
        "assets/../outside.txt",
        r"assets\outside.txt",
        "assets/%2foutside.txt",
        "assets/%5coutside.txt",
        "assets/%00outside.txt",
        "assets/evil\x00.js",
    ),
)
def test_static_resolver_rejects_malformed_paths(
    assets: Path, request_path: str
) -> None:
    from icor.preview.static import resolve_asset

    assert resolve_asset(assets, request_path) is None


def test_static_resolver_rejects_symlink_escape(assets: Path, tmp_path: Path) -> None:
    from icor.preview.static import resolve_asset

    outside = tmp_path / "outside.js"
    outside.write_text("secret", encoding="utf-8")
    link = assets / "assets" / "escaped.js"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    assert resolve_asset(assets, "assets/escaped.js") is None


def test_preview_variables_are_isolated() -> None:
    assert "ICOR_PREVIEW_USERS" not in os.environ
    assert "ICOR_PREVIEW_SESSION_SECRET" not in os.environ

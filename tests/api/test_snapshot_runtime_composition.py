from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient

from icor.api import app as app_module
from icor.api.app import create_app

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])


def test_runtime_has_no_demo_planner_fallback() -> None:
    source = inspect.getsource(app_module)

    assert "DemoPlannerRepository" not in source
    assert "demo-planner-v1" not in source


def test_missing_active_snapshot_fails_closed() -> None:
    with TestClient(
        create_app(snapshot_root=app_module.ROOT / ".local" / "missing-snapshot")
    ) as client:
        response = client.get("/api/v1/planner/options")

    assert response.status_code == 503
    assert response.json()["code"] == "planning_snapshot_unavailable"

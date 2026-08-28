from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from icor.api.app import create_app

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])
TOKEN = "local-review-export-token-2026-08-28"


class Export:
    snapshot_id = "snapshot-real-v1"

    def render_csv(self, cutoff):
        return f"snapshot_id,cutoff\r\nsnapshot-real-v1,{cutoff.isoformat()}\r\n"


def test_ml_export_requires_local_capability_token() -> None:
    with TestClient(
        create_app(ml_export_service=Export(), export_token=TOKEN)
    ) as client:
        denied = client.get("/api/exports/ml.csv?cutoff=2024-12-31")
        allowed = client.get(
            "/api/exports/ml.csv?cutoff=2024-12-31",
            headers={"X-ICOR-Export-Token": TOKEN},
        )

    assert denied.status_code == 403
    assert allowed.status_code == 200
    assert allowed.headers["cache-control"] == "no-store"
    assert "snapshot-real-v1,2024-12-31" in allowed.text


def test_short_export_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        create_app(ml_export_service=Export(), export_token="short")

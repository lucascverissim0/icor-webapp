"""Eurocodes over HTTP: rows carry them, the status says what is served, startup fails closed.

The map is the fictional demo map (`Q` codes, invented brands); nothing here is
ICOR data.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from icor.api import app as app_module
from icor.api.app import _configured_eurocodes, create_app
from icor.eurocodes.demo import demo_eurocodes, write_demo_eurocodes
from icor.infrastructure.demo_planner_repository import DemoPlannerRepository
from icor.infrastructure.sqlite_coverage_repository import SQLiteCoverageRepository

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])
FIXTURE = Path(__file__).parents[2] / "data" / "demo" / "planner-v1.json"


def _client(tmp_path: Path, **kwargs) -> TestClient:  # type: ignore[no-untyped-def]
    return TestClient(
        create_app(
            repository=DemoPlannerRepository.from_path(FIXTURE),
            coverage_repository=SQLiteCoverageRepository(tmp_path / "coverage.sqlite3"),
            **kwargs,
        )
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with _client(tmp_path, eurocodes=demo_eurocodes()) as test_client:
        yield test_client


def test_model_year_rows_show_certain_and_shared_units(client: TestClient) -> None:
    body = client.get(
        "/api/v1/opportunities", params={"group_by": "model_year", "page_size": 100}
    ).json()

    assert body["eurocodes_loaded"] is True
    rows = {(row["model"], row["model_year"]): row["eurocodes"] for row in body["items"]}
    launch = rows[("A1 Horizon", 2025)]
    assert [
        (code["code"], code["certain_units"], code["shared_with"]) for code in launch["codes"]
    ] == [("Q201", 0, ["Q202"]), ("Q202", 0, ["Q201"])]
    assert launch["codes"][0]["label"] == "Aurora Mobility A1 Horizon I"
    assert rows[("V3 Current", 2023)]["outside_units"] > 0
    assert rows[("N7 Atlas", 2027)]["codes"] == []
    for row in body["items"]:
        breakdown = row["eurocodes"]
        assert (
            breakdown["single_units"]
            + breakdown["shared_units"]
            + breakdown["outside_units"]
            + breakdown["unassignable_units"]
            + breakdown["unreviewed_units"]
        ) == row["demand"]["base_units"]


def test_brand_rows_list_no_eurocodes(client: TestClient) -> None:
    body = client.get("/api/v1/opportunities", params={"group_by": "brand"}).json()

    assert body["eurocodes_loaded"] is True
    assert all(row["eurocodes"] is None for row in body["items"])


def test_the_detail_carries_the_same_eurocodes_as_the_listing(client: TestClient) -> None:
    listed = client.get("/api/v1/opportunities", params={"group_by": "model"}).json()["items"][0]

    detail = client.get(
        f"/api/v1/opportunities/{listed['group_id']}", params={"group_by": "model"}
    ).json()

    assert detail["eurocodes"] == listed["eurocodes"]


def test_status_says_what_is_served_without_a_snapshot(client: TestClient) -> None:
    status = client.get("/api/v1/eurocodes/status").json()

    assert status["loaded"] is True
    assert status["catalogue_sha256"] == "0" * 64
    assert status["reviewed_identities"] == 3
    assert status["coverage"] == []
    assert status["warnings"] == ["No snapshot is open, so coverage is not measured."]


def test_without_eurocodes_rows_and_status_say_so(tmp_path: Path) -> None:
    with _client(tmp_path) as test_client:
        body = test_client.get("/api/v1/opportunities", params={"group_by": "model"}).json()
        status = test_client.get("/api/v1/eurocodes/status").json()

    assert body["eurocodes_loaded"] is False
    assert all(row["eurocodes"] is None for row in body["items"])
    assert status["loaded"] is False
    assert status["coverage"] == []


def test_an_explicit_eurocode_folder_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ICOR_EUROCODE_DIR", str(write_demo_eurocodes(tmp_path / "eurocodes")))

    with _client(tmp_path) as test_client:
        status = test_client.get("/api/v1/eurocodes/status").json()

    assert status["loaded"] is True


def test_an_explicit_eurocode_folder_that_cannot_load_stops_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ICOR_EUROCODE_DIR", str(tmp_path / "missing"))

    with pytest.raises(OSError):
        create_app(
            repository=DemoPlannerRepository.from_path(FIXTURE),
            coverage_repository=SQLiteCoverageRepository(tmp_path / "coverage.sqlite3"),
        )


def test_only_a_local_service_on_the_active_snapshot_falls_back_to_the_private_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ICOR_EUROCODE_DIR", raising=False)
    monkeypatch.setattr(
        app_module, "DEFAULT_EUROCODE_DIR", write_demo_eurocodes(tmp_path / "private")
    )

    assert _configured_eurocodes(local_active_snapshot=False) == (None, None)
    loaded, warning = _configured_eurocodes(local_active_snapshot=True)
    assert loaded is not None and warning is None


def test_a_broken_private_folder_is_reported_not_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = write_demo_eurocodes(tmp_path / "private")
    (folder / "decisions.json").write_text('{"metadata": {"catalogue_sha256": "x"}}')
    monkeypatch.delenv("ICOR_EUROCODE_DIR", raising=False)
    monkeypatch.setattr(app_module, "DEFAULT_EUROCODE_DIR", folder)

    loaded, warning = _configured_eurocodes(local_active_snapshot=True)

    assert loaded is None
    assert warning == "The private Eurocode files could not be loaded (ValueError)."

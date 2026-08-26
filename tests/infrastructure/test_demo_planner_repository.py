import json
from pathlib import Path

import pytest

from icor.domain.planner import EvidenceStatus
from icor.infrastructure.demo_planner_repository import (
    DemoPlannerRepository,
    FixtureError,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "demo" / "planner-v1.json"


@pytest.fixture
def repository() -> DemoPlannerRepository:
    return DemoPlannerRepository.from_path(FIXTURE)


def test_fixture_is_deterministic_and_demonstration_only(
    repository: DemoPlannerRepository,
) -> None:
    rows = repository.list_all()

    assert len(rows) >= 8
    assert len({row.configuration_id for row in rows}) == len(rows)
    assert all(row.configuration_id.startswith("demo-") for row in rows)
    assert {row.evidence_status for row in rows} == {EvidenceStatus.DEMONSTRATION}
    assert {row.data_version for row in rows} == {"demo-planner-v1"}
    assert {row.market for row in rows} == {"DE", "FR"}
    assert {row.forecast_horizon for row in rows} == {2028, 2030}


def test_fixture_preserves_unknown_fitment(repository: DemoPlannerRepository) -> None:
    rows = repository.list_all()

    assert any(row.equipment.hud is None for row in rows)
    assert any(row.drive_side is None for row in rows)


def test_repository_returns_immutable_records_and_identity_lookup(
    repository: DemoPlannerRepository,
) -> None:
    rows = repository.list_all()

    assert isinstance(rows, tuple)
    assert repository.get(rows[0].configuration_id) is rows[0]
    assert repository.get("missing") is None
    assert repository.data_version == "demo-planner-v1"
    assert repository.list_model_year_demand() == tuple(
        demand for row in rows for demand in row.model_year_demand
    )


def test_fixture_model_year_demand_reconciles_exactly(
    repository: DemoPlannerRepository,
) -> None:
    for row in repository.list_all():
        assert sum(item.demand.downside_units for item in row.model_year_demand) == (
            row.demand.downside_units
        )
        assert sum(item.demand.base_units for item in row.model_year_demand) == (
            row.demand.base_units
        )
        assert sum(item.demand.upside_units for item in row.model_year_demand) == (
            row.demand.upside_units
        )


def test_fixture_rejects_non_reconciling_model_year_demand(tmp_path: Path) -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["configurations"][0]["model_year_demand"][0]["base_units"] += 1
    fixture = tmp_path / "planner.json"
    fixture.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(FixtureError, match="invalid"):
        DemoPlannerRepository.from_path(fixture)


def test_repository_never_writes_fixture(repository: DemoPlannerRepository) -> None:
    before = FIXTURE.read_bytes()

    repository.list_all()
    repository.get("demo-missing")

    assert FIXTURE.read_bytes() == before


@pytest.mark.parametrize(
    "content",
    [
        "not-json-and-must-not-appear-in-error",
        '{"data_version":"wrong","configurations":[]}',
        (
            '{"data_version":"demo-planner-v1","configurations":['
            '{"configuration_id":"demo-duplicate"},'
            '{"configuration_id":"demo-duplicate"}]}'
        ),
    ],
)
def test_invalid_fixture_uses_safe_error_without_echoing_content(
    tmp_path: Path, content: str
) -> None:
    fixture = tmp_path / "planner.json"
    fixture.write_text(content, encoding="utf-8")

    with pytest.raises(FixtureError) as error:
        DemoPlannerRepository.from_path(fixture)

    assert str(error.value) == "Demonstration planner fixture is invalid"
    assert content not in str(error.value)

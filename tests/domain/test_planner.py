from dataclasses import replace
from datetime import UTC, datetime

import pytest

from icor.domain.planner import (
    Confidence,
    ConfidenceLevel,
    DemandRange,
    Equipment,
    EvidenceStatus,
    ModelYearDemand,
    PlannerQuery,
    PlanningConfiguration,
    SortDirection,
    SortField,
    SourceSummary,
    filter_sort_paginate,
)


def configuration(
    configuration_id: str,
    *,
    market: str = "FR",
    brand: str = "Renault",
    model: str = "Megane Vision",
    horizon: int = 2030,
    downside: int = 80,
    base: int = 100,
    upside: int = 130,
    hud: bool | None = None,
) -> PlanningConfiguration:
    source = SourceSummary(
        name="Synthetic planning fixture",
        description="Non-proprietary demonstration evidence.",
    )
    return PlanningConfiguration(
        configuration_id=configuration_id,
        sku=f"DEMO-{configuration_id.upper()}",
        part_family="Demo acoustic camera family",
        market=market,
        brand=brand,
        model=model,
        model_year_start=2025,
        model_year_end=2028,
        generation="Demo generation A",
        facelift=None,
        body_style="Hatchback",
        drive_side="left",
        equipment=Equipment(
            camera_adas=True,
            hud=hud,
            heated=False,
            acoustic=True,
            rain_light_sensor=None,
        ),
        forecast_horizon=horizon,
        demand=DemandRange(
            downside_units=downside,
            base_units=base,
            upside_units=upside,
        ),
        vehicle_exposure_units=5_000,
        replacement_rate=0.02,
        identity_confidence=Confidence(
            level=ConfidenceLevel.MEDIUM,
            reason="Synthetic identity distinctions are incomplete.",
        ),
        data_quality_confidence=Confidence(
            level=ConfidenceLevel.LOW,
            reason="Values are demonstration-only.",
        ),
        evidence_status=EvidenceStatus.DEMONSTRATION,
        sources=(source,),
        updated_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        data_version="demo-planner-v1",
        model_year_demand=(
            ModelYearDemand(
                configuration_id=configuration_id,
                model_year=2025,
                forecast_horizon=horizon,
                demand=DemandRange(
                    downside_units=downside,
                    base_units=base,
                    upside_units=upside,
                ),
                evidence_status=EvidenceStatus.DEMONSTRATION,
                data_version="demo-planner-v1",
                sources=(source,),
            ),
        ),
    )


def test_unknown_equipment_remains_unknown() -> None:
    assert configuration("demo-a").equipment.hud is None


@pytest.mark.parametrize(
    ("downside", "base", "upside"),
    [(-1, 0, 1), (20, 10, 30), (10, 40, 30), (1.5, 2, 3)],
)
def test_demand_range_rejects_invalid_units(
    downside: int, base: int, upside: int
) -> None:
    with pytest.raises(ValueError, match="integer units with downside <= base <= upside"):
        DemandRange(
            downside_units=downside,
            base_units=base,
            upside_units=upside,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"page": 0},
        {"page_size": 0},
        {"page_size": 101},
    ],
)
def test_planner_query_rejects_invalid_pagination(overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="pagination"):
        PlannerQuery(**overrides)


def test_filter_sort_paginate_is_deterministic_and_summarizes_all_matches() -> None:
    rows = (
        configuration("demo-low", base=100, downside=80, upside=130),
        configuration("demo-high-b", base=300, downside=250, upside=380),
        configuration("demo-high-a", base=300, downside=240, upside=370),
        configuration("demo-de", market="DE", base=900, downside=700, upside=1_100),
    )

    result = filter_sort_paginate(
        rows,
        PlannerQuery(markets=("FR",), page=1, page_size=2),
    )

    assert result.total == 3
    assert [row.configuration_id for row in result.items] == ["demo-high-a", "demo-high-b"]
    assert result.summary.candidate_count == 3
    assert result.summary.downside_units == 570
    assert result.summary.base_units == 700
    assert result.summary.upside_units == 880
    assert result.page == 1
    assert result.page_size == 2
    assert result.pages == 2


def test_filter_sort_paginate_combines_canonical_filters() -> None:
    rows = (
        configuration("demo-match", market="FR", brand="Renault", model="Megane Vision"),
        configuration("demo-market", market="DE", brand="Renault", model="Megane Vision"),
        configuration("demo-brand", market="FR", brand="Peugeot", model="Megane Vision"),
        configuration("demo-model", market="FR", brand="Renault", model="Scenic Vision"),
        configuration(
            "demo-horizon",
            market="FR",
            brand="Renault",
            model="Megane Vision",
            horizon=2028,
        ),
    )

    result = filter_sort_paginate(
        rows,
        PlannerQuery(
            markets=("FR",),
            horizons=(2030,),
            brands=("Renault",),
            models=("Megane Vision",),
            evidence=(EvidenceStatus.DEMONSTRATION,),
        ),
    )

    assert [row.configuration_id for row in result.items] == ["demo-match"]


def test_explicit_sort_direction_changes_results_without_mutating_input() -> None:
    rows = (
        configuration("demo-b", brand="Zephyr", base=200, upside=230),
        configuration("demo-a", brand="Aurora", base=100),
    )

    result = filter_sort_paginate(
        rows,
        PlannerQuery(
            sort=SortField.BRAND,
            direction=SortDirection.ASC,
        ),
    )

    assert [row.brand for row in result.items] == ["Aurora", "Zephyr"]
    assert [row.configuration_id for row in rows] == ["demo-b", "demo-a"]


def test_model_year_range_must_ascend() -> None:
    with pytest.raises(ValueError, match="model year range"):
        replace(configuration("demo-invalid"), model_year_start=2030, model_year_end=2029)


def test_model_year_demand_must_reconcile_with_configuration() -> None:
    row = configuration("demo-reconcile")
    invalid_demand = replace(
        row.model_year_demand[0],
        demand=DemandRange(downside_units=80, base_units=99, upside_units=130),
    )

    with pytest.raises(ValueError, match="model-year demand must reconcile"):
        replace(row, model_year_demand=(invalid_demand,))


def test_model_year_demand_requires_canonical_identity_and_range() -> None:
    row = configuration("demo-canonical")

    with pytest.raises(ValueError, match="identity and applicability"):
        replace(
            row,
            model_year_demand=(
                replace(row.model_year_demand[0], configuration_id="other", model_year=2035),
            ),
        )

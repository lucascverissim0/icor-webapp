from datetime import UTC, datetime
from pathlib import Path

import pytest

from icor.application.opportunities import (
    OpportunityGroupBy,
    OpportunityQuery,
    OpportunityService,
)
from icor.application.ranking import DemandReadinessV1
from icor.domain.opportunities import CoverageMatchType, CoverageStatus, ProductionCoverage
from icor.infrastructure.demo_planner_repository import DemoPlannerRepository

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "demo" / "planner-v1.json"


class StaticCoverageRepository:
    def __init__(self, rows: tuple[ProductionCoverage, ...] = ()) -> None:
        self.rows = rows

    def list_all(self) -> tuple[ProductionCoverage, ...]:
        return self.rows


def coverage(
    coverage_id: str,
    *,
    match_type: CoverageMatchType,
    configuration_id: str | None,
    brand: str,
    model: str,
    model_year: int,
    sku: str | None,
) -> ProductionCoverage:
    timestamp = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
    return ProductionCoverage(
        coverage_id=coverage_id,
        match_type=match_type,
        configuration_id=configuration_id,
        brand=brand,
        model=model,
        model_year=model_year,
        sku=sku,
        note=None,
        created_at=timestamp,
        updated_at=timestamp,
    )


def service(rows: tuple[ProductionCoverage, ...] = ()) -> OpportunityService:
    return OpportunityService(
        DemoPlannerRepository.from_path(FIXTURE),
        StaticCoverageRepository(rows),
        DemandReadinessV1(),
    )


@pytest.mark.parametrize("group_by", list(OpportunityGroupBy))
def test_every_grouping_reconciles_to_filtered_configuration_demand(
    group_by: OpportunityGroupBy,
) -> None:
    result = service().list(OpportunityQuery(group_by=group_by))

    assert sum(row.demand.base_units for row in result.items) == 6_560
    assert result.summary.base_units == 6_560
    assert result.summary.exact_covered_base_units == 0
    assert all(row.coverage_status is CoverageStatus.UNCOVERED for row in result.items)


def test_exact_match_precedes_fallback_without_double_counting() -> None:
    exact = coverage(
        "exact",
        match_type=CoverageMatchType.EXACT_CONFIGURATION,
        configuration_id="demo-aurora-a1-camera-fr-2030",
        brand="Aurora Mobility",
        model="A1 Horizon",
        model_year=2025,
        sku="DEMO-AUR-A1-CAM",
    )
    fallback = coverage(
        "fallback",
        match_type=CoverageMatchType.VEHICLE_YEAR_FALLBACK,
        configuration_id=None,
        brand="Aurora Mobility",
        model="A1 Horizon",
        model_year=2025,
        sku=None,
    )

    result = service((exact, fallback)).list(
        OpportunityQuery(group_by=OpportunityGroupBy.MODEL_YEAR)
    )
    row = next(
        item
        for item in result.items
        if item.brand == "Aurora Mobility" and item.model_year == 2025
    )

    assert row.demand.base_units == 460
    assert row.exact_covered_base_units == 250
    assert row.fallback_covered_base_units == 210
    assert row.uncovered_base_units == 0
    assert row.coverage_status is CoverageStatus.MIXED


def test_filters_apply_before_percentile_ranking_and_summary() -> None:
    result = service().list(
        OpportunityQuery(
            group_by=OpportunityGroupBy.BRAND,
            markets=("FR",),
            horizons=(2030,),
        )
    )

    assert result.summary.base_units == 2_720
    assert {row.brand for row in result.items} == {
        "Aurora Mobility",
        "Northstar Automotive",
        "Meridian Motors",
    }
    assert result.items[0].score.total_points >= result.items[-1].score.total_points


def test_high_demand_quartile_summary_uses_tie_aware_percentiles() -> None:
    result = service().list(OpportunityQuery(group_by=OpportunityGroupBy.BRAND))

    high_demand_rows = [row for row in result.items if row.score.demand_percentile >= 0.75]
    assert result.summary.high_demand_uncovered_base_units == sum(
        row.uncovered_base_units for row in high_demand_rows
    )


def test_drill_down_returns_only_contributing_model_year_rows() -> None:
    ranked = service().list(OpportunityQuery(group_by=OpportunityGroupBy.BRAND))
    aurora = next(row for row in ranked.items if row.brand == "Aurora Mobility")

    rows = service().drill_down(
        aurora.group_id,
        OpportunityQuery(group_by=OpportunityGroupBy.BRAND),
    )

    assert rows
    assert {row.configuration.brand for row in rows} == {"Aurora Mobility"}
    assert sum(row.model_year_demand.demand.base_units for row in rows) == 2_150


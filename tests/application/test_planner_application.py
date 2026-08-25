from dataclasses import replace
from datetime import UTC, datetime

from icor.application.planner import PlannerService
from icor.domain.planner import (
    Confidence,
    ConfidenceLevel,
    DemandRange,
    Equipment,
    EvidenceStatus,
    PlannerQuery,
    PlanningConfiguration,
    SourceSummary,
)


def configuration(
    configuration_id: str,
    *,
    market: str,
    brand: str,
    model: str,
    horizon: int,
    base_units: int,
) -> PlanningConfiguration:
    return PlanningConfiguration(
        configuration_id=configuration_id,
        sku=None,
        part_family="Synthetic family",
        market=market,
        brand=brand,
        model=model,
        model_year_start=2024,
        model_year_end=2027,
        generation="Synthetic generation",
        facelift=None,
        body_style="Hatchback",
        drive_side=None,
        equipment=Equipment(None, None, None, None, None),
        forecast_horizon=horizon,
        demand=DemandRange(base_units - 10, base_units, base_units + 20),
        vehicle_exposure_units=5_000,
        replacement_rate=0.02,
        identity_confidence=Confidence(ConfidenceLevel.MEDIUM, "Synthetic identity."),
        data_quality_confidence=Confidence(ConfidenceLevel.LOW, "Demonstration values."),
        evidence_status=EvidenceStatus.DEMONSTRATION,
        sources=(SourceSummary("Synthetic source", "Demonstration evidence."),),
        updated_at=datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
        data_version="demo-planner-v1",
    )


class FakeRepository:
    def __init__(self, records: tuple[PlanningConfiguration, ...]) -> None:
        self.records = records

    def list_all(self) -> tuple[PlanningConfiguration, ...]:
        return self.records

    def get(self, configuration_id: str) -> PlanningConfiguration | None:
        return next(
            (row for row in self.records if row.configuration_id == configuration_id),
            None,
        )


def records() -> tuple[PlanningConfiguration, ...]:
    first = configuration(
        "demo-fr-a",
        market="FR",
        brand="Renault",
        model="Megane Vision",
        horizon=2030,
        base_units=100,
    )
    return (
        first,
        replace(
            first,
            configuration_id="demo-de-a",
            market="DE",
            brand="Aurora",
            model="A1",
            forecast_horizon=2028,
            demand=DemandRange(180, 200, 240),
        ),
        replace(
            first,
            configuration_id="demo-fr-b",
            model="Scenic Vision",
            demand=DemandRange(130, 150, 190),
        ),
    )


def test_options_are_unique_sorted_and_include_scenario_metadata() -> None:
    options = PlannerService(FakeRepository(records())).options()

    assert options.markets == ("DE", "FR")
    assert options.horizons == (2028, 2030)
    assert options.brands == ("Aurora", "Renault")
    assert options.models == ("A1", "Megane Vision", "Scenic Vision")
    assert options.evidence_statuses == (EvidenceStatus.DEMONSTRATION,)
    assert options.scenario.name == "Windshield demand planning demonstration"
    assert options.scenario.evidence_status is EvidenceStatus.DEMONSTRATION
    assert options.scenario.data_version == "demo-planner-v1"
    assert options.scenario.updated_at == datetime(2026, 8, 25, 12, 0, tzinfo=UTC)


def test_search_applies_domain_rules_without_adapter_knowledge() -> None:
    result = PlannerService(FakeRepository(records())).search(
        PlannerQuery(markets=("FR",), page_size=1)
    )

    assert result.total == 2
    assert result.items[0].configuration_id == "demo-fr-b"
    assert result.summary.base_units == 250


def test_detail_returns_canonical_record_or_none() -> None:
    service = PlannerService(FakeRepository(records()))

    assert service.detail("demo-fr-a") == records()[0]
    assert service.detail("missing") is None

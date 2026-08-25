"""Strict read-only adapter for the synthetic planner fixture."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from icor.domain.planner import (
    Confidence,
    ConfidenceLevel,
    DemandRange,
    Equipment,
    EvidenceStatus,
    PlanningConfiguration,
    SourceSummary,
)

DATA_VERSION = "demo-planner-v1"


class FixtureError(RuntimeError):
    """Raised when demonstration data cannot satisfy the canonical contract."""


class DemoPlannerRepository:
    def __init__(
        self,
        records: tuple[PlanningConfiguration, ...],
        *,
        data_version: str,
    ) -> None:
        self._records = records
        self._by_id = {record.configuration_id: record for record in records}
        self.data_version = data_version

    @classmethod
    def from_path(cls, path: Path) -> DemoPlannerRepository:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or document.get("data_version") != DATA_VERSION:
                raise ValueError("unsupported fixture version")
            raw_records = document["configurations"]
            if not isinstance(raw_records, list) or not raw_records:
                raise ValueError("fixture requires configurations")
            records = tuple(_parse_configuration(value) for value in raw_records)
            identifiers = [record.configuration_id for record in records]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("duplicate configuration identity")
            if any(record.data_version != DATA_VERSION for record in records):
                raise ValueError("mixed configuration data versions")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise FixtureError("Demonstration planner fixture is invalid") from error
        return cls(records, data_version=DATA_VERSION)

    def list_all(self) -> tuple[PlanningConfiguration, ...]:
        return self._records

    def get(self, configuration_id: str) -> PlanningConfiguration | None:
        return self._by_id.get(configuration_id)


def _parse_confidence(value: Any) -> Confidence:
    if not isinstance(value, dict):
        raise TypeError("confidence must be an object")
    return Confidence(
        level=ConfidenceLevel(value["level"]),
        reason=value["reason"],
    )


def _parse_configuration(value: Any) -> PlanningConfiguration:
    if not isinstance(value, dict):
        raise TypeError("configuration must be an object")
    equipment = value["equipment"]
    demand = value["demand"]
    if not isinstance(equipment, dict) or not isinstance(demand, dict):
        raise TypeError("equipment and demand must be objects")
    return PlanningConfiguration(
        configuration_id=value["configuration_id"],
        sku=value["sku"],
        part_family=value["part_family"],
        market=value["market"],
        brand=value["brand"],
        model=value["model"],
        model_year_start=value["model_year_start"],
        model_year_end=value["model_year_end"],
        generation=value["generation"],
        facelift=value["facelift"],
        body_style=value["body_style"],
        drive_side=value["drive_side"],
        equipment=Equipment(
            camera_adas=equipment["camera_adas"],
            hud=equipment["hud"],
            heated=equipment["heated"],
            acoustic=equipment["acoustic"],
            rain_light_sensor=equipment["rain_light_sensor"],
        ),
        forecast_horizon=value["forecast_horizon"],
        demand=DemandRange(
            downside_units=demand["downside_units"],
            base_units=demand["base_units"],
            upside_units=demand["upside_units"],
        ),
        vehicle_exposure_units=value["vehicle_exposure_units"],
        replacement_rate=value["replacement_rate"],
        identity_confidence=_parse_confidence(value["identity_confidence"]),
        data_quality_confidence=_parse_confidence(value["data_quality_confidence"]),
        evidence_status=EvidenceStatus(value["evidence_status"]),
        sources=tuple(
            SourceSummary(name=source["name"], description=source["description"])
            for source in value["sources"]
        ),
        updated_at=datetime.fromisoformat(value["updated_at"]),
        data_version=value["data_version"],
    )

"""Explicit browser-test composition; never used by production or preview runners."""

from __future__ import annotations

import os
from pathlib import Path

from icor.api.app import create_app
from icor.application.completeness import CompletenessQueryService
from icor.application.evidence_review import EvidenceReviewService
from icor.application.registrations import RegistrationService
from icor.domain.planner import DemandRange
from icor.infrastructure.demo_planner_repository import DemoPlannerRepository
from icor.infrastructure.snapshot_vehicle_forecast_repository import (
    GenerationOption,
    MarketVehicleForecast,
    VehicleForecastOptions,
    VehicleForecastResult,
    VehicleOption,
)
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

ROOT = Path(__file__).resolve().parents[1]


class E2EVehicleForecasts:
    """Small deterministic contract fixture for the browser-only application."""

    def options(self, *, brand=None, model=None, **_kwargs):  # type: ignore[no-untyped-def]
        vehicles = (VehicleOption("Volkswagen", "Golf"),)
        if brand == "Volkswagen" and model == "Golf":
            return VehicleForecastOptions(
                vehicles,
                (2020, 2021, 2022, 2023, 2024, 2025),
                (
                    GenerationOption(
                        "volkswagen-golf-mk8-europe",
                        "Golf Mk8",
                        2019,
                        None,
                        "manufacturer_generation_window",
                        "high",
                    ),
                ),
                (2028, 2031),
            )
        return VehicleForecastOptions(vehicles, (), (), (2028, 2031), ("Volkswagen",))

    def forecast(self, **kwargs):  # type: ignore[no-untyped-def]
        names = (
            ("EU27", "Europe (EU27)"),
            ("BE", "Belgium"),
            ("FR", "France"),
            ("ES", "Spain"),
            ("NL", "The Netherlands"),
            ("GB", "United Kingdom (GB; England is not separable)"),
            ("DE", "Germany"),
            ("PL", "Poland"),
        )
        markets = tuple(
            MarketVehicleForecast(
                code,
                name,
                "available",
                1000,
                6,
                DemandRange(700, 800, 900),
                DemandRange(24, 32, 41),
            )
            for code, name in names
        )
        return VehicleForecastResult(
            "Volkswagen",
            "Golf",
            kwargs.get("year"),
            "volkswagen-golf-mk8-europe",
            "Golf Mk8",
            "manufacturer_generation_window",
            "high",
            "https://www.volkswagen-newsroom.com/en/50-years-of-golf-18048",
            2019,
            None,
            kwargs["horizon"],
            (2020, 2021, 2022, 2023, 2024, 2025),
            (2019,),
            (2026, 2027, 2028),
            markets,
            "constant-annual-retention-v1",
            "age-band-geography-hazard-v1",
            "seeded-triangular-propagation-v1",
            "assumption_led_without_proprietary_fitment_or_hazard_calibration",
            "e2e-fixture",
        )


def create_e2e_app():  # type: ignore[no-untyped-def]
    """Compose sealed evidence and synthetic interaction fixtures only for Playwright."""
    evidence_path = _required_candidate("ICOR_E2E_EVIDENCE_CANDIDATE")
    generation_path = _required_candidate("ICOR_E2E_GENERATION_CANDIDATE")
    evidence_service = EvidenceReviewService.from_candidate(evidence_path)
    generation_service = EvidenceReviewService.from_candidate(generation_path)
    generation_repository = SQLiteEvidenceRepository(generation_service.database_path)
    return create_app(
        repository=DemoPlannerRepository.from_path(ROOT / "data" / "demo" / "planner-v1.json"),
        evidence_service=evidence_service,
        registration_service=RegistrationService.from_candidate(evidence_path),
        completeness_service=CompletenessQueryService(
            generation_repository, generation_service.manifest
        ),
        vehicle_forecast_service=E2EVehicleForecasts(),
    )


def _required_candidate(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError("E2E candidate configuration is required")
    return Path(value)

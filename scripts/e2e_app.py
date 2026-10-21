"""Explicit browser-test composition; never used by production or preview runners."""

from __future__ import annotations

import json
import os
import tempfile
from decimal import Decimal
from pathlib import Path

from icor.api.app import create_app
from icor.api.eurocodes import report_path
from icor.application.completeness import CompletenessQueryService
from icor.application.evidence_review import EvidenceReviewService
from icor.application.registrations import RegistrationService
from icor.domain.planner import DemandRange
from icor.eurocodes.annotation import EurocodeBreakdown, EurocodeCell, eurocode_breakdown
from icor.eurocodes.data_coverage import (
    CountryComparison,
    FleetCell,
    FleetRange,
    OfficialStock,
    build_report,
)
from icor.eurocodes.demo import demo_eurocodes
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


def _fictional_eurocodes() -> EurocodeBreakdown:
    """1,000 registrations allocated through the demo map: single, shared, outside, unreviewed.

    Built by the application's own annotation so the fixture cannot drift from
    what the real repository serves.
    """

    cells = (
        ("aurora mobility|a1 horizon", 2024, 500),
        ("aurora mobility|a1 horizon", 2025, 200),
        ("aurora mobility|a1 horizon", 2019, 100),
        ("northstar automotive|n7 atlas", 2024, 200),
    )
    return eurocode_breakdown(
        demo_eurocodes(),
        tuple(
            EurocodeCell(identity, year, "BE", identity.split("|")[1], units)
            for identity, year, units in cells
        ),
    )


E2E_REPORT_SNAPSHOT_ID = "e2e-fixture"


def _fleet_cell(identity, raw_model, geography, year, evidence, cars, *, sources=()):  # type: ignore[no-untyped-def]
    units = Decimal(cars)
    return FleetCell(
        identity=identity,
        raw_model=raw_model,
        geography=geography,
        cohort_year=year,
        evidence=evidence,
        fleet=FleetRange(units * Decimal("0.9"), units, units * Decimal("1.1")),
        fleet_2028=units * Decimal("0.8"),
        sources=frozenset(sources),
    )


def _write_data_report() -> Path:
    """The Eurocode data report for fictional cars only, in a fresh temporary directory."""

    aurora, meridian = "aurora mobility|a1 horizon", "meridian motors|m5 vector"
    cells = [
        _fleet_cell(aurora, "A1 HORIZON", "DE", 2022, "registrations", 4000, sources={"pub-a"}),
        _fleet_cell(aurora, "A1 HORIZON", "FR", 2022, "registrations", 2500, sources={"pub-a"}),
        _fleet_cell(aurora, "A1 HORIZON", "NL", 2023, "interpolated", 900),
        # 2025 is a launch year both Aurora codes cover: shared, never split.
        _fleet_cell(aurora, "A1 HORIZON", "DE", 2025, "registrations", 600, sources={"pub-a"}),
        _fleet_cell(meridian, "M5 VECTOR", "DE", 2024, "stock", 300, sources={"pub-b"}),
        _fleet_cell(meridian, "M5 VECTOR", "DE", 2025, "interpolated", 1800),
        _fleet_cell(meridian, "M5 VECTOR", "FR", 2026, "forecast", 300),
    ]
    report = build_report(
        demo_eurocodes(),
        cells,
        year=2026,
        snapshot_id=E2E_REPORT_SNAPSHOT_ID,
        countries={
            # A market with age bands, an excess under two years and estimates for the gap.
            "DE": CountryComparison(
                OfficialStock(
                    geography="DE",
                    year=2025,
                    total=11_000,
                    bands={
                        "Y_LT2": 1_000,
                        "Y2-5": 2_500,
                        "Y5-10": 3_000,
                        "Y10-20": 3_500,
                        "Y_GT20": 1_000,
                    },
                    band_year=2025,
                    band_total=11_000,
                ),
                {2025: {2025: Decimal(2400), 2024: Decimal(300), 2022: Decimal(4000)}},
                convention="2y",
                before_sources={2025: {2008: Decimal(1200), 2002: Decimal(400)}},
                unnamed={2025: {2012: Decimal(300)}},
            ),
            # A market whose office publishes a total only.
            "FR": CountryComparison(
                OfficialStock(geography="FR", year=2025, total=9_000, status={"TOTAL": "i"}),
                {2025: {2022: Decimal(2500)}},
                before_sources={2025: {2007: Decimal(1500)}},
            ),
        },
    )
    directory = Path(tempfile.mkdtemp(prefix="icor-e2e-data-report-"))
    report_path(directory, E2E_REPORT_SNAPSHOT_ID).write_text(json.dumps(report), encoding="utf-8")
    return directory


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
                _fictional_eurocodes(),
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
        eurocodes=demo_eurocodes(),
        eurocode_report_dir=_write_data_report(),
        eurocode_report_snapshot_id=E2E_REPORT_SNAPSHOT_ID,
    )


def _required_candidate(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError("E2E candidate configuration is required")
    return Path(value)

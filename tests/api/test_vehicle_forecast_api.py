from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from icor.api.app import create_app
from icor.domain.planner import DemandRange
from icor.infrastructure.snapshot_vehicle_forecast_repository import (
    GenerationOption,
    MarketVehicleForecast,
    VehicleForecastOptions,
    VehicleForecastResult,
    VehicleForecastSelectionError,
    VehicleOption,
)

pytestmark = pytest.mark.allow_hosts(["127.0.0.1", "::1", "localhost"])


class VehicleForecasts:
    def options(self, **_kwargs):  # type: ignore[no-untyped-def]
        return VehicleForecastOptions(
            (VehicleOption("Volkswagen", "Golf"),),
            (2020, 2021),
            (
                GenerationOption(
                    "mk8", "Golf Mk8", 2019, None, "manufacturer_generation_window", "high"
                ),
            ),
            (2028, 2031),
            ("Volkswagen",),
        )

    def forecast(self, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs["year"] == 2019:
            raise VehicleForecastSelectionError("ambiguous transition year")
        return VehicleForecastResult(
            "Volkswagen",
            "Golf",
            kwargs["year"],
            "mk8",
            "Golf Mk8",
            "manufacturer_generation_window",
            "high",
            "https://example.test/golf",
            2019,
            None,
            kwargs["horizon"],
            (2020, 2021),
            (2019,),
            (2026, 2027, 2028),
            (
                MarketVehicleForecast(
                    "EU27",
                    "Europe (EU27)",
                    "available",
                    1000,
                    2,
                    DemandRange(700, 800, 900),
                    DemandRange(24, 32, 41),
                ),
            ),
            "survival-v1",
            "hazard-v1",
            "uncertainty-v1",
            "assumption_led_without_proprietary_fitment_or_hazard_calibration",
            "snapshot-test",
        )


def _client(*, client_release: bool = False) -> TestClient:
    return TestClient(
        create_app(
            vehicle_forecast_service=VehicleForecasts(),
            snapshot_root=Path("C:/local/missing-active-root"),
            client_release=client_release,
        )
    )


def test_vehicle_forecast_options_and_result_are_typed() -> None:
    client = _client()
    options = client.get("/api/v1/vehicle-forecasts/options", params={"search": "golf"})
    forecast = client.get(
        "/api/v1/vehicle-forecasts",
        params={"brand": "Volkswagen", "model": "Golf", "year": 2020, "horizon": 2028},
    )

    assert options.status_code == 200
    assert options.json()["brands"] == ["Volkswagen"]
    assert options.json()["vehicles"] == [{"brand": "Volkswagen", "model": "Golf"}]
    assert forecast.status_code == 200
    assert forecast.json()["generation_name"] == "Golf Mk8"
    assert forecast.json()["excluded_forecast_cohort_years"] == [2026, 2027, 2028]
    assert forecast.json()["markets"][0]["active_fleet"]["base_units"] == 800
    assert forecast.json()["markets"][0]["replacements"]["base_units"] == 32


def test_vehicle_forecast_requires_year_xor_generation_and_reports_safe_selection_error() -> None:
    client = _client()

    missing = client.get(
        "/api/v1/vehicle-forecasts",
        params={"brand": "Volkswagen", "model": "Golf", "horizon": 2028},
    )
    ambiguous = client.get(
        "/api/v1/vehicle-forecasts",
        params={"brand": "Volkswagen", "model": "Golf", "year": 2019, "horizon": 2028},
    )

    assert missing.status_code == 422
    assert missing.json()["code"] == "invalid_vehicle_forecast_selection"
    assert ambiguous.status_code == 422
    assert ambiguous.json()["message"] == "ambiguous transition year"


def test_client_release_rejects_direct_generation_selection() -> None:
    response = _client(client_release=True).get(
        "/api/v1/vehicle-forecasts",
        params={
            "brand": "Volkswagen",
            "model": "Golf",
            "generation": "mk8",
            "horizon": 2028,
        },
    )

    assert response.status_code == 422
    assert response.json()["message"] == (
        "The client catalog supports source model-year selection only."
    )

"""Guided vehicle/generation forecast routes."""

from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from icor.api.schemas import (
    ProblemResponse,
    VehicleForecastOptionsResponse,
    VehicleForecastResponse,
)
from icor.infrastructure.snapshot_vehicle_forecast_repository import (
    VehicleForecastSelectionError,
)

router = APIRouter()


def _problem(request: Request, message: str, status_code: int = 422) -> JSONResponse:
    body = ProblemResponse(
        code="invalid_vehicle_forecast_selection",
        message=message,
        correlation_id=request.state.correlation_id,
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


@router.get(
    "/api/v1/vehicle-forecasts/options",
    response_model=VehicleForecastOptionsResponse,
    responses={503: {"model": ProblemResponse}},
)
def vehicle_forecast_options(
    request: Request,
    search: Annotated[str | None, Query(max_length=100)] = None,
    brand: Annotated[str | None, Query(max_length=100)] = None,
    model: Annotated[str | None, Query(max_length=200)] = None,
) -> VehicleForecastOptionsResponse | JSONResponse:
    service = request.app.state.vehicle_forecast_service
    if service is None:
        return _problem(request, "Vehicle forecast data is unavailable.", 503)
    return VehicleForecastOptionsResponse.model_validate(
        service.options(search=search, brand=brand, model=model)
    )


@router.get(
    "/api/v1/vehicle-forecasts",
    response_model=VehicleForecastResponse,
    responses={422: {"model": ProblemResponse}, 503: {"model": ProblemResponse}},
)
def vehicle_forecast(
    request: Request,
    brand: Annotated[str, Query(min_length=1, max_length=100)],
    model: Annotated[str, Query(min_length=1, max_length=200)],
    horizon: Annotated[int, Query(ge=2020, le=2200)],
    year: Annotated[int | None, Query(ge=1900, le=2200)] = None,
    generation: Annotated[str | None, Query(max_length=200)] = None,
) -> VehicleForecastResponse | JSONResponse:
    service = request.app.state.vehicle_forecast_service
    if service is None:
        return _problem(request, "Vehicle forecast data is unavailable.", 503)
    if (year is None) == (generation is None):
        return _problem(request, "Select exactly one model year or generation.")
    try:
        result = service.forecast(
            brand=brand,
            model=model,
            year=year,
            generation=generation,
            horizon=horizon,
        )
    except VehicleForecastSelectionError as error:
        return _problem(request, str(error))
    return VehicleForecastResponse.model_validate(result)

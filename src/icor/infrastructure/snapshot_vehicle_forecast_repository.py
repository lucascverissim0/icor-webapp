"""Cohort-correct vehicle/generation forecasts over a verified SQLite snapshot."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from icor.domain.evidence import CanonicalVehicle
from icor.domain.planner import DemandRange
from icor.evidence.normalization import (
    normalize_vehicle_label,
    source_vehicle_display_label,
)
from icor.evidence.provenance import resolve_reported_method
from icor.evidence.vehicle_identity import (
    VehicleIdentityIndex,
    parse_vehicle_query,
)
from icor.forecasting.replacement_hazard import ReplacementHazardModel
from icor.forecasting.uncertainty import OpportunityUncertaintyModel
from icor.generations.public_catalog import (
    ReviewedGenerationCatalog,
    VehicleGenerationProfile,
    ranking_public_generation_catalog,
)

_GENERATION_REGISTRY = "public-generation-registry-v2"
_EU27 = frozenset(
    {
        "AT",
        "BE",
        "BG",
        "HR",
        "CY",
        "CZ",
        "DK",
        "EE",
        "FI",
        "FR",
        "DE",
        "GR",
        "HU",
        "IE",
        "IT",
        "LV",
        "LT",
        "LU",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SK",
        "SI",
        "ES",
        "SE",
    }
)
_TARGETS = (
    ("EU27", "Europe (EU27)"),
    ("BE", "Belgium"),
    ("FR", "France"),
    ("ES", "Spain"),
    ("NL", "Netherlands"),
    ("GB", "United Kingdom (GB; England is not separable)"),
    ("DE", "Germany"),
    ("PL", "Poland"),
)


class VehicleForecastSelectionError(ValueError):
    """The requested year/generation cannot be resolved without guessing."""


@dataclass(frozen=True, slots=True)
class VehicleOption:
    brand: str
    model: str


@dataclass(frozen=True, slots=True)
class GenerationOption:
    key: str
    name: str
    start_year: int
    end_year: int | None
    basis: str
    confidence: str


@dataclass(frozen=True, slots=True)
class VehicleForecastOptions:
    vehicles: tuple[VehicleOption, ...]
    years: tuple[int, ...]
    generations: tuple[GenerationOption, ...]
    horizons: tuple[int, ...]
    brands: tuple[str, ...] = ()
    #: The registration year found in a free-text search, so `VW Golf 2020` can
    #: preselect 2020 instead of matching it as text and finding nothing.
    search_year: int | None = None


@dataclass(frozen=True, slots=True)
class MarketVehicleForecast:
    code: str
    name: str
    availability: str
    registration_cohort_units: int | None
    cohort_count: int
    active_fleet: DemandRange | None
    replacements: DemandRange | None


@dataclass(frozen=True, slots=True)
class VehicleForecastResult:
    brand: str
    model: str
    selected_year: int | None
    generation_key: str
    generation_name: str
    generation_basis: str
    generation_confidence: str
    generation_source_url: str | None
    generation_start_year: int
    generation_end_year: int | None
    horizon: int
    included_cohort_years: tuple[int, ...]
    excluded_ambiguous_years: tuple[int, ...]
    excluded_forecast_cohort_years: tuple[int, ...]
    markets: tuple[MarketVehicleForecast, ...]
    survival_method: str
    hazard_method: str
    uncertainty_method: str
    calibration_status: str
    data_version: str


@dataclass(frozen=True, slots=True)
class _Selection:
    key: str
    name: str
    start_year: int
    end_year: int | None
    basis: str
    confidence: str
    source_url: str | None
    profile: VehicleGenerationProfile | None
    cohort_year: int | None = None


class SnapshotVehicleForecastRepository:
    """Search and aggregate every surviving cohort for one selected generation."""

    def __init__(
        self,
        path: Path,
        data_version: str,
        *,
        catalog: ReviewedGenerationCatalog | None = None,
        verified_only: bool = False,
        model_year_only: bool = False,
    ) -> None:
        self._path = path
        self._data_version = data_version
        self._catalog = catalog or ranking_public_generation_catalog()
        self._verified_only = verified_only
        self._model_year_only = model_year_only
        self._hazard = ReplacementHazardModel()
        self._uncertainty = OpportunityUncertaintyModel()
        self._all_vehicle_options: tuple[VehicleOption, ...] | None = None
        self._index: VehicleIdentityIndex | None = None

    def options(
        self,
        *,
        search: str | None = None,
        brand: str | None = None,
        model: str | None = None,
        include_all_brands: bool = False,
    ) -> VehicleForecastOptions:
        parsed = parse_vehicle_query(search) if search else None
        found_year = parsed.year if parsed else None
        with self._connect() as connection:
            brands = self._brands(connection, include_all=include_all_brands)
            vehicles = self._vehicle_options(connection, search or (brand if not model else None))
            if not brand or not model:
                horizons = tuple(
                    row[0]
                    for row in connection.execute(
                        "SELECT DISTINCT horizon_year FROM opportunity_estimate "
                        "ORDER BY horizon_year"
                    )
                )
                return VehicleForecastOptions(vehicles, (), (), horizons, brands, found_year)
            selected_vehicle = CanonicalVehicle("selection", brand, model, None, "Europe")
            profile = self._catalog.profile_for(selected_vehicle)
            if self._verified_only and profile is None:
                return VehicleForecastOptions(vehicles, (), (), (), brands, found_year)
            vehicle_ids = self._vehicle_ids(connection, brand, model, profile)
            if not vehicle_ids:
                return VehicleForecastOptions(vehicles, (), (), (), brands, found_year)
            available_horizons = self._horizons(connection, vehicle_ids)
            if not available_horizons:
                return VehicleForecastOptions(vehicles, (), (), (), brands, found_year)
            cohort_rows = self._cohort_rows(
                connection, vehicle_ids, available_horizons[0]
            )
            latest_observed_year = _latest_evidence_year(cohort_rows)
            years = tuple(
                sorted(
                    {
                        # Every observed cohort year is offered. Years used to be
                        # hidden when they straddled two reviewed generations,
                        # which dropped real registration cohorts from the picker
                        # to protect a generation label the user had not asked
                        # for. A year now selects its own cohort, so the label is
                        # reported rather than used to filter the choices.
                        row["registration_cohort_year"]
                        for row in cohort_rows
                        if row["registration_cohort_year"] <= latest_observed_year
                    }
                )
            )
            if self._model_year_only:
                generations = ()
            elif profile is not None:
                generations = tuple(
                    GenerationOption(
                        window.key,
                        window.display_name,
                        window.start_month.year,
                        window.end_month.year if window.end_month else None,
                        "manufacturer_generation_window",
                        "high",
                    )
                    for window in profile.windows
                )
            else:
                generations = tuple(
                    GenerationOption(
                        row[0],
                        row[1],
                        int(row[2][:4]),
                        int(row[3][:4]) if row[3] else None,
                        "estimated_generation_window",
                        "low",
                    )
                    for row in connection.execute(
                        f"""SELECT DISTINCT g.generation_id, g.display_name,
                            g.start_month, g.end_month
                        FROM generation_entry g
                        WHERE {_in_filter("g.canonical_vehicle_id", vehicle_ids)[0]}
                        ORDER BY g.start_month, g.display_name""",
                        vehicle_ids,
                    )
                )
        return VehicleForecastOptions(
            vehicles, years, generations, available_horizons, brands, found_year
        )

    def forecast(
        self,
        *,
        brand: str,
        model: str,
        year: int | None,
        generation: str | None,
        horizon: int,
    ) -> VehicleForecastResult:
        if (year is None) == (generation is None):
            raise VehicleForecastSelectionError("select exactly one year or generation")
        if self._model_year_only and year is None:
            raise VehicleForecastSelectionError(
                "the client catalog supports source model-year selection only"
            )
        vehicle = CanonicalVehicle("selection", brand, model, None, "Europe")
        profile = self._catalog.profile_for(vehicle)
        if self._verified_only and profile is None:
            raise VehicleForecastSelectionError(
                "the selected vehicle has no reviewed generation"
            )
        selection = self._selection(vehicle, profile, year, generation)
        with self._connect() as connection:
            vehicle_ids = self._vehicle_ids(connection, brand, model, profile)
            if not vehicle_ids:
                raise VehicleForecastSelectionError("the selected vehicle is unavailable")
            rows = self._cohort_rows(connection, vehicle_ids, horizon)
        if not rows:
            raise VehicleForecastSelectionError(
                "no forecast is available for this vehicle and horizon"
            )
        latest_observed_year = _latest_evidence_year(rows)
        forecast_years = {
            row["registration_cohort_year"]
            for row in rows
            if row["registration_cohort_year"] > latest_observed_year
        }
        existing_fleet_rows = [
            row
            for row in rows
            if row["registration_cohort_year"] <= latest_observed_year
        ]
        selected_rows, ambiguous_years = self._select_rows(
            existing_fleet_rows, vehicle, selection
        )
        if not selected_rows:
            raise VehicleForecastSelectionError("no cohorts match the selected generation")
        markets = tuple(
            self._market_result(code, name, selected_rows, selection, horizon)
            for code, name in _TARGETS
        )
        return VehicleForecastResult(
            brand=brand,
            model=model,
            selected_year=year,
            generation_key=selection.key,
            generation_name=selection.name,
            generation_basis=selection.basis,
            generation_confidence=selection.confidence,
            generation_source_url=selection.source_url,
            generation_start_year=selection.start_year,
            generation_end_year=selection.end_year,
            horizon=horizon,
            included_cohort_years=tuple(
                sorted({row["registration_cohort_year"] for row in selected_rows})
            ),
            excluded_ambiguous_years=tuple(sorted(ambiguous_years)),
            excluded_forecast_cohort_years=tuple(sorted(forecast_years)),
            markets=markets,
            survival_method=_survival_method(selected_rows),
            hazard_method=self._hazard.method,
            uncertainty_method=self._uncertainty.method,
            calibration_status="assumption_led_without_proprietary_fitment_or_hazard_calibration",
            data_version=self._data_version,
        )

    def _cohort_year_selection(self, vehicle, profile, year):  # type: ignore[no-untyped-def]
        """A chosen registration year selects that cohort, and only that cohort.

        A year used to resolve to a *generation* and then pull every cohort in
        it, so asking for 2020 returned 2010-2025 for any vehicle whose
        generation window was one estimated block — sixteen model years
        reported as one, with the 2020 label still on screen. The registration
        year is unambiguous evidence in its own right; the generation label is
        the uncertain part, so it is reported rather than used to widen the set.
        """

        entry = (
            self._catalog.entry_for_year(vehicle, year, registry_version=_GENERATION_REGISTRY)
            if profile is not None and not self._model_year_only
            else None
        )
        label = (
            f"{source_vehicle_display_label(vehicle.make)} "
            f"{source_vehicle_display_label(vehicle.model)} — "
            f"{year} registration cohort"
        )
        return _Selection(
            f"source-registration-year:{vehicle.make}:{vehicle.model}:{year}",
            label if entry is None else f"{entry.display_name} — {year} registration cohort",
            year,
            year,
            "official_source_registration_cohort",
            "source-reported" if entry is None else "high",
            entry.evidence_ids[0] if entry is not None and entry.evidence_ids else None,
            profile,
            year,
        )

    def _selection(self, vehicle, profile, year, generation):  # type: ignore[no-untyped-def]
        if year is not None:
            return self._cohort_year_selection(vehicle, profile, year)
        if self._model_year_only:
            raise VehicleForecastSelectionError(
                "the client catalog supports source model-year selection only"
            )
        if profile is not None:
            entries = self._catalog.entries_for(vehicle, registry_version=_GENERATION_REGISTRY)
            if year is not None:
                entry = self._catalog.entry_for_year(
                    vehicle, year, registry_version=_GENERATION_REGISTRY
                )
                if entry is None:
                    raise VehicleForecastSelectionError(
                        "the selected year has no unambiguous reviewed generation"
                    )
            else:
                entry = next(
                    (
                        item
                        for item in entries
                        if item.display_name == generation or item.generation_id == generation
                    ),
                    None,
                )
                if entry is None:
                    window = next(
                        (item for item in profile.windows if item.key == generation), None
                    )
                    entry = next(
                        (
                            item
                            for item in entries
                            if window and item.display_name == window.display_name
                        ),
                        None,
                    )
                if entry is None:
                    raise VehicleForecastSelectionError("the selected generation is unavailable")
            return _Selection(
                next(
                    item.key for item in profile.windows if item.display_name == entry.display_name
                ),
                entry.display_name,
                entry.start_month.year,
                entry.end_month.year if entry.end_month else None,
                "manufacturer_generation_window",
                "high",
                entry.evidence_ids[0],
                profile,
            )
        with self._connect() as connection:
            vehicle_ids = self._vehicle_ids(connection, vehicle.make, vehicle.model, None)
            if not vehicle_ids:
                raise VehicleForecastSelectionError("the selected vehicle is unavailable")
            vehicle_filter, vehicle_parameters = _in_filter("g.canonical_vehicle_id", vehicle_ids)
            if year is not None:
                candidates = connection.execute(
                    """SELECT DISTINCT g.generation_id, g.display_name, g.start_month,
                        g.end_month FROM generation_entry g
                    JOIN cohort_estimate c ON c.generation_id = g.generation_id
                    WHERE """
                    + vehicle_filter
                    + " AND c.registration_cohort_year = ?",
                    (*vehicle_parameters, year),
                ).fetchall()
            else:
                candidates = connection.execute(
                    """SELECT DISTINCT g.generation_id, g.display_name, g.start_month,
                        g.end_month FROM generation_entry g
                    WHERE """
                    + vehicle_filter
                    + " AND (g.generation_id = ? OR g.display_name = ?)",
                    (*vehicle_parameters, generation, generation),
                ).fetchall()
        if len(candidates) != 1:
            raise VehicleForecastSelectionError(
                "the selected year/generation is ambiguous or unavailable"
            )
        row = candidates[0]
        return _Selection(
            row[0],
            row[1],
            int(row[2][:4]),
            int(row[3][:4]) if row[3] else None,
            "estimated_generation_window",
            "low",
            None,
            None,
        )

    def _select_rows(self, rows, vehicle, selection):  # type: ignore[no-untyped-def]
        if selection.cohort_year is not None:
            return (
                [
                    row
                    for row in rows
                    if row["registration_cohort_year"] == selection.cohort_year
                ],
                set(),
            )
        if selection.profile is None:
            return [row for row in rows if row["generation_id"] == selection.key], set()
        selected = []
        ambiguous: set[int] = set()
        for row in rows:
            year = row["registration_cohort_year"]
            entry = self._catalog.entry_for_year(
                CanonicalVehicle("cohort", row["make"], row["model"], None, "Europe"),
                year,
                registry_version=_GENERATION_REGISTRY,
            )
            selection_end = selection.end_year if selection.end_year is not None else 9999
            if entry is None and selection.start_year <= year <= selection_end:
                ambiguous.add(year)
            elif entry is not None and entry.display_name == selection.name:
                selected.append(row)
        return selected, ambiguous

    def _market_result(self, code, name, rows, selection, horizon):  # type: ignore[no-untyped-def]
        selected = [
            row
            for row in rows
            if (row["geography"] in _EU27 if code == "EU27" else row["geography"] == code)
        ]
        if not selected:
            return MarketVehicleForecast(code, name, "unavailable", None, 0, None, None)
        registrations = sum((Decimal(row["registrations"]) for row in selected), Decimal(0))
        fleet = tuple(
            sum((Decimal(row[field]) for row in selected), Decimal(0))
            for field in ("active_fleet_p10", "active_fleet_p50", "active_fleet_p90")
        )
        components = [
            (
                tuple(
                    Decimal(row[field])
                    for field in ("active_fleet_p10", "active_fleet_p50", "active_fleet_p90")
                ),
                self._hazard.interval(
                    age_years=horizon - row["registration_cohort_year"],
                    geography=row["geography"],
                ),
            )
            for row in selected
        ]
        effective = tuple(_effective_hazard(components, index) for index in range(3))
        interval = self._uncertainty.estimate(
            active_fleet_p10=fleet[0],
            active_fleet_p50=fleet[1],
            active_fleet_p90=fleet[2],
            hazard_p10=effective[0],
            hazard_p50=effective[1],
            hazard_p90=effective[2],
            seed=_seed(self._data_version, selection.key, code, str(horizon)),
        )
        return MarketVehicleForecast(
            code,
            name,
            "available",
            _units(registrations),
            len(selected),
            DemandRange(*(_units(value) for value in fleet)),
            DemandRange(_units(interval.p10), _units(interval.p50), _units(interval.p90)),
        )

    def _identity_index(self, connection: sqlite3.Connection) -> VehicleIdentityIndex:
        """Build the canonical make/model index once per repository instance.

        Both subqueries are aggregated before they meet, because joining
        `opportunity_estimate` straight onto `cohort_estimate` fans out to one
        row per opportunity-cohort pair and multiplies the registration volume
        that the brand ranking depends on.
        """

        if self._index is None:
            self._index = VehicleIdentityIndex.from_rows(
                connection.execute(
                    """SELECT v.vehicle_id, v.make, v.model, COALESCE(c.registrations, 0)
                    FROM canonical_vehicle v
                    JOIN (
                        SELECT DISTINCT canonical_vehicle_id AS vehicle_id
                        FROM opportunity_estimate
                    ) o ON o.vehicle_id = v.vehicle_id
                    LEFT JOIN (
                        SELECT canonical_vehicle_id AS vehicle_id,
                            SUM(CAST(registrations AS REAL)) AS registrations
                        FROM cohort_estimate GROUP BY canonical_vehicle_id
                    ) c ON c.vehicle_id = v.vehicle_id"""
                ).fetchall()
            )
        return self._index

    def _brands(
        self, connection: sqlite3.Connection, *, include_all: bool = False
    ) -> tuple[str, ...]:
        index = self._identity_index(connection)
        if not self._verified_only:
            return index.makes(include_all=include_all)
        return tuple(
            sorted(
                {
                    option.brand
                    for option in self._vehicle_options(connection, None, include_all=True)
                },
                key=lambda value: (value.casefold(), value),
            )
        )

    def _vehicle_options(
        self,
        connection: sqlite3.Connection,
        search: str | None,
        *,
        include_all: bool = False,
    ) -> tuple[VehicleOption, ...]:
        """Canonical vehicles matching a free-text search or a chosen brand.

        Matching is an AND over the query's tokens against the canonical make
        and model, with any registration year removed first. The previous single
        `LIKE '%vw golf 2020%'` could not match anything once a year was typed,
        and offered raw trim strings when it did.
        """

        index = self._identity_index(connection)
        term = (search or "").strip()
        identities = (
            index.match(parse_vehicle_query(term)) if term else index.identities()
        )
        options: list[VehicleOption] = []
        reviewed: set[tuple[str, str]] = set()
        for identity in identities:
            vehicle = CanonicalVehicle(
                "option", identity.display_make, identity.display_model, None, "Europe"
            )
            profile = self._catalog.profile_for(vehicle)
            if profile is None:
                if self._verified_only:
                    continue
                option = VehicleOption(identity.display_make, identity.display_model)
            else:
                option = VehicleOption(*profile.aliases[0])
                reviewed.add((option.brand, option.model))
            options.append(option)
        unique: dict[tuple[str, str], VehicleOption] = {}
        for option in options:
            unique.setdefault((option.brand, option.model), option)
        if term:
            # `match` already ranks by exactness then volume; keep that order.
            return tuple(unique.values())
        ordered = tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    (item.brand, item.model) not in reviewed,
                    item.brand.casefold(),
                    item.model.casefold(),
                ),
            )
        )
        return ordered if include_all else ordered[:200]

    def _vehicle_ids(
        self,
        connection: sqlite3.Connection,
        brand: str,
        model: str,
        profile: VehicleGenerationProfile | None,
    ) -> tuple[str, ...]:
        """Every raw vehicle id stored under any spelling of this selection.

        Resolving only the exact `(make, model)` the user picked left each
        publisher's spelling in its own silo, which is why a selection could
        offer a truncated year range and report no data for markets whose rows
        were filed under another spelling.
        """

        index = self._identity_index(connection)
        resolved = set(index.vehicle_ids(brand, model))
        aliases = (
            set(profile._normalized_aliases)
            if profile is not None
            else {(normalize_vehicle_label(brand), normalize_vehicle_label(model))}
        )
        for row in connection.execute("SELECT vehicle_id, make, model FROM canonical_vehicle"):
            if (normalize_vehicle_label(row[1]), normalize_vehicle_label(row[2])) in aliases:
                resolved.add(row[0])
        return tuple(sorted(resolved))

    @staticmethod
    def _horizons(
        connection: sqlite3.Connection, vehicle_ids: tuple[str, ...]
    ) -> tuple[int, ...]:
        where, parameters = _in_filter("canonical_vehicle_id", vehicle_ids)
        return tuple(
            row[0]
            for row in connection.execute(
                f"SELECT DISTINCT horizon_year FROM opportunity_estimate "
                f"WHERE {where} ORDER BY horizon_year",
                parameters,
            )
        )

    @staticmethod
    def _cohort_rows(
        connection: sqlite3.Connection,
        vehicle_ids: tuple[str, ...],
        horizon: int,
    ) -> list[sqlite3.Row]:
        vehicle_where, vehicle_parameters = _in_filter(
            "canonical_vehicle_id", vehicle_ids
        )
        opportunity_ids = tuple(
            row[0]
            for row in connection.execute(
                f"SELECT opportunity_id FROM opportunity_estimate "
                f"WHERE horizon_year = ? AND {vehicle_where}",
                (horizon, *vehicle_parameters),
            )
        )
        rows: list[sqlite3.Row] = []
        for start in range(0, len(opportunity_ids), 400):
            batch = opportunity_ids[start : start + 400]
            placeholders = ", ".join("?" for _ in batch)
            rows.extend(
                connection.execute(
                    f"""SELECT c.cohort_id, c.generation_id, c.geography,
                        c.registration_cohort_year, c.registrations,
                        c.active_fleet_p10, c.active_fleet_p50, c.active_fleet_p90,
                        c.survival_method, c.reason_codes, v.make, v.model
                    FROM opportunity_input oi
                    JOIN cohort_estimate c ON c.cohort_id = oi.cohort_id
                    JOIN canonical_vehicle v ON v.vehicle_id = c.canonical_vehicle_id
                    WHERE oi.opportunity_id IN ({placeholders})""",
                    batch,
                ).fetchall()
            )
        return sorted(
            rows,
            key=lambda row: (
                row["geography"], row["registration_cohort_year"], row["cohort_id"]
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"{self._path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection


def _in_filter(column: str, values: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    return f"{column} IN ({', '.join('?' for _ in values)})", values


def _effective_hazard(components, position: int) -> Decimal:  # type: ignore[no-untyped-def]
    weighted = [(fleet[position], hazard[position]) for fleet, hazard in components]
    total = sum((fleet for fleet, _hazard in weighted), Decimal(0))
    return (
        Decimal(0)
        if total == 0
        else sum((fleet * hazard for fleet, hazard in weighted), Decimal(0)) / total
    )


def _units(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256(":".join(parts).encode()).digest()[:8], "big")


def _survival_method(rows: list[sqlite3.Row]) -> str:
    """Report the curve that actually produced these cohorts.

    The fleet quantiles are read from `cohort_estimate`, so constructing a
    survival model here and reporting its method described the code rather than
    the data. The shared rule now lives in `icor.evidence.provenance`, because
    this is the fourth channel to need it.
    """

    return resolve_reported_method(
        (row["survival_method"] for row in rows), label="survival method"
    )


def _latest_evidence_year(rows: list[sqlite3.Row]) -> int:
    evidence_years = [
        row["registration_cohort_year"]
        for row in rows
        if {
            "observed-registration-cohort",
            "reconciled-registration-cohort",
        }.intersection(json.loads(row["reason_codes"]))
    ]
    if not evidence_years:
        raise VehicleForecastSelectionError(
            "no observed or reconciled registration boundary is available for this vehicle"
        )
    return max(evidence_years)

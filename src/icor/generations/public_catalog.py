"""Strict, reviewed public generation windows for explicitly matched vehicles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlparse

from icor.domain.evidence import CanonicalVehicle
from icor.domain.generations import GenerationEntry, GenerationIdentityKind
from icor.evidence.normalization import normalize_vehicle_label, stable_evidence_id

_VOLKSWAGEN_GOLF_MAKE_ALIASES = (
    "Volkswagen",
    "Volkswagen VW",
    "Volkswagen. VW",
    "Volkswagen, VW",
    "VW",
)
_VOLKSWAGEN_GOLF_MODEL_ALIASES = (
    "Golf",
    "Golf GTE",
    "Golf / 1.5 / TSI AUT.",
    "Golf / 1.5 / TSI",
    "Golf Variant / 1.5 / TSI AUT.",
    "Golf Variant / 1.5 / TSI",
    "Golf Variant / 2.0 / TDI AUT.",
    "Golf Variant / 2.0 / TDI",
    "Golf / 2.0 / GTI AUT.",
    "Golf / 2.0 / TDI AUT.",
    "Golf / 2.0 / TDI",
    "Golf Variant",
    "Golf / 1.0 / TSI",
    "Golf / 2.0 / TSI AUT.",
    "Golf Style eHybrid",
    "Golf / 2.0 / TSI 4M AUT.",
    "Golf Variant / 2.0 / TDI 4M AUT.",
    "Golf Life",
    "Golf Style",
    "Golf Variant / 1.0 / TSI",
)


@dataclass(frozen=True, slots=True)
class GenerationWindow:
    key: str
    display_name: str
    start_month: date
    end_month: date | None
    source_url: str
    confidence_reason: str
    body_style: str | None = None
    facelift: str | None = None
    platform: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.key, "generation key"),
            (self.display_name, "generation display name"),
            (self.confidence_reason, "generation confidence reason"),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{label} is required")
        if type(self.start_month) is not date or self.start_month.day != 1:
            raise ValueError("generation start month must use first-day month precision")
        if self.end_month is not None:
            if type(self.end_month) is not date or self.end_month.day != 1:
                raise ValueError("generation end month must use first-day month precision")
            if self.end_month < self.start_month:
                raise ValueError("generation months must be ordered")
        parsed_url = urlparse(self.source_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("generation source URL must be an absolute HTTPS URL")


@dataclass(frozen=True, slots=True)
class VehicleGenerationProfile:
    aliases: tuple[tuple[str, str], ...]
    market: str
    dependency_group: str
    windows: tuple[GenerationWindow, ...]
    _normalized_aliases: tuple[tuple[str, str], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.aliases:
            raise ValueError("generation profile aliases are required")
        normalized_aliases: list[tuple[str, str]] = []
        for make, model in self.aliases:
            normalized_make = normalize_vehicle_label(make)
            normalized_model = normalize_vehicle_label(model)
            if normalized_make is None or normalized_model is None:
                raise ValueError("generation profile alias is invalid")
            normalized_aliases.append((normalized_make, normalized_model))
        if len(normalized_aliases) != len(set(normalized_aliases)):
            raise ValueError("generation profile contains a duplicate alias")
        if type(self.market) is not str or not self.market.strip():
            raise ValueError("generation profile market is required")
        if type(self.dependency_group) is not str or not self.dependency_group.strip():
            raise ValueError("generation profile dependency group is required")
        if not self.windows:
            raise ValueError("generation profile windows are required")
        starts = tuple(window.start_month for window in self.windows)
        if starts != tuple(sorted(starts)) or len(starts) != len(set(starts)):
            raise ValueError("generation profile windows must be chronological")
        keys = tuple(window.key for window in self.windows)
        if len(keys) != len(set(keys)):
            raise ValueError("generation profile contains a duplicate generation key")
        if any(window.end_month is None for window in self.windows[:-1]):
            raise ValueError("only the last generation window may be open-ended")
        object.__setattr__(self, "_normalized_aliases", tuple(normalized_aliases))


@dataclass(frozen=True, slots=True)
class ReviewedGenerationCatalog:
    profiles: tuple[VehicleGenerationProfile, ...]
    _profiles_by_alias: dict[tuple[str, str], VehicleGenerationProfile] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        profiles_by_alias: dict[tuple[str, str], VehicleGenerationProfile] = {}
        for profile in self.profiles:
            for alias in profile._normalized_aliases:
                if alias in profiles_by_alias:
                    raise ValueError("duplicate reviewed vehicle alias")
                profiles_by_alias[alias] = profile
        object.__setattr__(self, "_profiles_by_alias", profiles_by_alias)

    def entries_for(
        self,
        vehicle: CanonicalVehicle,
        *,
        registry_version: str,
    ) -> tuple[GenerationEntry, ...]:
        make = normalize_vehicle_label(vehicle.make)
        model = normalize_vehicle_label(vehicle.model)
        profile = self._profiles_by_alias.get((make, model)) if make and model else None
        if profile is None or profile.market != vehicle.market:
            return ()
        return tuple(
            GenerationEntry(
                generation_id=stable_evidence_id(
                    "generation-public",
                    vehicle.vehicle_id,
                    window.key,
                    registry_version,
                ),
                canonical_vehicle_id=vehicle.vehicle_id,
                display_name=window.display_name,
                market=vehicle.market,
                start_month=window.start_month,
                end_month=window.end_month,
                identity_kind=GenerationIdentityKind.MANUFACTURER_CONFIRMED,
                body_style=window.body_style,
                facelift=window.facelift,
                platform=window.platform,
                evidence_ids=(window.source_url,),
                dependency_groups=(profile.dependency_group,),
                confidence_reasons=(window.confidence_reason,),
                registry_version=registry_version,
            )
            for window in profile.windows
        )

    def profile_for(self, vehicle: CanonicalVehicle) -> VehicleGenerationProfile | None:
        """Return the reviewed profile for an exact explicit alias and market."""

        make = normalize_vehicle_label(vehicle.make)
        model = normalize_vehicle_label(vehicle.model)
        profile = self._profiles_by_alias.get((make, model)) if make and model else None
        return profile if profile is not None and profile.market == vehicle.market else None

    def entry_for_year(
        self,
        vehicle: CanonicalVehicle,
        registration_year: int,
        *,
        registry_version: str,
    ) -> GenerationEntry | None:
        """Return only an unambiguous reviewed window for an annual cohort."""

        if type(registration_year) is not int:
            raise ValueError("registration year must be an integer")
        candidates = tuple(
            entry
            for entry in self.entries_for(vehicle, registry_version=registry_version)
            if entry.start_month.year <= registration_year
            and (entry.end_month is None or entry.end_month.year >= registration_year)
        )
        return candidates[0] if len(candidates) == 1 else None


def official_public_generation_catalog() -> ReviewedGenerationCatalog:
    """Return the reviewed catalog; absence always falls back instead of guessing."""

    return ReviewedGenerationCatalog(
        (
            VehicleGenerationProfile(
                aliases=tuple(
                    (make, model)
                    for make in _VOLKSWAGEN_GOLF_MAKE_ALIASES
                    for model in _VOLKSWAGEN_GOLF_MODEL_ALIASES
                ),
                market="Europe",
                dependency_group="manufacturer-model-history",
                windows=(
                    GenerationWindow(
                        key="volkswagen-golf-mk6-europe",
                        display_name="Golf Mk6",
                        start_month=date(2008, 1, 1),
                        end_month=date(2012, 12, 1),
                        source_url=("https://www.volkswagen-newsroom.com/en/golf-6-20082012-19484"),
                        confidence_reason=(
                            "Volkswagen identifies Golf VI as the 2008-2012 generation."
                        ),
                        platform="PQ35",
                    ),
                    GenerationWindow(
                        key="volkswagen-golf-mk7-europe",
                        display_name="Golf Mk7",
                        start_month=date(2012, 9, 1),
                        end_month=date(2019, 12, 1),
                        source_url=("https://www.volkswagen-newsroom.com/en/golf-7-20122019-20035"),
                        confidence_reason=(
                            "Volkswagen identifies Golf VII as the 2012-2019 generation; "
                            "its world premiere was in September 2012."
                        ),
                        platform="MQB",
                    ),
                    GenerationWindow(
                        key="volkswagen-golf-mk8-europe",
                        display_name="Golf Mk8",
                        start_month=date(2019, 10, 1),
                        end_month=None,
                        source_url=(
                            "https://www.volkswagen-newsroom.com/en/press-releases/"
                            "a-world-bestseller-celebrates-its-50th-birthday-volkswagen-"
                            "started-production-of-the-first-golf-on-29-march-1974-18313"
                        ),
                        confidence_reason=(
                            "Volkswagen presented Golf VIII in October 2019 and identifies "
                            "the 2024 update as an evolutionary stage of that generation."
                        ),
                    ),
                ),
            ),
        )
    )

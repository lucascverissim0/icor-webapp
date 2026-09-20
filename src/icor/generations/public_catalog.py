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


def _ranked_vehicle_profiles() -> tuple[VehicleGenerationProfile, ...]:
    """Manufacturer-backed generations for the vehicle-years ranked for ICOR work."""

    def profile(
        aliases: tuple[tuple[str, str], ...],
        key: str,
        name: str,
        start_year: int,
        end_year: int | None,
        source_url: str,
        reason: str,
        *,
        platform: str | None = None,
    ) -> VehicleGenerationProfile:
        return VehicleGenerationProfile(
            aliases=aliases,
            market="Europe",
            dependency_group="manufacturer-model-history",
            windows=(
                GenerationWindow(
                    key=key,
                    display_name=name,
                    start_month=date(start_year, 1, 1),
                    end_month=date(end_year, 12, 1) if end_year is not None else None,
                    source_url=source_url,
                    confidence_reason=reason,
                    platform=platform,
                ),
            ),
        )

    volkswagen_makes = ("Volkswagen", "Volkswagen VW", "Volkswagen. VW", "Volkswagen, VW", "VW")
    return (
        profile(
            tuple((make, "Octavia") for make in ("Skoda", "Skoda Auto")),
            "skoda-octavia-mk3-5e-europe",
            "Octavia Mk3 (5E)",
            2012,
            2020,
            "https://www.skoda-storyboard.com/en/press-releases/a-legend-celebrates-its-25th-anniversary-four-generations-of-the-brands-bestseller-the-skoda-octavia/",
            "Skoda documents the third modern Octavia generation as produced from 2012 to 2020.",
            platform="5E",
        ),
        profile(
            tuple((make, "Passat") for make in volkswagen_makes),
            "volkswagen-passat-b8-3g-europe",
            "Passat B8 (3G)",
            2015,
            2023,
            "https://www.volkswagen-newsroom.com/en/passat-b7-20102014-20036",
            "Volkswagen records the B7 ending in February 2015 when the B8 replaced it.",
            platform="3G",
        ),
        profile(
            (("Ford", "Focus"),),
            "ford-focus-mk4-c519-europe",
            "Focus Mk4 (C519)",
            2018,
            None,
            "https://media.ford.com/content/fordmedia/feu/en/news/2018/04/10/ford-unveils-all-new-focus--most-innovative--dynamic-and-excitin.html",
            "Ford introduced the fourth Focus generation in Europe in 2018.",
            platform="C519",
        ),
        profile(
            (("Ford", "Fiesta"),),
            "ford-fiesta-mk7-b299-europe",
            "Fiesta Mk7 (B299)",
            2008,
            2017,
            "https://media.ford.com/content/fordmedia/feu/gb/en/news/2020/10/26/ford-seals-awards-haul-with-car--van-and-pick-up-hat-trick-from-.html",
            "Ford identifies the 2008-2017 Fiesta as the seventh generation.",
            platform="B299",
        ),
        VehicleGenerationProfile(
            aliases=(("Ford", "Kuga"),),
            market="Europe",
            dependency_group="manufacturer-model-history",
            windows=(
                GenerationWindow(
                    key="ford-kuga-mk2-c520-europe",
                    display_name="Kuga Mk2 (C520)",
                    start_month=date(2012, 1, 1),
                    end_month=date(2019, 12, 1),
                    source_url="https://media.ford.com/content/fordmedia/feu/gb/en/news/2015/09/10/efficient-and-refined-all-new-ford-edge-expands-ford-suv-and-awd0.html",
                    confidence_reason=(
                        "Ford identifies the Kuga introduced in Europe in 2012 "
                        "as the second generation."
                    ),
                    platform="C520",
                ),
                GenerationWindow(
                    key="ford-kuga-mk3-cx482-europe",
                    display_name="Kuga Mk3 (CX482)",
                    start_month=date(2020, 1, 1),
                    end_month=None,
                    source_url="https://media.ford.com/content/fordmedia/feu/de/de/news/2020/02/21/neuer-ford-kuga--kraftstoffverbrauch-und-co2-emissionen-um-ueber.html",
                    confidence_reason="Ford launched the third Kuga generation in Europe in 2020.",
                    platform="CX482",
                ),
            ),
        ),
        profile(
            (("Hyundai", "I20"),),
            "hyundai-i20-mk2-gb-europe",
            "i20 Mk2 (GB)",
            2015,
            2020,
            "https://www.hyundai.news/newsroom/dam/eu/press-kits/2015_i20_coupe/NewGeneration_i20_Coupe_Press_Information_032015.pdf",
            "Hyundai's European press kit identifies the new-generation i20 introduced for 2015.",
            platform="GB",
        ),
        profile(
            (("Citroen", "C4"), ("Citroen", "C4 Berline")),
            "citroen-c4-mk2-b7-europe",
            "C4 Mk2 (B7)",
            2010,
            2018,
            "https://www.media.stellantis.com/em-en/citroen/press/the-citroen-c4-new-range-what-s-new-in-2015",
            "Citroen identifies the C4 launched in 2010 as its second generation.",
            platform="B7",
        ),
        profile(
            (("Volvo", "Xc40"),),
            "volvo-xc40-mk1-536-europe",
            "XC40 Mk1 (536)",
            2018,
            None,
            "https://www.media.volvocars.com/uk/en-gb/models/xc40/2022",
            "Volvo's model archive starts the original XC40 generation at model year 2018.",
            platform="536",
        ),
        profile(
            tuple((make, "Superb") for make in ("Skoda", "Skoda Auto")),
            "skoda-superb-mk3-3v-europe",
            "Superb Mk3 (3V)",
            2015,
            2023,
            "https://www.skoda-storyboard.com/en/press-kits/new-skoda-superb-press-kit/",
            "Skoda launched the third Superb generation in June 2015.",
            platform="3V",
        ),
        profile(
            (("Ford", "Mondeo"),),
            "ford-mondeo-mk4-cd391-europe",
            "Mondeo Mk4 (CD391)",
            2014,
            2022,
            "https://media.ford.com/content/fordmedia/feu/en/news/2014/08/06/all-new-ford-mondeo-pricing-announced--petrol--diesel-and-first-.html",
            "Ford introduced Europe's fourth Mondeo generation in late 2014.",
            platform="CD391",
        ),
        profile(
            tuple((make, "Sharan") for make in volkswagen_makes),
            "volkswagen-sharan-mk2-7n-europe",
            "Sharan Mk2 (7N)",
            2010,
            2022,
            "https://www.volkswagen-newsroom.com/en/images/albums/sharan-2nd-generation-2010-2022-2267",
            "Volkswagen identifies the second Sharan generation as 2010-2022.",
            platform="7N",
        ),
        profile(
            (("Audi", "A3"),),
            "audi-a3-mk3-8v-europe",
            "A3 Mk3 (8V)",
            2012,
            2019,
            "https://www.audi-mediacenter.com/en/press-releases/generational-change-start-of-production-of-the-new-audi-a3-sportback-in-ingolstadt-12638",
            "Audi's model history identifies the third A3 generation as introduced in 2012.",
            platform="8V",
        ),
        profile(
            (("BMW", "Ix3"),),
            "bmw-ix3-mk1-g08-europe",
            "iX3 Mk1 (G08)",
            2021,
            2024,
            "https://www.press.bmwgroup.com/united-kingdom/article/detail/T0339376EN_GB/the-new-bmw-ix3",
            "BMW identifies the first iX3 under series code G08 for European deliveries.",
            platform="G08",
        ),
        profile(
            (("Volvo", "V40"),),
            "volvo-v40-mk2-525-526-europe",
            "V40 Mk2 (525/526)",
            2012,
            2019,
            "https://www.media.volvocars.com/global/en-gb/media/pressreleases/190783/volvo-v40-cross-country-model-year-20178",
            "Volvo records the compact V40 generation entering production in 2012.",
            platform="525/526",
        ),
        profile(
            (("Volvo", "Xc90"),),
            "volvo-xc90-mk2-256-europe",
            "XC90 Mk2 (256)",
            2015,
            None,
            "https://www.media.volvocars.com/global/en-gb/models/xc90/2015",
            "Volvo's model archive identifies the redesigned XC90 for model year 2015.",
            platform="256",
        ),
        profile(
            (("Opel", "Insignia"),),
            "opel-insignia-mk2-b-europe",
            "Insignia Mk2 (B)",
            2017,
            2022,
            "https://www.media.stellantis.com/es-es/opel/press/nuevo-motor-diesel-biturbo-para-el-opel-insignia",
            "Opel identifies the 2017 Insignia as the second generation.",
            platform="B",
        ),
        profile(
            (("Mazda", "3"), ("Mazda", "Mazda3")),
            "mazda3-mk3-bm-bn-europe",
            "Mazda3 Mk3 (BM/BN)",
            2013,
            2018,
            "https://fr.mazda-press.com/api/assets/download/edf28a25-49d5-4c88-a9ee-c81a29cf2367_Pdf?isDownload=false",
            "Mazda's press archive places the successor's world premiere in "
            "November 2018 for 2019 sale.",
            platform="BM/BN",
        ),
        profile(
            (("Honda", "Jazz"),),
            "honda-jazz-mk3-gk-europe",
            "Jazz Mk3 (GK)",
            2015,
            2019,
            "https://hondanews.eu/es/es/media/pressreleases/44022/el-nuevo-honda-jazz-redefine-el-segmento-b",
            "Honda introduced the third Jazz generation in Europe in 2015.",
            platform="GK",
        ),
    )


def ranking_public_generation_catalog() -> ReviewedGenerationCatalog:
    """Return reviewed display profiles without changing persisted generation mapping."""

    official = official_public_generation_catalog()
    return ReviewedGenerationCatalog((*official.profiles, *_ranked_vehicle_profiles()))

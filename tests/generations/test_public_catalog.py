from datetime import date

import pytest

from icor.domain.evidence import CanonicalVehicle
from icor.domain.generations import GenerationIdentityKind
from icor.generations.public_catalog import (
    GenerationWindow,
    ReviewedGenerationCatalog,
    VehicleGenerationProfile,
    official_public_generation_catalog,
)


def test_official_catalog_maps_only_explicit_reviewed_aliases() -> None:
    catalog = official_public_generation_catalog()
    volkswagen = CanonicalVehicle(
        "vehicle-volkswagen-golf-eu", "Volkswagen", "Golf", None, "Europe"
    )
    publisher_alias = CanonicalVehicle(
        "vehicle-volkswagen-vw-golf-eu",
        "VOLKSWAGEN. VW",
        "GOLF / 2.0 / GTI AUT.",
        None,
        "Europe",
    )
    unrelated = CanonicalVehicle(
        "vehicle-volkswagen-golf-plus-eu", "Volkswagen", "Golf Plus", None, "Europe"
    )

    entries = catalog.entries_for(volkswagen, registry_version="public-generations-v1")
    alias_entries = catalog.entries_for(
        publisher_alias, registry_version="public-generations-v1"
    )

    assert [entry.display_name for entry in entries] == ["Golf Mk6", "Golf Mk7", "Golf Mk8"]
    assert [entry.display_name for entry in alias_entries] == [
        "Golf Mk6",
        "Golf Mk7",
        "Golf Mk8",
    ]
    assert {entry.canonical_vehicle_id for entry in entries} == {volkswagen.vehicle_id}
    assert all(
        entry.identity_kind is GenerationIdentityKind.MANUFACTURER_CONFIRMED
        for entry in entries
    )
    selected = catalog.entry_for_year(
        volkswagen,
        2024,
        registry_version="public-generations-v1",
    )
    assert selected is not None
    assert selected.display_name == "Golf Mk8"
    assert (
        catalog.entry_for_year(
            volkswagen,
            2019,
            registry_version="public-generations-v1",
        )
        is None
    )
    assert catalog.entries_for(unrelated, registry_version="public-generations-v1") == ()


def test_catalog_rejects_an_alias_assigned_to_two_profiles() -> None:
    window = GenerationWindow(
        key="one",
        display_name="One",
        start_month=date(2020, 1, 1),
        end_month=None,
        source_url="https://manufacturer.example/one",
        confidence_reason="Manufacturer identifies this generation.",
    )
    first = VehicleGenerationProfile(
        aliases=(("Example", "Alpha"),),
        market="Europe",
        dependency_group="manufacturer-history",
        windows=(window,),
    )
    second = VehicleGenerationProfile(
        aliases=((" example ", "ALPHA"),),
        market="Europe",
        dependency_group="manufacturer-history",
        windows=(window,),
    )

    with pytest.raises(ValueError, match="duplicate reviewed vehicle alias"):
        ReviewedGenerationCatalog((first, second))


def test_catalog_rejects_unordered_generation_windows() -> None:
    newer = GenerationWindow(
        key="newer",
        display_name="Newer",
        start_month=date(2024, 1, 1),
        end_month=None,
        source_url="https://manufacturer.example/newer",
        confidence_reason="Manufacturer identifies this generation.",
    )
    older = GenerationWindow(
        key="older",
        display_name="Older",
        start_month=date(2020, 1, 1),
        end_month=date(2023, 12, 1),
        source_url="https://manufacturer.example/older",
        confidence_reason="Manufacturer identifies this generation.",
    )

    with pytest.raises(ValueError, match="chronological"):
        VehicleGenerationProfile(
            aliases=(("Example", "Alpha"),),
            market="Europe",
            dependency_group="manufacturer-history",
            windows=(newer, older),
        )

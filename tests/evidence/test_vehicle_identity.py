"""Canonical vehicle identity over raw publisher labels.

The snapshot stores whatever each publisher wrote. `canonical_vehicle` therefore
holds 557 spellings of a make and 22,892 make/model pairs, including 14 spellings
of Volkswagen and 1,194 trim-level "models" under Audi. Resolving those to one
identity at query time is what lets a single selection aggregate every silo.

Over-merging corrupts a forecast; under-merging only leaves a duplicate in a
dropdown. Every rule below is written to fail in the second direction.
"""

from __future__ import annotations

import pytest

from icor.evidence.vehicle_identity import (
    VehicleIdentityIndex,
    fold_make,
    identity_display_label,
    parse_vehicle_query,
)

VOLKSWAGEN_SPELLINGS = (
    "VOLKSWAGEN",
    "volkswagen",
    "vw",
    "VW",
    "volkswagen vw",
    "volkswagen, vw",
    "volkswagen,vw",
    "volkswagen. vw",
    "volkswagen - vw",
    "volkswagen v w",
    "volksawgen, vw",
    "Volkswagen AG",
)


@pytest.mark.parametrize("spelling", VOLKSWAGEN_SPELLINGS)
def test_every_volkswagen_spelling_folds_to_one_make(spelling: str) -> None:
    assert fold_make(spelling) == "volkswagen"


def test_distinct_manufacturers_are_never_merged() -> None:
    """ALPINA and ALPINE are different companies that fold to similar strings."""

    assert fold_make("ALPINA") == "alpina"
    assert fold_make("alpine") == "alpine"


def test_coachbuilt_sub_brands_stay_separate_from_the_parent_make() -> None:
    """`volkswagen knaus` is a campervan converter, not a Volkswagen."""

    assert fold_make("volkswagen/knaus") != "volkswagen"
    assert fold_make("mercedes-benz hymer") != fold_make("mercedes-benz")


def test_marque_and_corporate_forms_fold_together() -> None:
    assert fold_make("Mercedes") == fold_make("MERCEDES-BENZ") == "mercedes-benz"
    assert fold_make("Vauxhall") == fold_make("OPEL") == "opel"
    assert fold_make("Audi AG") == fold_make("AUDI") == "audi"


def test_accented_makes_fold_onto_their_plain_spelling() -> None:
    """Dropping the accent instead of folding it made `Škoda` a make called `koda`."""

    assert fold_make("Škoda") == fold_make("SKODA") == "skoda"
    assert fold_make("ŠKODA AUTO") == "skoda"


def test_sub_brands_that_share_model_identity_fold_to_the_parent() -> None:
    assert fold_make("BMW i") == fold_make("BMW") == "bmw"


def test_empty_and_missing_make_labels_are_rejected() -> None:
    assert fold_make("") is None
    assert fold_make("   ") is None
    assert fold_make("-") is None


def test_identity_display_label_restores_acronyms_and_casing() -> None:
    assert identity_display_label("volkswagen") == "Volkswagen"
    assert identity_display_label("bmw") == "BMW"
    assert identity_display_label("mercedes-benz") == "Mercedes-Benz"
    assert identity_display_label("a4") == "A4"


class TestParseVehicleQuery:
    def test_a_trailing_year_is_extracted_not_matched_as_text(self) -> None:
        """`VW Golf 2020` returned 0 matches because the year was matched literally."""

        parsed = parse_vehicle_query("VW Golf 2020")

        assert parsed.make == "volkswagen"
        assert parsed.tokens == ("golf",)
        assert parsed.year == 2020

    def test_a_year_anywhere_in_the_query_is_found(self) -> None:
        assert parse_vehicle_query("2020 vw golf").year == 2020
        assert parse_vehicle_query("golf 2020 gti").year == 2020

    def test_implausible_four_digit_numbers_are_not_treated_as_years(self) -> None:
        assert parse_vehicle_query("mazda 1900").year is None
        assert parse_vehicle_query("peugeot 3008").year is None

    def test_a_query_without_a_known_make_keeps_every_token(self) -> None:
        parsed = parse_vehicle_query("golf sportsvan")

        assert parsed.make is None
        assert parsed.tokens == ("golf", "sportsvan")
        assert parsed.year is None

    def test_an_empty_query_is_inert(self) -> None:
        parsed = parse_vehicle_query("   ")

        assert parsed.make is None
        assert parsed.tokens == ()
        assert parsed.year is None


ROWS = (
    # vehicle_id, raw make, raw model, registrations
    ("v1", "volkswagen", "golf", 7_000_000.0),
    ("v2", "VOLKSWAGEN", "Golf", 3_000_000.0),
    ("v3", "volkswagen, vw", "volkswagen golf", 2_000_000.0),
    ("v4", "vw", "golf se navigation tdi s-a", 900_000.0),
    ("v5", "volkswagen", "golf sportsvan", 1_100_000.0),
    ("v6", "volkswagen", "golf plus", 400_000.0),
    ("v7", "audi", "a4", 1_300_000.0),
    ("v8", "AUDI", "A4 2.0 TDI AUT.", 90_000.0),
    ("v9", "audi", "a4 avant", 1_400_000.0),
    ("v10", "audi ag", "audi a4", 600_000.0),
    ("v11", "audi", "a4 / 2.0 / tdi q aut.", 40_000.0),
    ("v12", "AGRIFAC", "condor", 12.0),
    ("v13", "audi", "a4 , s4", 5_000.0),
)


@pytest.fixture
def index() -> VehicleIdentityIndex:
    return VehicleIdentityIndex.from_rows(ROWS)


class TestVehicleIdentityIndex:
    def test_one_make_appears_once_however_many_ways_it_was_spelled(
        self, index: VehicleIdentityIndex
    ) -> None:
        makes = index.makes(include_all=True)

        assert makes.count("Volkswagen") == 1
        assert makes.count("Audi") == 1
        assert len(makes) == len(set(makes))

    def test_trim_labels_collapse_onto_the_base_model(
        self, index: VehicleIdentityIndex
    ) -> None:
        """`a4 / 2.0 / tdi q aut.` and `A4 2.0 TDI AUT.` are the same car as `a4`."""

        models = index.models("audi")

        assert "A4" in models
        assert not any(model.lower().startswith("a4 2.0") for model in models)
        assert not any("/" in model for model in models)

    def test_a_model_label_that_repeats_the_make_is_not_a_separate_model(
        self, index: VehicleIdentityIndex
    ) -> None:
        assert index.models("volkswagen").count("Golf") == 1
        assert "Volkswagen Golf" not in index.models("volkswagen")

    def test_distinct_bodies_are_kept_apart(self, index: VehicleIdentityIndex) -> None:
        """Merging these would sum two different windshields into one forecast."""

        assert set(index.models("audi")) >= {"A4", "A4 Avant"}
        assert set(index.models("volkswagen")) >= {"Golf", "Golf Sportsvan", "Golf Plus"}

    def test_a_selection_resolves_to_every_spelling_it_was_stored_under(
        self, index: VehicleIdentityIndex
    ) -> None:
        """This is what recovers the cohorts and markets the silos were hiding."""

        assert set(index.vehicle_ids("volkswagen", "golf")) == {"v1", "v2", "v3", "v4"}
        assert set(index.vehicle_ids("audi", "a4")) == {"v7", "v8", "v10", "v11"}

    def test_a_selection_by_display_label_resolves_the_same_way(
        self, index: VehicleIdentityIndex
    ) -> None:
        assert index.vehicle_ids("Volkswagen", "Golf") == index.vehicle_ids(
            "volkswagen", "golf"
        )

    def test_merged_volume_is_the_sum_of_the_silos(
        self, index: VehicleIdentityIndex
    ) -> None:
        assert index.volume("volkswagen", "golf") == pytest.approx(12_900_000.0)

    def test_a_label_naming_two_models_is_excluded_rather_than_mismerged(
        self, index: VehicleIdentityIndex
    ) -> None:
        assert "v13" not in index.vehicle_ids("audi", "a4")
        assert not any(model.lower() == "a4 , s4" for model in index.models("audi"))

    def test_low_volume_makes_are_hidden_by_default_and_never_deleted(
        self, index: VehicleIdentityIndex
    ) -> None:
        """AGRIFAC builds farm sprayers; it should not head a passenger-car list."""

        assert "Agrifac" not in index.makes(include_all=False)
        assert "Agrifac" in index.makes(include_all=True)

    def test_makes_are_listed_alphabetically_for_a_scannable_dropdown(
        self, index: VehicleIdentityIndex
    ) -> None:
        makes = index.makes(include_all=True)

        assert list(makes) == sorted(makes, key=str.casefold)

    def test_search_matches_on_all_tokens_not_one_substring(
        self, index: VehicleIdentityIndex
    ) -> None:
        matches = index.match(parse_vehicle_query("VW Golf 2020"))

        assert [(item.make, item.model) for item in matches][0] == ("volkswagen", "golf")

    def test_search_ranks_the_exact_model_above_its_variants(
        self, index: VehicleIdentityIndex
    ) -> None:
        matches = index.match(parse_vehicle_query("volkswagen golf"))

        assert [item.model for item in matches][0] == "golf"
        assert {"golf sportsvan", "golf plus"} <= {item.model for item in matches}

    def test_search_without_a_make_still_finds_the_model(
        self, index: VehicleIdentityIndex
    ) -> None:
        assert ("audi", "a4") in [
            (item.make, item.model) for item in index.match(parse_vehicle_query("a4"))
        ]

    def test_an_unmatched_query_returns_nothing_rather_than_guessing(
        self, index: VehicleIdentityIndex
    ) -> None:
        assert index.match(parse_vehicle_query("ferrari f40")) == ()

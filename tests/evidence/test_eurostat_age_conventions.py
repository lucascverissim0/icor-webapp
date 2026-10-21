"""Which registration years each office counts in Eurostat's youngest age band."""

from __future__ import annotations

import pytest

from icor.evidence.eurostat_age_conventions import (
    THREE_YEARS,
    TWO_YEARS,
    UNCLEAR,
    classify,
    measure,
)
from icor.evidence.eurostat_new_registrations import NewRegistrations

REGISTRATIONS = {2021: 100, 2022: 100, 2023: 100, 2024: 100, 2025: 100}


@pytest.mark.parametrize(
    ("youngest", "convention"),
    [(196, TWO_YEARS), (290, THREE_YEARS), (240, UNCLEAR), (120, UNCLEAR)],
)
def test_the_youngest_band_is_matched_to_two_or_three_registration_years(youngest, convention):
    assert classify(youngest, REGISTRATIONS, 2025)[0] == convention


def test_missing_registrations_leave_the_convention_unclear():
    assert classify(200, {2025: 100}, 2025) == (UNCLEAR, None, None)


def test_registrations_with_used_imports_or_another_definition_cannot_test_the_band():
    registrations = {year: NewRegistrations(100) for year in range(2008, 2013)}
    flagged = {**registrations, 2012: NewRegistrations(100, "d")}

    # Before 2013 Poland's new registrations included used imports.
    assert measure({2012: 200}, registrations, "PL")[2012].convention == UNCLEAR
    assert measure({2012: 200}, registrations, "DE")[2012].convention == TWO_YEARS
    assert measure({2012: 200}, flagged, "DE")[2012].convention == UNCLEAR


def test_a_three_year_convention_shifts_band_edges_by_one():
    (convention,) = measure(
        {2025: 300}, {y: NewRegistrations(100) for y in REGISTRATIONS}, "AT"
    ).values()

    assert (convention.convention, convention.shift) == (THREE_YEARS, 1)
    assert (convention.two_year_ratio, convention.three_year_ratio) == (1.5, 1.0)

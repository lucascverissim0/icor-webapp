"""Which registration years each office counts in Eurostat's youngest age band."""

from __future__ import annotations

import pytest

from icor.evidence.eurostat_age_conventions import (
    THREE_YEARS,
    TWO_FOUR_YEARS,
    TWO_YEARS,
    UNCLEAR,
    classify,
    classify_second,
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

    assert convention.convention == THREE_YEARS
    assert (convention.two_year_ratio, convention.three_year_ratio) == (1.5, 1.0)


def _italy_like(years: range, *, second_years: int) -> tuple[dict, dict, dict]:
    """Flat new registrations; the youngest band is two years, the second ``second_years``."""

    registrations = {year: NewRegistrations(100) for year in range(years[0] - 10, years[-1] + 1)}
    youngest = {year: 200 for year in years}
    second = {year: 100 * second_years for year in years}
    return youngest, registrations, second


def test_the_second_band_tells_three_from_four_registration_years():
    flat = {year: 100 for year in range(2010, 2026)}

    assert classify_second(300, 1.0, 2, flat, 2025)[0] == 3
    assert classify_second(400, 1.0, 2, flat, 2025)[0] == 4
    # Halfway between, the test cannot tell.
    assert classify_second(350, 1.0, 2, flat, 2025)[0] is None
    assert classify_second(400, 1.0, 2, {2025: 100}, 2025) == (None, None, None)


def test_five_consecutive_four_year_findings_make_the_italian_layout():
    youngest, registrations, second = _italy_like(range(2013, 2026), second_years=4)

    found = measure(youngest, registrations, "IT", second)

    assert {item.convention for item in found.values()} == {TWO_FOUR_YEARS}
    assert (found[2025].second_three_year_ratio, found[2025].second_four_year_ratio) == (
        1.3333,
        1.0,
    )
    assert found[2025].second_years == 4


def test_a_shorter_run_of_four_year_findings_keeps_the_two_year_reading():
    # Young used imports can swell "2 to 5 years" for a few years (Romania 2018-2021).
    youngest, registrations, second = _italy_like(range(2013, 2026), second_years=3)
    second.update({year: 400 for year in range(2018, 2022)})

    found = measure(youngest, registrations, "HR", second)

    assert {item.convention for item in found.values()} == {TWO_YEARS}
    assert found[2019].second_years == 4


def test_without_the_second_band_the_youngest_band_decides_alone():
    youngest, registrations, _second = _italy_like(range(2013, 2026), second_years=4)

    found = measure(youngest, registrations, "IT")

    assert {item.convention for item in found.values()} == {TWO_YEARS}
    assert found[2025].second_years is None

"""Which registration years each office counts in Eurostat's age bands.

Eurostat's `road_eqs_carage` bands read "less than 2 years", "2 to 5 years" and so
on, but national offices age a car differently. Some count the stock year and
the one before as "less than 2" (a car registered in the stock year or the year
before), others count three registration years, which shifts every later band
edge by one year as well. The convention is not in the data's metadata, so it is
measured: the youngest band is compared with the office's own new registrations
(`road_eqr_carmot`) of the last two and the last three years. Under the right
reading the ratio is near one (a few cars are scrapped or exported, a few young
used imports arrive); under the wrong one it is near 1.5 or 0.67.

A second, independent test reads the "2 to 5 years" band. Few cars leave a
fleet between their second and sixth year, so under the right reading that band
keeps nearly the youngest band's ratio to new registrations. Italy counts four
registration years there (ages 2 to 5), not three: the three-year reading gives
1.26-1.49 every year from 2013, the four-year one 0.98-1.03. Young used imports
and exports blur this test from one year to the next, so the four-year layout
is adopted only when the test finds it in at least five consecutive years.

A market whose young used imports or exports blur the first test is left
unclear, and the report compares it under both readings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from icor.domain.age_bands import THREE_YEARS, TWO_FOUR_YEARS, TWO_YEARS, UNCLEAR
from icor.eurocodes.data_coverage import OfficialStock
from icor.evidence.eurostat_new_registrations import NewRegistrations, includes_used_imports

__all__ = [
    "THREE_YEARS",
    "TWO_FOUR_YEARS",
    "TWO_YEARS",
    "UNCLEAR",
    "Convention",
    "classify",
    "classify_second",
    "for_stock",
    "measure",
]

#: The youngest band divided by the matching registrations lies within this range
#: under the right reading.
LOW, HIGH = 0.85, 1.15
#: Under the right reading the second band's ratio lies this close to the
#: youngest band's ...
SECOND_TOLERANCE = 0.15
#: ... and at least this much closer than under the other reading.
SECOND_MARGIN = 0.10
#: Consecutive years the second test must find four registration years in.
FOUR_YEAR_RUN = 5


@dataclass(frozen=True, slots=True)
class Convention:
    geography: str
    year: int
    convention: str
    #: Youngest band over the last two and last three years' new registrations.
    two_year_ratio: float | None
    three_year_ratio: float | None
    #: "2 to 5 years" over three and four registration years after the youngest
    #: band's, and how many the second test found (3, 4 or None when it cannot tell).
    second_three_year_ratio: float | None = None
    second_four_year_ratio: float | None = None
    second_years: int | None = None


def classify(
    youngest: int, registrations: Mapping[int, int], year: int
) -> tuple[str, float | None, float | None]:
    two = [registrations.get(year - offset) for offset in range(2)]
    three = [*two, registrations.get(year - 2)]
    two_ratio = youngest / sum(two) if all(two) else None  # type: ignore[arg-type]
    three_ratio = youngest / sum(three) if all(three) else None  # type: ignore[arg-type]
    if two_ratio is None or three_ratio is None:
        return UNCLEAR, two_ratio, three_ratio
    two_fits = LOW <= two_ratio <= HIGH
    three_fits = LOW <= three_ratio <= HIGH
    if two_fits and not three_fits:
        return TWO_YEARS, two_ratio, three_ratio
    if three_fits and not two_fits:
        return THREE_YEARS, two_ratio, three_ratio
    return UNCLEAR, two_ratio, three_ratio


def classify_second(
    second: int, youngest_ratio: float, first_age: int, registrations: Mapping[int, int], year: int
) -> tuple[int | None, float | None, float | None]:
    """How many registration years "2 to 5 years" counts, starting at ``first_age``."""

    years = [registrations.get(year - age) for age in range(first_age, first_age + 4)]
    three_ratio = second / sum(years[:3]) if all(years[:3]) else None  # type: ignore[arg-type]
    four_ratio = second / sum(years) if all(years) else None  # type: ignore[arg-type]
    if three_ratio is None or four_ratio is None:
        return None, three_ratio, four_ratio
    three_off = abs(three_ratio - youngest_ratio)
    four_off = abs(four_ratio - youngest_ratio)
    if three_off <= SECOND_TOLERANCE and four_off - three_off >= SECOND_MARGIN:
        return 3, three_ratio, four_ratio
    if four_off <= SECOND_TOLERANCE and three_off - four_off >= SECOND_MARGIN:
        return 4, three_ratio, four_ratio
    return None, three_ratio, four_ratio


def _four_year_runs(found: Mapping[int, int | None]) -> set[int]:
    """Years inside a run of at least `FOUR_YEAR_RUN` consecutive four-year findings."""

    years: set[int] = set()
    run: list[int] = []
    for year in sorted(found):
        if found[year] == 4 and (not run or year == run[-1] + 1):
            run.append(year)
            continue
        if len(run) >= FOUR_YEAR_RUN:
            years.update(run)
        run = [year] if found[year] == 4 else []
    if len(run) >= FOUR_YEAR_RUN:
        years.update(run)
    return years


def measure(
    youngest_by_year: Mapping[int, int],
    registrations: Mapping[int, NewRegistrations],
    geography: str,
    second_by_year: Mapping[int, int] | None = None,
) -> dict[int, Convention]:
    """The convention of each year a market reports its youngest band in.

    Registrations that include used imports (some offices before 2013) or whose
    definition differs (flag ``d``) cannot test the band and are left out.
    """

    usable = {
        year: item.count
        for year, item in registrations.items()
        if item.status != "d" and not includes_used_imports(geography, year)
    }
    second_by_year = second_by_year or {}
    first: dict[int, tuple[str, float | None, float | None]] = {}
    second: dict[int, tuple[int | None, float | None, float | None]] = {}
    for year, youngest in sorted(youngest_by_year.items()):
        convention, two, three = classify(youngest, usable, year)
        first[year] = (convention, two, three)
        ratio = two if convention == TWO_YEARS else three
        if convention != UNCLEAR and ratio is not None and year in second_by_year:
            first_age = 2 if convention == TWO_YEARS else 3
            second[year] = classify_second(second_by_year[year], ratio, first_age, usable, year)
    four_years = _four_year_runs({year: found for year, (found, _three, _four) in second.items()})
    conventions = {}
    for year, (convention, two, three) in first.items():
        found, second_three, second_four = second.get(year, (None, None, None))
        if convention == TWO_YEARS and year in four_years:
            convention = TWO_FOUR_YEARS
        conventions[year] = Convention(
            geography,
            year,
            convention,
            _round(two),
            _round(three),
            _round(second_three),
            _round(second_four),
            found,
        )
    return conventions


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def for_stock(
    stock: OfficialStock,
    youngest_by_year: Mapping[int, int],
    registrations: Mapping[int, NewRegistrations],
    second_by_year: Mapping[int, int] | None = None,
) -> Convention | None:
    """The convention of the year the market's bands come from, if it has bands."""

    if stock.band_year is None:
        return None
    return measure(youngest_by_year, registrations, stock.geography, second_by_year).get(
        stock.band_year
    )

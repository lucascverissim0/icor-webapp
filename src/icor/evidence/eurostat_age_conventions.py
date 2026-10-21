"""Which registration years each office counts in Eurostat's youngest age band.

Eurostat's `road_eqs_carage` bands read "less than 2 years", "2 to 5 years" and so
on, but national offices age a car differently. Some count the stock year and
the one before as "less than 2" (a car registered in the stock year or the year
before), others count three registration years, which shifts every later band
edge by one year as well. The convention is not in the data's metadata, so it is
measured: the youngest band is compared with the office's own new registrations
(`road_eqr_carmot`) of the last two and the last three years. Under the right
reading the ratio is near one (a few cars are scrapped or exported, a few young
used imports arrive); under the wrong one it is near 1.5 or 0.67.

A market whose young used imports or exports blur the test is left unclear, and
the report compares it under both readings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from icor.eurocodes.data_coverage import OfficialStock
from icor.evidence.eurostat_new_registrations import NewRegistrations, includes_used_imports

TWO_YEARS = "2y"
THREE_YEARS = "3y"
UNCLEAR = "unclear"

#: The youngest band divided by the matching registrations lies within this range
#: under the right reading.
LOW, HIGH = 0.85, 1.15


@dataclass(frozen=True, slots=True)
class Convention:
    geography: str
    year: int
    convention: str
    #: Youngest band over the last two and last three years' new registrations.
    two_year_ratio: float | None
    three_year_ratio: float | None

    @property
    def shift(self) -> int:
        """Years every band edge moves up: 1 when three registration years are under 2."""

        return 1 if self.convention == THREE_YEARS else 0


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


def measure(
    youngest_by_year: Mapping[int, int],
    registrations: Mapping[int, NewRegistrations],
    geography: str,
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
    conventions = {}
    for year, youngest in sorted(youngest_by_year.items()):
        convention, two, three = classify(youngest, usable, year)
        conventions[year] = Convention(
            geography,
            year,
            convention,
            None if two is None else round(two, 4),
            None if three is None else round(three, 4),
        )
    return conventions


def for_stock(
    stock: OfficialStock,
    youngest_by_year: Mapping[int, int],
    registrations: Mapping[int, NewRegistrations],
) -> Convention | None:
    """The convention of the year the market's bands come from, if it has bands."""

    if stock.band_year is None:
        return None
    return measure(youngest_by_year, registrations, stock.geography).get(stock.band_year)

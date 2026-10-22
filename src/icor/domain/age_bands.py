"""Eurostat's passenger-car age bands and the registration years each office counts in them.

Eurostat's `road_eqs_carage` bands read "less than 2 years", "2 to 5 years" and so
on; ages are whole years at the end of the stock year. National offices fill
them differently, so every band's ages depend on the office's convention (see
`icor.evidence.eurostat_age_conventions`, which measures it).
"""

from __future__ import annotations

from collections.abc import Mapping

#: The bands for an office that counts two registration years as "less than 2 years".
AGE_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("Y_LT2", 0, 1),
    ("Y2-5", 2, 4),
    ("Y5-10", 5, 9),
    ("Y10-20", 10, 19),
    ("Y_GT20", 20, None),
)

TWO_YEARS, THREE_YEARS, TWO_FOUR_YEARS, UNCLEAR = "2y", "3y", "2y-4y", "unclear"

#: The ages each band covers under each convention. "3y" moves every edge up a
#: year. "2y-4y" is Italy's: two registration years under 2, four in "2 to 5"
#: (ages 2-5), then ages 6-9 and 10 and over (its last band is open-ended, so
#: the 20-year edge is never used).
BAND_LAYOUTS: Mapping[str, tuple[tuple[str, int, int | None], ...]] = {
    TWO_YEARS: AGE_BANDS,
    THREE_YEARS: (
        ("Y_LT2", 0, 2),
        ("Y2-5", 3, 5),
        ("Y5-10", 6, 10),
        ("Y10-20", 11, 20),
        ("Y_GT20", 21, None),
    ),
    TWO_FOUR_YEARS: (
        ("Y_LT2", 0, 1),
        ("Y2-5", 2, 5),
        ("Y5-10", 6, 9),
        ("Y10-20", 10, 19),
        ("Y_GT20", 20, None),
    ),
}


def band_layout(convention: str | None) -> tuple[tuple[str, int, int | None], ...]:
    """The bands of a convention; an unclear or unknown one reads as two years."""

    return BAND_LAYOUTS.get(convention or TWO_YEARS, AGE_BANDS)


def band_ages(name: str, convention: str | None) -> range:
    """The ages one bounded band covers under a convention."""

    for band, low, high in band_layout(convention):
        if band == name:
            if high is None:
                raise ValueError(f"band {name} is open-ended")
            return range(low, high + 1)
    raise ValueError(f"unknown band {name}")

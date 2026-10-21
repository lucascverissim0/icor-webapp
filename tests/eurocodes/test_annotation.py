"""A row's Eurocodes: certain units, shared units never split, and every unit accounted for.

All codes and vehicles are fictional (`Q` prefix, which the real catalogue does
not use).
"""

from __future__ import annotations

from decimal import Decimal

from icor.eurocodes.annotation import EurocodeCell, eurocode_breakdown
from icor.eurocodes.catalog import parse_rows
from icor.eurocodes.demo import demo_eurocodes
from icor.eurocodes.runtime import EurocodeMap

SHA = "b" * 64


def _shared_windscreen_map() -> EurocodeMap:
    """One code under two families, a successor without a code, and a pooled label."""

    catalogue = parse_rows(
        (
            ("Q501", "ZEPHYR", "GALE/GUST 2018-", None, 2018),
            ("Q502", "ZEPHYR", "GUST CABRIO 2019-", None, 2019),
        ),
        make_aliases={},
        source_sha256=SHA,
    )
    document = {
        "format": "icor-eurocode-decisions",
        "version": 1,
        "metadata": {"catalogue_sha256": SHA},
        "families": {
            "zephyr/gale": {
                "label": "Zephyr Gale",
                "make": "zephyr",
                "codes": [
                    {
                        "code": "Q501",
                        "label": "Gale",
                        "first_year": 2018,
                        "uncoded_successor": 2024,
                        "rows": [2],
                    }
                ],
            },
            "zephyr/gust": {
                "label": "Zephyr Gust",
                "make": "zephyr",
                "codes": [
                    {"code": "Q501", "label": "Gust", "first_year": 2018, "rows": [2]},
                    {
                        "code": "Q502",
                        "label": "Gust Cabrio",
                        "first_year": 2019,
                        "bodies": ["convertible"],
                        "role": "minor",
                        "rows": [3],
                    },
                ],
            },
        },
        "identities": {
            "zephyr|gale": {"family": "zephyr/gale"},
            "zephyr|gust": {
                "family": "zephyr/gust",
                "variants": [
                    {"match": "cabrio", "family": "zephyr/gust", "body": "convertible"},
                    {"match": "bundle", "family": ["zephyr/gust", "zephyr/gale"]},
                ],
            },
            "zephyr|any": {"status": "ambiguous_identity"},
        },
    }
    return EurocodeMap.from_documents(catalogue, document)


def _cell(identity: str, year: int, units: int, raw_model: str = "", geography: str = "DE"):
    return EurocodeCell(identity, year, geography, raw_model or identity.split("|")[-1], units)


def test_a_launch_span_is_shared_never_split_and_both_codes_say_so() -> None:
    breakdown = eurocode_breakdown(
        demo_eurocodes(),
        (
            _cell("aurora mobility|a1 horizon", 2024, 100),
            _cell("aurora mobility|a1 horizon", 2025, 40),
            _cell("aurora mobility|a1 horizon", 2027, 7),
        ),
    )

    first, second = breakdown.codes
    assert (first.code, first.certain_units, first.shared_units) == ("Q201", 100, 40)
    assert first.shared_with == ("Q202",)
    assert (second.code, second.certain_units, second.shared_units) == ("Q202", 7, 40)
    assert second.shared_with == ("Q201",)
    assert (breakdown.single_units, breakdown.shared_units) == (107, 40)
    assert breakdown.total_units == 147


def test_unassigned_units_are_reported_by_reason() -> None:
    breakdown = eurocode_breakdown(
        demo_eurocodes(),
        (
            _cell("velora works|v3 current", 2023, 5),
            _cell("velora works|v3 current", 2024, 9),
            _cell("northstar automotive|n7 atlas", 2027, 11),
        ),
    )

    assert [share.code for share in breakdown.codes] == ["Q401"]
    assert (breakdown.single_units, breakdown.outside_units, breakdown.unreviewed_units) == (
        9,
        5,
        11,
    )
    assert breakdown.total_units == 25


def test_an_ambiguous_identity_is_unassignable_and_names_no_code() -> None:
    breakdown = eurocode_breakdown(_shared_windscreen_map(), (_cell("zephyr|any", 2020, 3),))

    assert breakdown.codes == ()
    assert breakdown.unassignable_units == 3


def test_a_shared_windscreen_reads_with_the_routed_familys_label_and_years() -> None:
    eurocodes = _shared_windscreen_map()

    gale = eurocode_breakdown(eurocodes, (_cell("zephyr|gale", 2020, 4),)).codes[0]
    gust = eurocode_breakdown(eurocodes, (_cell("zephyr|gust", 2020, 4),)).codes[0]

    assert (gale.code, gale.label, gale.last_year) == ("Q501", "Zephyr Gale", 2024)
    assert (gust.code, gust.label, gust.last_year) == ("Q501", "Zephyr Gust", None)


def test_a_successor_without_a_code_is_flagged_on_the_shared_units() -> None:
    breakdown = eurocode_breakdown(
        _shared_windscreen_map(),
        (_cell("zephyr|gale", 2023, 6), _cell("zephyr|gale", 2024, 2)),
    )

    (gale,) = breakdown.codes
    assert (gale.certain_units, gale.shared_units, gale.shared_with) == (6, 2, ())
    assert gale.shared_with_uncoded
    assert (breakdown.single_units, breakdown.shared_units) == (6, 2)


def test_each_cell_is_allocated_with_its_own_spelling() -> None:
    breakdown = eurocode_breakdown(
        _shared_windscreen_map(),
        (
            _cell("zephyr|gust", 2020, 10, raw_model="gust"),
            _cell("zephyr|gust", 2020, 3, raw_model="gust cabrio"),
        ),
    )

    assert {share.code: share.certain_units for share in breakdown.codes} == {
        "Q501": 10,
        "Q502": 3,
    }


def test_cells_without_units_add_no_code() -> None:
    breakdown = eurocode_breakdown(
        demo_eurocodes(), (_cell("meridian motors|m5 vector", 2025, 0),)
    )

    assert breakdown.codes == ()
    assert breakdown.total_units == 0


def test_fractional_registrations_round_once_and_still_reconcile() -> None:
    breakdown = eurocode_breakdown(
        demo_eurocodes(),
        (
            _cell("aurora mobility|a1 horizon", 2024, Decimal("0.6")),
            _cell("aurora mobility|a1 horizon", 2027, Decimal("0.6")),
            _cell("aurora mobility|a1 horizon", 2025, Decimal("2.4")),
            _cell("northstar automotive|n7 atlas", 2027, Decimal("0.3")),
        ),
    )

    assert breakdown.total_units == 4  # 3.9 rounded once, not 1 + 1 + 2 + 0
    assert (breakdown.single_units, breakdown.shared_units, breakdown.unreviewed_units) == (
        1,
        3,
        0,
    )
    assert sum(share.certain_units for share in breakdown.codes) == breakdown.single_units
    assert all(share.shared_units == 2 for share in breakdown.codes)

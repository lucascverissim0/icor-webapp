"""Parsing the Eurocode catalogue: every row parsed or rejected, nothing guessed.

The real catalogue is ICOR's proprietary data and never enters the repository.
Every row here is fictional: codes use the `Q` prefix, which the real catalogue
does not use, and the models are invented.
"""

from __future__ import annotations

import json

import pytest

from icor.eurocodes.catalog import (
    from_document,
    parse_brand,
    parse_rows,
    to_document,
)

ALIASES = {"new zephyr": "zephyr"}


def _parse(*rows: tuple[object, ...]):
    return parse_rows(rows, make_aliases=ALIASES, source_sha256="0" * 64)


def _one(*row: object):
    catalogue = _parse(tuple(row))
    assert not catalogue.rejected, catalogue.rejected
    (parsed,) = catalogue.rows
    return parsed


def test_every_row_is_parsed_or_rejected_with_a_reason() -> None:
    catalogue = _parse(
        ("Q001", "ZEPHYR", "ARROW 5D HBK 2012-", "0", 2012),
        (None, None, None, None, 0),
        (None, "ZEPHYR", "ARROW", "0", 2012),
        ("Q02", "ZEPHYR", "ARROW", "0", 2012),
        ("Q003 X", "ZEPHYR", "ARROW", "0", 2012),
        ("Q004", "ZEPHYR", None, "0", 2012),
        ("Q005", "  ", "ARROW", "0", 2012),
    )

    assert catalogue.source_rows == 7
    assert len(catalogue.rows) + len(catalogue.rejected) == 7
    assert [row.code for row in catalogue.rows] == ["Q001"]
    assert {row.row: row.reason for row in catalogue.rejected} == {
        3: "blank_row",
        4: "missing_code",
        5: "code_not_four_characters",
        6: "code_not_four_characters",
        7: "missing_model",
        8: "missing_brand",
    }


def test_brand_qualifiers_become_scope_and_rebadges_yield_both_makes() -> None:
    assert parse_brand("ZEPHYR  ", ALIASES) == (("zephyr",), None)
    assert parse_brand("ZEPHYR AUTO'S & VANS", ALIASES) == (("zephyr",), None)
    assert parse_brand("ZEPHYR VRACHTWAGEN", ALIASES) == (("zephyr",), "truck")
    assert parse_brand("ZEPHYR BUS", ALIASES) == (("zephyr",), "bus")
    assert parse_brand("ZEPHYR USA", ALIASES) == (("zephyr",), "non_european")
    assert parse_brand("NEW ZEPHYR", ALIASES) == (("zephyr",), None)
    assert parse_brand("ZEPHYR/BOREAS", ALIASES) == (("zephyr", "boreas"), None)
    # An alias applies to the folded spelling too, so punctuation cannot defeat it.
    assert parse_brand("NEW-ZEPHYR", ALIASES) == (("zephyr",), None)
    # The shared make rules still apply: VW folds to Volkswagen.
    assert parse_brand("VW", {}) == (("volkswagen",), None)


def test_truck_and_north_american_rows_keep_their_scope() -> None:
    assert _one("Q010", "ZEPHYR VRACHTWAGEN", "HAULER 2010-", "0", 2010).scope == "truck"
    assert (
        _one("Q011", "ZEPHYR", "ARROW (NORTH AMERICAN VERSION) 2010-", "0", 2010).scope
        == "non_european"
    )
    assert (
        _one("Q012", "ZEPHYR", "ARROW (JAPANESE MARKET) 2010-", "0", 2010).scope == "non_european"
    )
    assert _one("Q013", "ZEPHYR", "ARROW 5D SUV (EU) 2014-", "0", 2014).scope == "passenger"
    assert (
        _one("Q015", "ZEPHYR", "ARROW BRK 09/99-07/2000 VOOR AUSTRALIE", "2000", 1999).scope
        == "non_european"
    )


def test_production_origin_is_a_flag_not_a_foreign_market() -> None:
    row = _one("Q014", "ZEPHYR", "ARROW JAPANESE PRODUCTION 2005-", "0", 2005)

    assert row.scope == "passenger"
    assert "production_origin" in row.flags


def test_drive_side_and_body_words_are_read_but_not_inferred() -> None:
    rhd = _one("Q020", "ZEPHYR", "ARROW II 5D HBK RHD 2015-", "0", 2015)
    plain = _one("Q021", "ZEPHYR", "ARROW II 2015-", "0", 2015)
    both = _one("Q022", "ZEPHYR", "ARROW LHD/RHD 2015-", "0", 2015)

    assert (rhd.drive, rhd.bodies) == ("RHD", ("hatchback",))
    assert (plain.drive, plain.bodies) == (None, ())
    assert both.drive is None and "both_drive_sides" in both.flags


def test_dutch_van_meaning_from_is_not_a_van_body() -> None:
    part_number = _one("Q029", "ZEPHYR", "ARROW 5D HBK VAN DAM 1234567 2011-", "0", 2011)
    dated = _one("Q030", "ZEPHYR", "ARROW 3D/5D HBK VAN 10/2009-", "0", 2009)
    van = _one("Q031", "ZEPHYR", "ARROW HBK+3D VAN 10/2001-", "0", 2001)
    commercial = _one("Q034", "ZEPHYR", "ARROW HBK VAN CONVERSION", "0", 2001)

    assert part_number.bodies == ("hatchback",)
    # Either "from 10/2009" or a van: not settled, so flagged rather than read.
    assert dated.bodies == ("hatchback",)
    assert "van_word_ambiguous" in dated.flags
    assert van.bodies == ("hatchback", "van")
    assert commercial.bodies == ("hatchback", "van")


def test_a_pickup_cabin_is_not_a_convertible() -> None:
    assert _one("Q032", "ZEPHYR", "HAULER DOUBLE CAB PU 2016-", "0", 2016).bodies == ("pickup",)
    assert _one("Q033", "ZEPHYR", "ARROW CAB 2016-", "0", 2016).bodies == ("convertible",)


def test_text_years_and_columns_are_both_read_and_the_earliest_start_wins() -> None:
    # The start column states the later facelift; the text states the launch.
    row = _one("Q040", "ZEPHYR", "ARROW 3D/5D HBK 2012 - 5D EST 2013-", "0", 2013)

    assert [(m.year, m.role) for m in row.years] == [(2012, "start"), (2013, "start")]
    assert (row.start, row.end, row.sop, row.eop) == (2012, None, 2013, None)


def test_closed_ranges_two_digit_years_and_dutch_until() -> None:
    closed = _one("Q041", "ZEPHYR", "ARROW 5D BRK2007-2013", "2013", 2007)
    old = _one("Q042", "ZEPHYR", "ARROW 3D 74-83", "1983", 1974)
    until = _one("Q043", "ZEPHYR", "ARROW 3D HBK TOT 10/2009-", "2009", 2009)
    month = _one("Q044", "ZEPHYR", "ARROW 5D 04/99-", "0", 1999)

    assert (closed.start, closed.end) == (2007, 2013)
    assert (old.start, old.end) == (1974, 1983)
    assert [(m.year, m.role) for m in until.years] == [(2009, "end")]
    assert (until.start, until.end) == (2009, 2009)
    assert (month.start, month.end) == (1999, None)


def test_a_leading_four_digit_model_name_is_not_a_year() -> None:
    lead = _one("Q050", "ZEPHYR", "2030 5D SUV 2019-", "0", 2019)
    after_slash = _one("Q051", "ZEPHYR", "ARROW 3D/5D HBK 2011/2030 5D SUV 2013-", "0", 2011)
    year_before_slash = _one("Q052", "ZEPHYR", "ARROW 5D HBK 2008/BOLT 5D SUV 2010-", "0", 2008)

    assert [m.year for m in lead.years] == [2019]
    assert lead.start == 2019
    assert [m.year for m in after_slash.years] == [2011, 2013]
    assert [m.year for m in year_before_slash.years] == [2008, 2010]


def test_a_year_after_a_month_or_day_is_a_date_not_a_model() -> None:
    month = _one("Q053", "ZEPHYR", "ARROW SAL 04/2007 & 01/2011-", "0", 2007)
    day = _one("Q054", "ZEPHYR", "ARROW LHD 27/07/2017 TOT 18/09/2017-", "2017", 2017)

    assert [(m.year, m.role) for m in month.years] == [(2007, "start"), (2011, "start")]
    assert [(m.year, m.role) for m in day.years] == [(2017, "start"), (2017, "end")]
    assert "leading_number_read_as_model" not in month.flags + day.flags


def test_mixed_ranges_and_month_dash_dates() -> None:
    mixed = _one("Q057", "ZEPHYR", "ARROW SAL 97-2004", "0", 1997)
    month_end = _one("Q058", "ZEPHYR", "ARROW 95-03/2000", "0", 1995)
    month_dash = _one("Q059", "ZEPHYR", "ARROW 5D SUV RHD 05-2022-", "0", 2022)
    early_range = _one("Q05A", "ZEPHYR", "ARROW 05-2012", "0", 2005)

    assert [(m.year, m.role) for m in mixed.years] == [(1997, "start"), (2004, "end")]
    assert [(m.year, m.role) for m in month_end.years] == [(1995, "start"), (2000, "end")]
    assert [(m.year, m.role) for m in month_dash.years] == [(2022, "start")]
    assert [(m.year, m.role) for m in early_range.years] == [(2005, "start"), (2012, "end")]


def test_engine_and_chassis_numbers_are_not_years() -> None:
    engines = _one("Q05B", "ZEPHYR", "ARROW 1400-1990 SAL 62-75", "1975", 1962)
    listed = _one("Q05C", "ZEPHYR", "1950,2700 BRK 63-77", "1977", 1963)
    chassis = _one("Q05D", "ZEPHYR", "ARROW VANAF CH 88-K-31000", "0", 1991)
    suffix = _one("Q05E", "ZEPHYR", "ARROW 18-27M SAL 66-72", "1972", 1966)

    assert [(m.year, m.role) for m in engines.years] == [(1962, "start"), (1975, "end")]
    assert [(m.year, m.role) for m in listed.years] == [(1963, "start"), (1977, "end")]
    assert chassis.years == ()
    assert [(m.year, m.role) for m in suffix.years] == [(1966, "start"), (1972, "end")]


def test_numbers_glued_to_letters_or_after_the_end_are_not_years() -> None:
    glued_model = _one("Q05F", "ZEPHYR", "K1990 VAN+PU 84-97", "1997", 1984)
    glued_year = _one("Q05G", "ZEPHYR", "ARROW 5D BRK2007-", "0", 2007)
    engine = _one("Q05H", "ZEPHYR", "ARROW 2000 SAL 78-84", "1984", 1978)
    until = _one("Q05J", "ZEPHYR", "ARROW SAL UNTIL 2003-", "0", 1995)

    assert [(m.year, m.role) for m in glued_model.years] == [(1984, "start"), (1997, "end")]
    assert [(m.year, m.role) for m in glued_year.years] == [(2007, "start")]
    assert [(m.year, m.role) for m in engine.years] == [(1978, "start"), (1984, "end")]
    assert "start_after_end_dropped" in engine.flags
    assert [(m.year, m.role) for m in until.years] == [(2003, "end")]


def test_pre_war_and_two_digit_years_expand_into_the_right_century() -> None:
    prewar = _one("Q055", "ZEPHYR", "ARROW 4D SAL 1947-1961", "1961", 1947)
    fifties = _one("Q056", "ZEPHYR", "ARROW 49-53", "1953", 1949)

    assert (prewar.start, prewar.end) == (1947, 1961)
    assert (fifties.start, fifties.end) == (1949, 1953)


def test_impossible_years_are_flagged_and_never_used() -> None:
    negative = _one("Q060", "ZEPHYR", "ARROW", "0", -2022)
    zero = _one("Q061", "ZEPHYR", "HAULER", "0", 0)
    reversed_window = _one("Q062", "ZEPHYR", "ARROW 2015-", "2010", 2015)

    assert negative.sop is None and "sop_year_out_of_range" in negative.flags
    assert "no_start_year" in negative.flags
    assert zero.sop is None and "sop_missing" in zero.flags
    assert reversed_window.eop is None and "eop_before_sop" in reversed_window.flags
    assert (reversed_window.start, reversed_window.end) == (2015, None)


def test_codes_group_rows_and_whitespace_is_normalised() -> None:
    catalogue = _parse(
        ("Q070", "ZEPHYR ", "ARROW  II 2015-", "0", 2015),
        (" Q070", "ZEPHYR", "ARROW II FACELIFT 2018-", "0", 2018),
    )

    ((code, rows),) = catalogue.codes().items()
    assert code == "Q070"
    assert [row.description for row in rows] == ["ARROW II 2015-", "ARROW II FACELIFT 2018-"]
    assert {row.brand for row in rows} == {"ZEPHYR"}


def test_the_document_round_trips_byte_for_byte_and_checks_its_accounting() -> None:
    catalogue = _parse(
        ("Q080", "ZEPHYR/BOREAS", "ARROW 5D HBK RHD 2012-", "0", 2012),
        (None, None, None, None, None),
    )
    document = to_document(catalogue)
    encoded = json.dumps(document, sort_keys=True)

    restored = from_document(json.loads(encoded))

    assert restored == catalogue
    assert json.dumps(to_document(restored), sort_keys=True) == encoded
    document["rejected"] = []
    with pytest.raises(ValueError, match="every source row"):
        from_document(document)

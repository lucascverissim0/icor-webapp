"""Attributing registrations to Eurocodes: computed from reviewed facts, never split.

All codes and vehicles are fictional (`Q` prefix, which the real catalogue does
not use).
"""

from __future__ import annotations

import pytest

from icor.eurocodes.allocation import (
    Allocation,
    EurocodeDecisions,
    body_hint,
    drive_side,
)


def _document(**families: dict) -> dict:
    return {
        "format": "icor-eurocode-decisions",
        "version": 1,
        "families": families,
        "identities": {},
    }


ARROW = {
    "label": "Zephyr Arrow",
    "make": "zephyr",
    "codes": [
        # Generation I, built until its successor took over.
        {"code": "Q101", "first_year": 2005, "bodies": ["hatchback", "estate"], "rows": [1]},
        # Generation II: the catalogue states 2012 for the hatch and 2013 for a facelift.
        {
            "code": "Q102",
            "first_year": 2012,
            "first_year_latest": 2013,
            "bodies": ["hatchback"],
            "replaces": ["Q101"],
            "rows": [2],
        },
        # Generation II estate kept a separate windscreen, built 2013-2016.
        {
            "code": "Q103",
            "first_year": 2013,
            "last_year": 2016,
            "bodies": ["estate"],
            "replaces": ["Q101"],
            "rows": [3],
        },
        # A cabriolet with its own code: niche, only for registrations naming it.
        {
            "code": "Q104",
            "first_year": 2014,
            "last_year": 2018,
            "bodies": ["convertible"],
            "role": "minor",
            "rows": [4],
        },
        # Generation III, one windscreen for every body.
        {
            "code": "Q105",
            "first_year": 2020,
            "bodies": ["all"],
            "replaces": ["Q102", "Q103"],
            "rows": [5],
        },
    ],
}


def _decisions(identities: dict | None = None, **families: dict) -> EurocodeDecisions:
    document = _document(**(families or {"zephyr/arrow": ARROW}))
    document["identities"] = identities or {
        "zephyr|arrow": {"family": "zephyr/arrow"},
        "zephyr|arrow estate": {"family": "zephyr/arrow", "body": "estate"},
        "zephyr|amg": {"status": "ambiguous_identity", "why": "names several models"},
    }
    return EurocodeDecisions.from_document(document)


def _codes(decisions: EurocodeDecisions, year: int, *, geography: str = "NL", body=None):
    return decisions.allocate("zephyr|arrow", year, geography, body).codes


def test_a_launch_span_is_shared_with_the_code_it_replaces() -> None:
    decisions = _decisions()

    assert _codes(decisions, 2011) == ("Q101",)
    # Both 2012 and 2013 may hold run-out generation I cars: never split.
    assert decisions.allocate("zephyr|arrow", 2012, "NL") == Allocation(
        ("Q101", "Q102"), "shared", "zephyr/arrow"
    )
    assert _codes(decisions, 2013) == ("Q101", "Q102", "Q103")


def test_a_main_body_with_its_own_code_is_shared_when_the_body_is_unknown() -> None:
    decisions = _decisions()

    # Estate and hatch carry different codes; an unlabelled registration may be either.
    assert _codes(decisions, 2015) == ("Q102", "Q103")
    # The estate's explicit last year 2016 leaves one run-out year.
    assert _codes(decisions, 2017) == ("Q102", "Q103")
    assert _codes(decisions, 2018) == ("Q102",)


def test_a_named_body_selects_its_own_code() -> None:
    decisions = _decisions()

    assert decisions.allocate("zephyr|arrow estate", 2015, "NL").codes == ("Q103",)
    assert _codes(decisions, 2015, body="estate") == ("Q103",)
    assert _codes(decisions, 2015, body="hatchback") == ("Q102",)


def test_a_niche_body_gets_demand_only_from_registrations_that_name_it() -> None:
    decisions = _decisions()

    assert "Q104" not in _codes(decisions, 2016)
    assert _codes(decisions, 2016, body="convertible") == ("Q104",)
    # Outside the niche code's years a named body falls back to the main codes.
    assert _codes(decisions, 2021, body="convertible") == ("Q105",)


def test_a_body_whose_own_line_has_no_code_that_year_gets_none() -> None:
    decisions = _bolt(
        {"code": "Q501", "first_year": 2010, "bodies": ["coupe"], "rows": [1]},
        {
            "code": "Q502",
            "first_year": 2010,
            "last_year": 2015,
            "bodies": ["convertible"],
            "role": "minor",
            "rows": [2],
        },
        {
            "code": "Q503",
            "first_year": 2020,
            "bodies": ["coupe"],
            "replaces": ["Q501"],
            "rows": [3],
        },
    )

    # The next cabriolet has no catalogue code: never lend it the coupe's.
    assert _bolt_codes(decisions, 2014, "convertible") == ("Q502",)
    assert _bolt_codes(decisions, 2022, "convertible") == ()
    assert decisions.allocate("zephyr|bolt", 2022, "NL", "convertible").status == "outside"
    # A body this family never distinguishes still reads as the main codes.
    assert _bolt_codes(decisions, 2022, "estate") == ("Q503",)


def test_a_code_for_every_body_takes_over_and_runs_open_ended() -> None:
    decisions = _decisions()

    assert _codes(decisions, 2020) == ("Q102", "Q105")
    assert _codes(decisions, 2021) == ("Q105",)
    assert _codes(decisions, 2031) == ("Q105",)
    assert _codes(decisions, 2021, body="estate") == ("Q105",)


def _bolt(*codes: dict) -> EurocodeDecisions:
    family = {"label": "Zephyr Bolt", "make": "zephyr", "codes": list(codes)}
    return _decisions({"zephyr|bolt": {"family": "zephyr/bolt"}}, **{"zephyr/bolt": family})


def _bolt_codes(decisions: EurocodeDecisions, year: int, body: str | None = None):
    return decisions.allocate("zephyr|bolt", year, "NL", body).codes


@pytest.mark.parametrize("last_year", [None, 2011, 2012])
def test_the_run_out_does_not_depend_on_whether_the_end_is_stated(last_year) -> None:
    decisions = _bolt(
        {"code": "Q501", "first_year": 2005, "last_year": last_year, "rows": [1]},
        {"code": "Q502", "first_year": 2012, "replaces": ["Q501"], "rows": [2]},
    )

    assert _bolt_codes(decisions, 2011) == ("Q501",)
    assert _bolt_codes(decisions, 2012) == ("Q501", "Q502")
    assert _bolt_codes(decisions, 2013) == ("Q502",)


def test_production_beyond_the_successor_launch_keeps_its_run_out_year() -> None:
    decisions = _bolt(
        {"code": "Q501", "first_year": 2005, "last_year": 2014, "rows": [1]},
        {"code": "Q502", "first_year": 2012, "replaces": ["Q501"], "rows": [2]},
    )

    assert _bolt_codes(decisions, 2015) == ("Q501", "Q502")
    assert _bolt_codes(decisions, 2016) == ("Q502",)


def test_each_body_ends_when_its_own_successor_takes_over() -> None:
    decisions = _bolt(
        {"code": "Q501", "first_year": 2005, "bodies": ["hatchback", "estate"], "rows": [1]},
        {
            "code": "Q502",
            "first_year": 2012,
            "bodies": ["hatchback"],
            "replaces": ["Q501"],
            "rows": [2],
        },
        {
            "code": "Q503",
            "first_year": 2015,
            "bodies": ["estate"],
            "replaces": ["Q501"],
            "rows": [3],
        },
    )

    assert _bolt_codes(decisions, 2014, "hatchback") == ("Q502",)
    assert _bolt_codes(decisions, 2014, "estate") == ("Q501",)
    assert _bolt_codes(decisions, 2014) == ("Q501", "Q502")
    assert _bolt_codes(decisions, 2016) == ("Q502", "Q503")


def test_a_body_dropped_when_the_line_was_replaced_has_no_extra_run_out_year() -> None:
    decisions = _bolt(
        {
            "code": "Q501",
            "first_year": 2012,
            "last_year": 2019,
            "bodies": ["hatchback", "estate"],
            "rows": [1],
        },
        {
            "code": "Q502",
            "first_year": 2019,
            "bodies": ["hatchback"],
            "replaces": ["Q501"],
            "rows": [2],
        },
    )

    # The estate ended with the line; its stock is no likelier than the hatch's.
    assert _bolt_codes(decisions, 2019) == ("Q501", "Q502")
    assert _bolt_codes(decisions, 2020) == ("Q502",)
    assert _bolt_codes(decisions, 2019, "estate") == ("Q501",)


def test_a_body_built_on_after_its_line_was_replaced_keeps_its_run_out() -> None:
    decisions = _bolt(
        {
            "code": "Q501",
            "first_year": 2012,
            "last_year": 2021,
            "bodies": ["hatchback", "estate"],
            "rows": [1],
        },
        {
            "code": "Q502",
            "first_year": 2019,
            "bodies": ["hatchback"],
            "replaces": ["Q501"],
            "rows": [2],
        },
    )

    assert _bolt_codes(decisions, 2022) == ("Q501", "Q502")
    # The last year is the code's, not one body's: a hatch may have been built on too.
    assert _bolt_codes(decisions, 2022, "hatchback") == ("Q501", "Q502")
    assert _bolt_codes(decisions, 2023) == ("Q502",)


def test_a_code_replaced_for_some_bodies_must_end_for_the_others() -> None:
    with pytest.raises(ValueError, match=r"replaced but not for \['estate'\]"):
        _bolt(
            {"code": "Q501", "first_year": 2005, "bodies": ["hatchback", "estate"], "rows": [1]},
            {
                "code": "Q502",
                "first_year": 2012,
                "bodies": ["hatchback"],
                "replaces": ["Q501"],
                "rows": [2],
            },
        )


def test_years_before_the_first_code_are_outside_not_guessed() -> None:
    decisions = _decisions()

    assert decisions.allocate("zephyr|arrow", 2003, "NL") == Allocation(
        (), "outside", "zephyr/arrow"
    )


def test_unreviewed_and_unassignable_identities_are_reported_as_such() -> None:
    decisions = _decisions()

    assert decisions.allocate("zephyr|unknown", 2015, "NL").status == "unreviewed"
    assert decisions.allocate("zephyr|amg", 2015, "NL") == Allocation((), "unassignable")


def test_right_hand_drive_codes_apply_only_in_right_hand_drive_markets() -> None:
    bolt = {
        "label": "Zephyr Bolt",
        "make": "zephyr",
        "codes": [
            {"code": "Q201", "first_year": 2010, "drive": "LHD", "rows": [1]},
            {"code": "Q202", "first_year": 2010, "drive": "RHD", "rows": [2]},
            {"code": "Q203", "first_year": 2018, "replaces": ["Q201", "Q202"], "rows": [3]},
        ],
    }
    decisions = _decisions({"zephyr|bolt": {"family": "zephyr/bolt"}}, **{"zephyr/bolt": bolt})

    assert drive_side("GB") == drive_side("ie") == "RHD"
    assert drive_side("NL") == "LHD"
    assert decisions.allocate("zephyr|bolt", 2014, "DE").codes == ("Q201",)
    assert decisions.allocate("zephyr|bolt", 2014, "GB").codes == ("Q202",)
    assert decisions.allocate("zephyr|bolt", 2018, "MT").codes == ("Q202", "Q203")


def test_a_shared_windscreen_belongs_to_each_family_independently() -> None:
    shared = {"code": "Q301", "first_year": 2012, "rows": [1]}
    decisions = _decisions(
        {"zephyr|alpha": {"family": "zephyr/alpha"}, "zephyr|beta": {"family": "zephyr/beta"}},
        **{
            "zephyr/alpha": {"label": "Alpha", "make": "zephyr", "codes": [shared]},
            "zephyr/beta": {"label": "Beta", "make": "zephyr", "codes": [shared]},
        },
    )

    assert decisions.allocate("zephyr|alpha", 2015, "NL").codes == ("Q301",)
    assert decisions.allocate("zephyr|beta", 2015, "NL").codes == ("Q301",)


def test_validation_refuses_invented_codes_and_foreign_citations() -> None:
    catalogue = {"Q101": [1], "Q102": [2], "Q103": [3], "Q104": [4], "Q105": [5]}
    document = _document(**{"zephyr/arrow": ARROW})
    EurocodeDecisions.from_document(document, catalogue_codes=catalogue)

    without_q104 = {key: rows for key, rows in catalogue.items() if key != "Q104"}
    with pytest.raises(ValueError, match="not in the catalogue"):
        EurocodeDecisions.from_document(document, catalogue_codes=without_q104)
    uncited = _document(
        **{"zephyr/bolt": {"label": "Bolt", "codes": [{"code": "Q101", "first_year": 2005}]}}
    )
    with pytest.raises(ValueError, match="cites no catalogue row"):
        EurocodeDecisions.from_document(uncited, catalogue_codes=catalogue)
    with pytest.raises(ValueError, match="cites rows of another code"):
        EurocodeDecisions.from_document(document, catalogue_codes={**catalogue, "Q101": [9]})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"first_year_latest": 2001}, "first_year_latest precedes"),
        ({"last_year": 2001}, "last_year precedes"),
        ({"bodies": ["limo"]}, "unknown body"),
        ({"role": "spare"}, "unknown role"),
        ({"drive": "left"}, "unknown drive"),
        ({"replaces": ["Q999"]}, "outside the family"),
    ],
)
def test_validation_refuses_inconsistent_facts(change: dict, message: str) -> None:
    family = {
        "label": "Bolt",
        "make": "zephyr",
        "codes": [{"code": "Q401", "first_year": 2005, **change}],
    }

    with pytest.raises(ValueError, match=message):
        EurocodeDecisions.from_document(_document(**{"zephyr/bolt": family}))


def test_identity_decisions_must_be_coherent() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        _decisions({"zephyr|arrow": {"family": "zephyr/missing"}})
    with pytest.raises(ValueError, match="mapped without a family"):
        _decisions({"zephyr|arrow": {"status": "mapped"}})
    with pytest.raises(ValueError, match="only a mapped identity"):
        _decisions({"zephyr|arrow": {"status": "not_a_vehicle", "family": "zephyr/arrow"}})


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("golf variant 2.0 tdi", "estate"),
        ("308 s.w. hdi", "estate"),
        ("a4 allroad quattro", "estate"),
        ("astra sports tourer", "estate"),
        ("golf cabriolet", "convertible"),
        ("c 220 coupe", "coupe"),
        ("a3 limousine", "saloon"),
        ("focus st-line", None),
        ("golf se navigation tdi", None),
        ("cabrio estate", None),
        (None, None),
    ],
)
def test_body_hints_read_only_unambiguous_body_words(label: str | None, body: str | None) -> None:
    assert body_hint(label) == body

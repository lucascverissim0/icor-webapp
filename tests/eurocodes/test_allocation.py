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


@pytest.mark.parametrize("last_year", [None, 2026])
def test_a_successor_the_catalogue_lacks_shares_its_launch_span_then_ends_the_code(
    last_year,
) -> None:
    # The successor has no code. In its launch years a registration may be it or
    # the last stock of Q501: Q501 stays possible, flagged as shared with a vehicle
    # the catalogue lacks. Once the successor certainly took over, Q501 explains
    # nothing.
    decisions = _bolt(
        {
            "code": "Q501",
            "first_year": 2005,
            "last_year": last_year,
            "uncoded_successor": 2026,
            "uncoded_successor_latest": 2027,
            "rows": [1],
        },
    )

    assert decisions.allocate("zephyr|bolt", 2025, "NL") == Allocation(
        ("Q501",), "single", "zephyr/bolt"
    )
    for year in (2026, 2027):
        assert decisions.allocate("zephyr|bolt", year, "NL") == Allocation(
            ("Q501",), "shared", "zephyr/bolt", uncoded=True
        )
    assert decisions.allocate("zephyr|bolt", 2028, "NL") == Allocation((), "outside", "zephyr/bolt")
    assert decisions.timeline("zephyr/bolt").spans()["Q501"] == (2005, 2027)


def test_an_uncoded_successor_launch_span_defaults_to_its_first_year() -> None:
    decisions = _bolt({"code": "Q501", "first_year": 2005, "uncoded_successor": 2026, "rows": [1]})

    assert decisions.allocate("zephyr|bolt", 2026, "NL").uncoded
    assert decisions.allocate("zephyr|bolt", 2027, "NL").status == "outside"


def test_a_sibling_the_catalogue_lacks_keeps_the_code_but_flags_its_years() -> None:
    # An electric sibling sold alongside under the same name: the code stays,
    # but from the sibling's launch a registration may be either.
    decisions = _bolt({"code": "Q501", "first_year": 2005, "uncoded_sibling": 2026, "rows": [1]})

    assert decisions.allocate("zephyr|bolt", 2025, "NL").status == "single"
    assert decisions.allocate("zephyr|bolt", 2030, "NL") == Allocation(
        ("Q501",), "shared", "zephyr/bolt", uncoded=True
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


def test_a_generation_covering_every_body_stays_shared_with_its_body_specific_successor() -> None:
    decisions = _decisions(
        {"zephyr|swift": {"family": "zephyr/swift"}},
        **{
            "zephyr/swift": {
                "label": "Swift",
                "make": "zephyr",
                "codes": [
                    {"code": "Q701", "first_year": 2007, "rows": [7]},
                    {
                        "code": "Q702",
                        "first_year": 2014,
                        "first_year_latest": 2015,
                        "bodies": ["hatchback", "estate"],
                        "replaces": ["Q701"],
                        "rows": [8],
                    },
                    {
                        "code": "Q703",
                        "first_year": 2016,
                        "bodies": ["convertible"],
                        "role": "minor",
                        "rows": [9],
                    },
                ],
            }
        },
    )

    # The estate's previous generation covered every body: the launch span is shared.
    assert decisions.allocate("zephyr|swift", 2015, "NL", "estate").codes == ("Q701", "Q702")
    assert decisions.allocate("zephyr|swift", 2016, "NL", "estate").codes == ("Q702",)
    # A parallel body line does not borrow the all-body code.
    assert decisions.allocate("zephyr|swift", 2016, "NL", "convertible").codes == ("Q703",)


def test_variants_route_lumped_spellings_to_their_own_family() -> None:
    families = {
        "zephyr/arrow": ARROW,
        "zephyr/volt": {
            "label": "Zephyr Arrow Volt",
            "make": "zephyr",
            "codes": [{"code": "Q601", "first_year": 2020, "rows": [6]}],
        },
    }
    identities = {
        "zephyr|arrow": {
            "family": "zephyr/arrow",
            "variants": [
                {"match": r"\bvolt\b", "family": "zephyr/volt"},
                {"match": r"\bx import\b", "status": "not_in_catalogue"},
                {"match": r"^arrow$", "geographies": ["GB"], "family": "zephyr/volt"},
            ],
        },
        "zephyr|grand": {
            "status": "ambiguous_identity",
            "variants": [{"match": r"grand arrow", "family": "zephyr/arrow", "body": "estate"}],
        },
    }
    decisions = _decisions(identities, **families)

    assert decisions.allocate("zephyr|arrow", 2021, "NL", raw_model="ARROW VOLT 80kWh").codes == (
        "Q601",
    )
    assert decisions.allocate("zephyr|arrow", 2021, "NL", raw_model="arrow 1.5").codes == ("Q105",)
    # A variant rescues the spellings it names from an ambiguous identity; the rest stay out.
    assert decisions.allocate("zephyr|grand", 2015, "NL", raw_model="Grand Arrow").codes == (
        "Q103",
    )
    assert decisions.allocate("zephyr|grand", 2015, "NL", raw_model="grand").status == (
        "unassignable"
    )
    # A variant can also leave the labels it names unassigned.
    assert decisions.allocate("zephyr|arrow", 2015, "NL", raw_model="arrow x import").status == (
        "unassignable"
    )
    # A variant limited to a geography applies only there.
    assert decisions.allocate("zephyr|arrow", 2021, "GB", raw_model="arrow").codes == ("Q601",)
    assert decisions.allocate("zephyr|arrow", 2021, "NL", raw_model="arrow").codes == ("Q105",)
    # The raw label also supplies the body when none is given.
    assert decisions.allocate("zephyr|arrow", 2015, "NL", raw_model="arrow variant").codes == (
        "Q103",
    )


def test_a_label_naming_several_families_pools_their_codes() -> None:
    families = {
        "zephyr/arrow": ARROW,
        "zephyr/volt": {
            "label": "Zephyr Arrow Volt",
            "make": "zephyr",
            "codes": [{"code": "Q601", "first_year": 2020, "rows": [6]}],
        },
    }
    decisions = _decisions(
        {
            "zephyr|arrow": {
                "family": "zephyr/arrow",
                "variants": [
                    {"match": ".", "geographies": ["GB"], "family": ["zephyr/arrow", "zephyr/volt"]}
                ],
            }
        },
        **families,
    )

    pooled = decisions.allocate("zephyr|arrow", 2021, "GB", raw_model="arrow")
    assert (pooled.codes, pooled.status, pooled.family) == (
        ("Q105", "Q601"),
        "shared",
        "zephyr/arrow",
    )
    assert decisions.allocate("zephyr|arrow", 2021, "NL", raw_model="arrow").codes == ("Q105",)
    # Some publishers drop the model name only in some registration years.
    yearly = _decisions(
        {
            "zephyr|arrow": {
                "family": "zephyr/arrow",
                "variants": [
                    {"match": ".", "years": [2021, 2022], "family": ["zephyr/arrow", "zephyr/volt"]}
                ],
            }
        },
        **families,
    )
    assert yearly.allocate("zephyr|arrow", 2021, "NL", raw_model="arrow").codes == ("Q105", "Q601")
    assert yearly.allocate("zephyr|arrow", 2022, "GB", raw_model="arrow").codes == ("Q105", "Q601")
    assert yearly.allocate("zephyr|arrow", 2020, "NL", raw_model="arrow").codes == ("Q102", "Q105")
    assert yearly.allocate("zephyr|arrow", 2023, "NL", raw_model="arrow").codes == ("Q105",)
    with pytest.raises(ValueError, match="listed twice"):
        _decisions({"zephyr|arrow": {"family": ["zephyr/arrow", "zephyr/arrow"]}}, **families)


def test_variants_must_name_known_families_and_readable_patterns() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        _decisions(
            {
                "zephyr|arrow": {
                    "family": "zephyr/arrow",
                    "variants": [{"match": "volt", "family": "zephyr/missing"}],
                }
            }
        )
    with pytest.raises(ValueError, match="unassigned variant names a family"):
        _decisions(
            {
                "zephyr|arrow": {
                    "family": "zephyr/arrow",
                    "variants": [
                        {"match": "x", "status": "not_in_catalogue", "family": "zephyr/arrow"}
                    ],
                }
            }
        )
    with pytest.raises(ValueError, match="geographies are country codes"):
        _decisions(
            {
                "zephyr|arrow": {
                    "family": "zephyr/arrow",
                    "variants": [{"match": "x", "family": "zephyr/arrow", "geographies": "GB"}],
                }
            }
        )
    for years in ([2021], [2022, 2021], "2021", [2021, "2022"]):
        with pytest.raises(ValueError, match="variant years are a first and a last year"):
            _decisions(
                {
                    "zephyr|arrow": {
                        "family": "zephyr/arrow",
                        "variants": [{"match": "x", "family": "zephyr/arrow", "years": years}],
                    }
                }
            )
    with pytest.raises(ValueError, match="mapped variant names no family"):
        _decisions({"zephyr|arrow": {"family": "zephyr/arrow", "variants": [{"match": "x"}]}})
    with pytest.raises(ValueError, match="unreadable variant pattern"):
        _decisions(
            {
                "zephyr|arrow": {
                    "family": "zephyr/arrow",
                    "variants": [{"match": "(", "family": "zephyr/arrow"}],
                }
            }
        )


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
        ({"uncoded_successor": 2005}, "uncoded_successor must follow first_year"),
        ({"last_year": 2010, "uncoded_successor": 2012}, "uncoded_successor after the run-out"),
        (
            {"uncoded_successor": 2020, "uncoded_successor_latest": 2019},
            "uncoded_successor_latest before uncoded_successor",
        ),
        (
            {"last_year": 2020, "uncoded_successor": 2021, "uncoded_successor_latest": 2022},
            "uncoded_successor_latest after the run-out",
        ),
        ({"uncoded_successor_latest": 2020}, "uncoded_successor_latest without uncoded_successor"),
        ({"uncoded_sibling": 2004}, "uncoded_sibling before first_year"),
        ({"last_year": 2010, "uncoded_sibling": 2012}, "uncoded_sibling after the run-out"),
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

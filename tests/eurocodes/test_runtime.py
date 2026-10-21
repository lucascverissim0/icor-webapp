"""Loading reviewed Eurocodes for the app: verified against their catalogue, described honestly.

All codes and vehicles are fictional (`Q` prefix, which the real catalogue does
not use).
"""

from __future__ import annotations

import pytest

from icor.eurocodes.catalog import parse_rows
from icor.eurocodes.runtime import EurocodeMap, identity_key

SHA = "a" * 64

CATALOGUE = parse_rows(
    (
        ("Q101", "ZEPHYR", "ARROW 5D HBK 2005-", None, 2005),
        ("Q102", "ZEPHYR", "ARROW 5D HBK 2012-", None, 2012),
        ("Q301", "ZEPHYR", "ALPHA/BETA 2014-", None, 2014),
        ("Q999", "ZEPHYR", "HAULER VRACHTWAGEN 2001-", None, 2001),
    ),
    make_aliases={},
    source_sha256=SHA,
)


def _document(**metadata: str) -> dict:
    return {
        "format": "icor-eurocode-decisions",
        "version": 1,
        "metadata": {
            "catalogue_sha256": SHA,
            "snapshot_id": "snapshot-one",
            "identity_rules_sha256": "rules-one",
            **metadata,
        },
        "families": {
            "zephyr/arrow": {
                "label": "Zephyr Arrow",
                "make": "zephyr",
                "codes": [
                    {"code": "Q101", "label": "Arrow I", "first_year": 2005, "rows": [2]},
                    {
                        "code": "Q102",
                        "label": "Arrow II",
                        "first_year": 2012,
                        "replaces": ["Q101"],
                        "rows": [3],
                    },
                ],
            },
            "zephyr/alpha": {
                "label": "Zephyr Alpha",
                "make": "zephyr",
                "codes": [
                    {
                        "code": "Q301",
                        "label": "Alpha",
                        "first_year": 2014,
                        "last_year": 2019,
                        "rows": [4],
                    }
                ],
            },
            "zephyr/beta": {
                "label": "Zephyr Beta",
                "make": "zephyr",
                "codes": [
                    {"code": "Q301", "label": "Zephyr Beta", "first_year": 2015, "rows": [4]}
                ],
            },
        },
        "identities": {
            "zephyr|arrow": {
                "family": "zephyr/arrow",
                "variants": [
                    {"match": "alpha", "family": "zephyr/alpha", "why": "names the Alpha"}
                ],
            },
            "zephyr|alpha": {"family": "zephyr/alpha"},
            "zephyr|beta": {"family": "zephyr/beta"},
        },
    }


def test_decisions_reviewed_against_another_catalogue_are_refused() -> None:
    with pytest.raises(ValueError, match="another catalogue"):
        EurocodeMap.from_documents(CATALOGUE, _document(catalogue_sha256="b" * 64))


def test_codes_are_described_with_make_label_and_registration_years() -> None:
    eurocodes = EurocodeMap.from_documents(CATALOGUE, _document())

    arrow_one = eurocodes.summary("Q101")
    assert arrow_one is not None
    assert (arrow_one.label, arrow_one.first_year, arrow_one.last_year) == (
        "Zephyr Arrow I",
        2005,
        2012,
    )
    current = eurocodes.summary("Q102")
    assert current is not None and current.last_year is None
    # One windscreen for two models carries both names and the widest years.
    shared = eurocodes.summary("Q301")
    assert shared is not None
    assert shared.label == "Zephyr Alpha / Zephyr Beta"
    assert shared.families == ("zephyr/alpha", "zephyr/beta")
    assert (shared.first_year, shared.last_year) == (2014, None)


def test_the_publishers_model_label_selects_a_reviewed_variant() -> None:
    eurocodes = EurocodeMap.from_documents(CATALOGUE, _document())

    assert eurocodes.allocate("zephyr|arrow", 2016, "DE").codes == ("Q102",)
    routed = eurocodes.allocate("zephyr|arrow", 2016, "DE", raw_model="ARROW ALPHA 1.6")
    assert (routed.codes, routed.family) == (("Q301",), "zephyr/alpha")


def test_catalogue_codes_without_reviewed_vehicles_are_known_but_undescribed() -> None:
    eurocodes = EurocodeMap.from_documents(CATALOGUE, _document())

    assert eurocodes.is_catalogue_code("Q999")
    assert eurocodes.summary("Q999") is None
    assert not eurocodes.is_catalogue_code("Q000")


def test_state_reports_stale_decisions_and_orphaned_identities() -> None:
    eurocodes = EurocodeMap.from_documents(CATALOGUE, _document())

    current = eurocodes.state(
        snapshot_id="snapshot-one",
        identity_rules_sha256="rules-one",
        present_identities={"zephyr|arrow", "zephyr|alpha", "zephyr|beta"},
    )
    assert current.loaded and current.warnings == () and current.orphaned_identities == 0
    assert current.catalogue_sha256 == SHA and current.reviewed_identities == 3

    stale = eurocodes.state(
        snapshot_id="snapshot-two",
        identity_rules_sha256="rules-two",
        present_identities={"zephyr|arrow"},
    )
    assert stale.orphaned_identities == 2
    assert len(stale.warnings) == 3


class _Index:
    def identity_for(self, vehicle_id: str) -> tuple[str, str] | None:
        return ("zephyr", "arrow") if vehicle_id == "folded" else None


def test_identity_keys_fold_or_keep_normalised_raw_labels() -> None:
    assert identity_key(_Index(), "folded", "ZEPHYR", "ARROW 1.6") == "zephyr|arrow"
    assert identity_key(_Index(), "raw", " Zephyr  Motors ", "Arrow  GT ") == (
        "raw|zephyr motors|arrow gt"
    )


def test_a_search_names_a_code_exactly_whatever_its_case() -> None:
    eurocodes = EurocodeMap.from_documents(CATALOGUE, _document())

    assert eurocodes.searched_code(" q102 ") == "Q102"
    assert eurocodes.searched_code("Q99") is None
    assert eurocodes.searched_code("Q1020") is None
    assert eurocodes.searched_code("arrow") is None


def test_the_identities_that_may_carry_a_code_include_variant_routes() -> None:
    eurocodes = EurocodeMap.from_documents(CATALOGUE, _document())

    # The Arrow's "alpha" spelling routes to the Alpha family, so it may carry Q301.
    assert eurocodes.identities_carrying("Q301") == {"zephyr|arrow", "zephyr|alpha", "zephyr|beta"}
    assert eurocodes.identities_carrying("Q101") == {"zephyr|arrow"}
    assert eurocodes.identities_carrying("Q999") == frozenset()

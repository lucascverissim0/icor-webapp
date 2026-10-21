from __future__ import annotations

from decimal import Decimal

from icor.evidence.sources.eea_model_recovery import ModelRecovery


def _key(
    model: str, *, country="DK", make="aurora", tan="e1*1", kind="t1", variant="a", version="v1"
):  # type: ignore[no-untyped-def]
    return (country, make, model, tan, kind, variant, version, "petrol")


def test_a_clear_type_approval_names_the_unnamed_rows() -> None:
    recovery = ModelRecovery()
    recovery.add_named(_key("Horizon"), 30)
    recovery.add_named(_key("horizon"), 10)
    recovery.add_named(_key("Vector"), 1)

    found = recovery.resolve(_key(""))

    assert found is not None
    assert (found.model, found.level, found.support) == ("Horizon", "MS+Mk+TAN+T+Va+Ve", 41)
    assert found.share == Decimal("0.9756")


def test_an_ambiguous_or_thin_key_names_nothing() -> None:
    ambiguous = ModelRecovery()
    ambiguous.add_named(_key("Horizon"), 90)
    ambiguous.add_named(_key("Vector"), 10)
    thin = ModelRecovery()
    thin.add_named(_key("Horizon"), 19)

    assert ambiguous.resolve(_key("")) is None
    assert thin.resolve(_key("")) is None


def test_the_most_specific_known_key_decides_and_a_make_is_never_crossed() -> None:
    recovery = ModelRecovery()
    # Only another country knows the version: the next rung, without the country, decides.
    recovery.add_named(_key("Horizon", country="SE"), 25)
    recovery.add_named(_key("Vector", make="northstar"), 500)

    found = recovery.resolve(_key("-"))

    assert found is not None and (found.model, found.level) == ("Horizon", "Mk+TAN+T+Va+Ve")
    assert recovery.resolve(_key("", make="meridian")) is None


def test_a_specific_ambiguous_key_is_not_rescued_by_a_broader_one() -> None:
    recovery = ModelRecovery()
    recovery.add_named(_key("Horizon"), 50)
    recovery.add_named(_key("Vector"), 50)
    recovery.add_named(_key("Horizon", variant="b"), 1000)

    assert recovery.resolve(_key("")) is None


def test_only_rows_with_a_country_and_make_but_no_name_are_candidates() -> None:
    assert ModelRecovery.is_candidate(_key(""))
    assert ModelRecovery.is_candidate(_key("n/a"))
    assert not ModelRecovery.is_candidate(_key("Horizon"))
    assert not ModelRecovery.is_candidate(_key("", make=""))
    assert not ModelRecovery.is_candidate(_key("", country=""))

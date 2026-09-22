import pytest

from icor.evidence.provenance import is_mixed, resolve_reported_method


def test_one_agreed_method_is_reported_as_itself() -> None:
    assert (
        resolve_reported_method(
            ("uk-dft-licensed-stock-band-v1", "uk-dft-licensed-stock-band-v1"),
            label="survival method",
        )
        == "uk-dft-licensed-stock-band-v1"
    )


def test_disagreeing_rows_are_reported_as_mixed_rather_than_resolved() -> None:
    """A snapshot mixing methods is a build problem, not something to average away.

    Picking a plausible winner here is how the same provenance defect survived
    two rounds of fixing.
    """

    reported = resolve_reported_method(
        ("constant-annual-retention-v1", "uk-dft-licensed-stock-band-v1"),
        label="survival method",
    )

    assert is_mixed(reported)
    assert "constant-annual-retention-v1" in reported
    assert "uk-dft-licensed-stock-band-v1" in reported


def test_mixed_reporting_is_order_independent() -> None:
    forward = resolve_reported_method(("b-v1", "a-v1"), label="survival method")
    backward = resolve_reported_method(("a-v1", "b-v1"), label="survival method")

    assert forward == backward


def test_nulls_and_blanks_are_not_methods() -> None:
    assert (
        resolve_reported_method((None, "  ", "a-v1"), label="survival method") == "a-v1"
    )


def test_no_method_at_all_is_refused_rather_than_defaulted() -> None:
    """A channel that cannot say how a number was produced must not guess."""

    with pytest.raises(ValueError, match="survival method"):
        resolve_reported_method((None, ""), label="survival method")


def test_a_plain_method_is_not_mixed() -> None:
    assert not is_mixed("uk-dft-licensed-stock-band-v1")

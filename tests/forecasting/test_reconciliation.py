from decimal import Decimal
from itertools import permutations

import pytest

from icor.forecasting.reconciliation import (
    CoverageCandidate,
    RegistrationCoverageSelector,
    RegistrationInput,
    RegistrationReconciler,
    coverage_precedence,
    release_precedence,
)


def test_dependency_group_is_not_double_counted() -> None:
    result = RegistrationReconciler().reconcile(
        (
            RegistrationInput("eea", "shared-register", Decimal("100"), priority=10),
            RegistrationInput("national", "shared-register", Decimal("100"), priority=20),
        )
    )

    assert result.value == Decimal("100")
    assert result.selected_input_ids == ("national",)
    assert result.excluded_input_ids == ("eea",)
    assert result.independent_evidence_count == 1


def test_independent_dependency_groups_corroborate_rather_than_add() -> None:
    """Two publishers of one population are rival measurements, never addends."""

    result = RegistrationReconciler().reconcile(
        (
            RegistrationInput("first", "register-a", Decimal("100"), priority=10),
            RegistrationInput("second", "register-b", Decimal("25"), priority=10),
        )
    )

    assert result.value == Decimal("100")
    assert result.selected_input_ids == ("first",)
    assert result.excluded_input_ids == ()
    assert result.corroborating_input_ids == ("second",)
    assert result.independent_evidence_count == 2
    assert result.agreement == "disputed"


def test_gb_two_publishers_select_one_published_value() -> None:
    """The GB regression: the 2018 total is one publisher's figure, not their sum."""

    result = RegistrationReconciler().reconcile(
        (
            RegistrationInput(
                "eea-co2cars-2018-final-v18-r1",
                "european-passenger-car-registrations-2018",
                Decimal("2355350"),
                priority=20,
            ),
            RegistrationInput(
                "uk-dft-veh0160-gb-2025-final-20260713",
                "uk-dvla-vehicle-register",
                Decimal("2341505"),
                priority=30,
            ),
        )
    )

    assert result.value == Decimal("2341505")
    assert result.value != Decimal("4696855")
    assert result.selected_input_ids == ("uk-dft-veh0160-gb-2025-final-20260713",)
    assert result.corroborating_input_ids == ("eea-co2cars-2018-final-v18-r1",)
    assert result.independent_evidence_count == 2
    assert result.agreement == "corroborated"
    assert result.max_relative_disagreement < Decimal("0.01")


def test_german_shared_group_behaviour_is_unchanged() -> None:
    """The one geography that was already correct must stay correct."""

    result = RegistrationReconciler().reconcile(
        (
            RegistrationInput(
                "eea-co2cars-2024-final-v30-r1",
                "european-passenger-car-registrations-2024",
                Decimal("2728802"),
                priority=20,
            ),
            RegistrationInput(
                "kba-fz10-2024-12-final-v3",
                "european-passenger-car-registrations-2024",
                Decimal("2797633"),
                priority=30,
            ),
        )
    )

    assert result.value == Decimal("2797633")
    assert result.selected_input_ids == ("kba-fz10-2024-12-final-v3",)
    assert result.excluded_input_ids == ("eea-co2cars-2024-final-v30-r1",)
    assert result.corroborating_input_ids == ()
    assert result.independent_evidence_count == 1


def test_correlated_and_independent_inputs_are_reported_separately() -> None:
    inputs = (
        RegistrationInput("a1", "group-a", Decimal("100"), priority=30),
        RegistrationInput("a2", "group-a", Decimal("90"), priority=10),
        RegistrationInput("b1", "group-b", Decimal("101"), priority=20),
        RegistrationInput("c1", "group-c", Decimal("99"), priority=20),
    )

    result = RegistrationReconciler().reconcile(inputs)

    assert result.selected_input_ids == ("a1",)
    assert result.excluded_input_ids == ("a2",)
    assert result.corroborating_input_ids == ("b1", "c1")
    assert result.independent_evidence_count == 3
    assert (
        len(result.selected_input_ids)
        + len(result.excluded_input_ids)
        + len(result.corroborating_input_ids)
    ) == len(inputs)


def test_single_source_is_reported_as_single_source() -> None:
    result = RegistrationReconciler().reconcile(
        (RegistrationInput("only", "group-a", Decimal("42"), priority=20),)
    )

    assert result.agreement == "single-source"
    assert result.independent_evidence_count == 1
    assert result.max_relative_disagreement == Decimal(0)


@pytest.mark.parametrize(
    ("other", "expected"),
    ((Decimal("95"), "corroborated"), (Decimal("89"), "disputed")),
)
def test_agreement_is_decided_by_the_tolerance(other: Decimal, expected: str) -> None:
    result = RegistrationReconciler(tolerance=Decimal("0.10")).reconcile(
        (
            RegistrationInput("winner", "group-a", Decimal("100"), priority=30),
            RegistrationInput("other", "group-b", other, priority=10),
        )
    )

    assert result.agreement == expected


def test_reconciliation_is_order_independent() -> None:
    inputs = (
        RegistrationInput("a1", "group-a", Decimal("100"), priority=30),
        RegistrationInput("a2", "group-a", Decimal("90"), priority=10),
        RegistrationInput("b1", "group-b", Decimal("101"), priority=20),
        RegistrationInput("c1", "group-c", Decimal("99"), priority=20),
    )
    reconciler = RegistrationReconciler()
    expected = reconciler.reconcile(inputs)

    for permutation in permutations(inputs):
        assert reconciler.reconcile(permutation) == expected


def test_equal_priority_across_groups_breaks_ties_on_input_id() -> None:
    result = RegistrationReconciler().reconcile(
        (
            RegistrationInput("zulu", "group-z", Decimal("10"), priority=20),
            RegistrationInput("alpha", "group-a", Decimal("20"), priority=20),
        )
    )

    assert result.selected_input_ids == ("alpha",)
    assert result.value == Decimal("20")


def test_zero_values_report_zero_disagreement() -> None:
    result = RegistrationReconciler().reconcile(
        (
            RegistrationInput("winner", "group-a", Decimal("0"), priority=30),
            RegistrationInput("other", "group-b", Decimal("0"), priority=10),
        )
    )

    assert result.max_relative_disagreement == Decimal(0)
    assert result.agreement == "corroborated"


def test_a_zero_against_a_positive_value_is_total_disagreement() -> None:
    result = RegistrationReconciler().reconcile(
        (
            RegistrationInput("winner", "group-a", Decimal("0"), priority=30),
            RegistrationInput("other", "group-b", Decimal("5"), priority=10),
        )
    )

    assert result.max_relative_disagreement == Decimal(1)
    assert result.agreement == "disputed"


def test_empty_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="requires inputs"):
        RegistrationReconciler().reconcile(())


def test_duplicate_input_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="uniquely identified"):
        RegistrationReconciler().reconcile(
            (
                RegistrationInput("same", "group-a", Decimal("1"), priority=10),
                RegistrationInput("same", "group-b", Decimal("2"), priority=10),
            )
        )


def test_release_precedence_prefers_national_registers_for_rival_values() -> None:
    assert release_precedence("uk-dft-veh0160") == 30
    assert release_precedence("kba-fz10") == 30
    assert release_precedence("eea-co2-monitoring") == 20
    assert release_precedence("some-other-source") == 10


def test_coverage_precedence_prefers_the_pan_european_compilation() -> None:
    """The decomposer is chosen for granularity and consistency, not for totals."""

    assert coverage_precedence("eea-co2-monitoring") > coverage_precedence("uk-dft-veh0160")
    assert coverage_precedence("eea-co2-monitoring") > coverage_precedence("kba-fz10")
    assert coverage_precedence("kba-fz10") > coverage_precedence("some-other-source")


def _candidate(source_id: str, status: str = "final") -> CoverageCandidate:
    return CoverageCandidate(source_id=source_id, publication_status=status)


def test_coverage_selects_the_pan_european_source_when_both_cover_the_year() -> None:
    selection = RegistrationCoverageSelector().select(
        (_candidate("uk-dft-veh0160"), _candidate("eea-co2-monitoring"))
    )

    assert selection.source_id == "eea-co2-monitoring"
    assert selection.corroborating_source_ids == ("uk-dft-veh0160",)


def test_coverage_falls_back_to_the_only_source_present() -> None:
    """GB before 2010 and after 2020 has no EEA coverage at all."""

    selection = RegistrationCoverageSelector().select((_candidate("uk-dft-veh0160"),))

    assert selection.source_id == "uk-dft-veh0160"
    assert selection.corroborating_source_ids == ()


def test_coverage_prefers_a_final_release_over_a_provisional_one() -> None:
    selection = RegistrationCoverageSelector().select(
        (
            _candidate("eea-co2-monitoring", "provisional"),
            _candidate("kba-fz10", "final"),
        )
    )

    assert selection.source_id == "kba-fz10"
    assert selection.corroborating_source_ids == ("eea-co2-monitoring",)


def test_coverage_treats_a_correction_as_final_grade() -> None:
    selection = RegistrationCoverageSelector().select(
        (
            _candidate("eea-co2-monitoring", "corrected"),
            _candidate("kba-fz10", "final"),
        )
    )

    assert selection.source_id == "eea-co2-monitoring"


def test_coverage_selection_is_order_independent() -> None:
    candidates = (
        _candidate("eea-co2-monitoring"),
        _candidate("kba-fz10"),
        _candidate("uk-dft-veh0160"),
    )
    selector = RegistrationCoverageSelector()
    expected = selector.select(candidates)

    for permutation in permutations(candidates):
        assert selector.select(permutation) == expected


def test_coverage_requires_candidates() -> None:
    with pytest.raises(ValueError, match="requires candidates"):
        RegistrationCoverageSelector().select(())

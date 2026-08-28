from decimal import Decimal

from icor.forecasting.reconciliation import RegistrationInput, RegistrationReconciler


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


def test_independent_dependency_groups_are_additive() -> None:
    result = RegistrationReconciler().reconcile(
        (
            RegistrationInput("first", "register-a", Decimal("100"), priority=10),
            RegistrationInput("second", "register-b", Decimal("25"), priority=10),
        )
    )

    assert result.value == Decimal("125")
    assert result.independent_evidence_count == 2

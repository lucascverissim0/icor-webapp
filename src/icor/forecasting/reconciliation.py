"""Dependency-aware reconciliation of semantically equivalent registration totals."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class RegistrationInput:
    input_id: str
    dependency_group: str
    value: Decimal
    priority: int

    def __post_init__(self) -> None:
        if not self.input_id.strip() or not self.dependency_group.strip():
            raise ValueError("registration input identity is required")
        if not self.value.is_finite() or self.value < 0:
            raise ValueError("registration input value must be finite and non-negative")
        if type(self.priority) is not int:
            raise ValueError("registration input priority must be an integer")


@dataclass(frozen=True, slots=True)
class ReconciledRegistration:
    value: Decimal
    selected_input_ids: tuple[str, ...]
    excluded_input_ids: tuple[str, ...]
    independent_evidence_count: int
    status: str = "reconciled"


class RegistrationReconciler:
    """Select one deterministic winner per correlated dependency group."""

    def reconcile(
        self, inputs: tuple[RegistrationInput, ...]
    ) -> ReconciledRegistration:
        if not inputs:
            raise ValueError("registration reconciliation requires inputs")
        grouped: dict[str, list[RegistrationInput]] = {}
        for item in inputs:
            grouped.setdefault(item.dependency_group, []).append(item)
        selected: list[RegistrationInput] = []
        excluded: list[str] = []
        for dependency_group in sorted(grouped):
            candidates = sorted(
                grouped[dependency_group],
                key=lambda item: (-item.priority, item.input_id),
            )
            selected.append(candidates[0])
            excluded.extend(item.input_id for item in candidates[1:])
        return ReconciledRegistration(
            value=sum((item.value for item in selected), start=Decimal(0)),
            selected_input_ids=tuple(item.input_id for item in selected),
            excluded_input_ids=tuple(sorted(excluded)),
            independent_evidence_count=len(selected),
        )

"""Reconciliation of semantically equivalent registration totals.

Two publishers reporting new registrations for one country and one year are
measuring the same cars. That is corroboration, not addition, and this module
keeps the two apart.

There are two decisions, at two different scopes.

*Which publisher decomposes a country-year* is the coverage decision. When the
EEA compilation and a national register both cover a year they do so at different
granularities -- 6,447 GB vehicles against 1,054, 2,267 DE vehicles against 366 --
so merging them per vehicle adds the rows only one publisher happens to name and
inflates the country by up to a factor of two. Exactly one publisher must
decompose a country-year; the others become corroboration.

*Which value wins among rival measurements of one vehicle-year* is the
reconciliation decision, below it. The answer is always a figure some publisher
actually published, never a sum and never a mean, because the observation ids
cited alongside it have to explain the number.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

# Agreement within this band reads as two publishers corroborating one another;
# beyond it, the disagreement is recorded on the cohort so it stays visible.
# Measured against the promoted snapshot, GB cross-publisher disagreement sits at
# p50 0.16, so this admits the agreeing majority without silencing the rest.
DEFAULT_CORROBORATION_TOLERANCE = Decimal("0.10")

_PAN_EUROPEAN_SOURCE = "eea-co2-monitoring"
_NATIONAL_REGISTER_PREFIXES = ("kba-", "uk-dft-")

# A correction supersedes the figure it corrects and is final-grade evidence.
_FINAL_GRADE_STATUSES = frozenset({"final", "corrected"})
_PROVISIONAL_STATUSES = frozenset({"provisional"})


def release_precedence(source_id: str) -> int:
    """Rank rival measurements of one vehicle-year.

    A national register counts its own country's registrations administratively,
    so where it and the compilation disagree about the same vehicle, it wins.
    """

    if source_id.startswith(_NATIONAL_REGISTER_PREFIXES):
        return 30
    if source_id == _PAN_EUROPEAN_SOURCE:
        return 20
    return 10


def coverage_precedence(source_id: str) -> int:
    """Rank publishers as candidate decomposers of a whole country-year.

    This deliberately inverts `release_precedence`. Both publishers report the
    same national total to within a percent or so, so the total is not what
    distinguishes them: granularity and cross-market consistency are. The EEA
    compilation resolves several times more vehicles and is the one source
    covering all thirty markets on the same basis, so it decomposes the year
    wherever it is present, and the national register corroborates it.
    """

    if source_id == _PAN_EUROPEAN_SOURCE:
        return 30
    if source_id.startswith(_NATIONAL_REGISTER_PREFIXES):
        return 20
    return 10


def _status_rank(publication_status: str) -> int:
    normalized = publication_status.strip().casefold()
    if normalized in _FINAL_GRADE_STATUSES:
        return 0
    if normalized in _PROVISIONAL_STATUSES:
        return 1
    return 2


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
    corroborating_input_ids: tuple[str, ...] = ()
    max_relative_disagreement: Decimal = Decimal(0)
    agreement: str = "single-source"
    status: str = "reconciled"


@dataclass(frozen=True, slots=True)
class CoverageCandidate:
    source_id: str
    publication_status: str

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("coverage candidate source is required")
        if not self.publication_status.strip():
            raise ValueError("coverage candidate publication status is required")


@dataclass(frozen=True, slots=True)
class SelectedCoverage:
    source_id: str
    corroborating_source_ids: tuple[str, ...]


class RegistrationCoverageSelector:
    """Choose the one publisher that decomposes a given country-year."""

    def select(self, candidates: tuple[CoverageCandidate, ...]) -> SelectedCoverage:
        if not candidates:
            raise ValueError("coverage selection requires candidates")
        distinct = {candidate.source_id: candidate for candidate in candidates}
        winner = min(distinct.values(), key=self._precedence)
        return SelectedCoverage(
            source_id=winner.source_id,
            corroborating_source_ids=tuple(
                sorted(source for source in distinct if source != winner.source_id)
            ),
        )

    @staticmethod
    def _precedence(candidate: CoverageCandidate) -> tuple[int, int, str]:
        return (
            _status_rank(candidate.publication_status),
            -coverage_precedence(candidate.source_id),
            candidate.source_id,
        )


def _relative_gap(winner: Decimal, other: Decimal) -> Decimal:
    """Symmetric, bounded in [0, 1], and safe when either value is zero."""

    largest = max(winner, other)
    if largest == 0:
        return Decimal(0)
    return (abs(winner - other) / largest).quantize(Decimal("0.000001"))


class RegistrationReconciler:
    """Select one published value per vehicle-year, and record who agreed."""

    def __init__(self, tolerance: Decimal = DEFAULT_CORROBORATION_TOLERANCE) -> None:
        if not tolerance.is_finite() or tolerance < 0:
            raise ValueError("corroboration tolerance must be finite and non-negative")
        self.tolerance = tolerance

    def reconcile(self, inputs: tuple[RegistrationInput, ...]) -> ReconciledRegistration:
        if not inputs:
            raise ValueError("registration reconciliation requires inputs")
        if len({item.input_id for item in inputs}) != len(inputs):
            raise ValueError("registration inputs must be uniquely identified")

        grouped: dict[str, list[RegistrationInput]] = {}
        for item in inputs:
            grouped.setdefault(item.dependency_group, []).append(item)

        representatives: list[RegistrationInput] = []
        excluded: list[str] = []
        for dependency_group in sorted(grouped):
            ordered = sorted(grouped[dependency_group], key=_precedence)
            representatives.append(ordered[0])
            excluded.extend(item.input_id for item in ordered[1:])

        winner = min(representatives, key=_precedence)
        corroborating = tuple(
            sorted(
                item.input_id
                for item in representatives
                if item.input_id != winner.input_id
            )
        )
        disagreement = max(
            (
                _relative_gap(winner.value, item.value)
                for item in representatives
                if item.input_id != winner.input_id
            ),
            default=Decimal(0),
        )
        if not corroborating:
            agreement = "single-source"
        elif disagreement <= self.tolerance:
            agreement = "corroborated"
        else:
            agreement = "disputed"

        return ReconciledRegistration(
            value=winner.value,
            selected_input_ids=(winner.input_id,),
            excluded_input_ids=tuple(sorted(excluded)),
            independent_evidence_count=len(representatives),
            corroborating_input_ids=corroborating,
            max_relative_disagreement=disagreement,
            agreement=agreement,
        )


def _precedence(item: RegistrationInput) -> tuple[int, str]:
    return (-item.priority, item.input_id)

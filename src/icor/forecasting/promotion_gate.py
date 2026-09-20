"""Fail-closed promotion policy for registration-forecast challengers."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_ZERO = Decimal(0)
_HUNDRED = Decimal(100)


@dataclass(frozen=True, slots=True)
class ValidationSlice:
    """Comparable baseline and challenger results for one governed segment."""

    name: str
    point_count: int
    actual_units: Decimal
    baseline_absolute_error: Decimal
    challenger_absolute_error: Decimal
    interval_covered_units: Decimal | None = None
    interval_total_units: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("validation slice name is required")
        if type(self.point_count) is not int or self.point_count < 1:
            raise ValueError("validation slice point count must be positive")
        for value, label in (
            (self.actual_units, "actual units"),
            (self.baseline_absolute_error, "baseline absolute error"),
            (self.challenger_absolute_error, "challenger absolute error"),
        ):
            _non_negative(value, label)
        if self.actual_units == 0:
            raise ValueError("validation slice actual units must be positive")
        interval_values = self.interval_covered_units, self.interval_total_units
        if any(value is not None for value in interval_values):
            if any(value is None for value in interval_values):
                raise ValueError(
                    "interval coverage numerator and denominator are required together"
                )
            covered, total = interval_values
            _non_negative(covered, "interval covered units")
            _non_negative(total, "interval total units")
            if total == 0 or covered > total:
                raise ValueError("interval coverage units must be ordered and positive")

    @property
    def baseline_wape(self) -> Decimal:
        return self.baseline_absolute_error / self.actual_units

    @property
    def challenger_wape(self) -> Decimal:
        return self.challenger_absolute_error / self.actual_units

    @property
    def relative_improvement_percent(self) -> Decimal:
        if self.baseline_absolute_error == 0:
            return _ZERO if self.challenger_absolute_error == 0 else Decimal("-Infinity")
        return (
            (Decimal(1) - self.challenger_absolute_error / self.baseline_absolute_error)
            * _HUNDRED
        )

    @property
    def interval_coverage(self) -> Decimal | None:
        if self.interval_total_units is None or self.interval_covered_units is None:
            return None
        return self.interval_covered_units / self.interval_total_units


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    """All evidence needed to make one deterministic promotion decision."""

    outcome: str
    holdout_snapshot_id: str
    development_snapshot_ids: tuple[str, ...]
    observed_targets_only: bool
    publication_vintages_as_of_origin: bool
    unseen_holdout: bool
    eligible_target_units: Decimal
    evaluated_target_units: Decimal
    overall: ValidationSlice
    horizons: tuple[ValidationSlice, ...]
    countries: tuple[ValidationSlice, ...]
    identity_confidence: tuple[ValidationSlice, ...]
    history_lengths: tuple[ValidationSlice, ...]
    volume_deciles: tuple[ValidationSlice, ...]

    def __post_init__(self) -> None:
        if self.outcome != "annual_vehicle_registrations":
            raise ValueError("promotion evidence must target annual vehicle registrations")
        if not self.holdout_snapshot_id.strip():
            raise ValueError("holdout snapshot ID is required")
        if not self.development_snapshot_ids or any(
            not value.strip() for value in self.development_snapshot_ids
        ):
            raise ValueError("development snapshot IDs are required")
        if len(set(self.development_snapshot_ids)) != len(self.development_snapshot_ids):
            raise ValueError("development snapshot IDs must be unique")
        if type(self.observed_targets_only) is not bool:
            raise ValueError("observed-target flag must be boolean")
        if type(self.publication_vintages_as_of_origin) is not bool:
            raise ValueError("publication-vintage flag must be boolean")
        if type(self.unseen_holdout) is not bool:
            raise ValueError("unseen-holdout flag must be boolean")
        _non_negative(self.eligible_target_units, "eligible target units")
        _non_negative(self.evaluated_target_units, "evaluated target units")
        if self.eligible_target_units == 0:
            raise ValueError("eligible target units must be positive")
        if self.evaluated_target_units > self.eligible_target_units:
            raise ValueError("evaluated target units cannot exceed eligible target units")
        if self.overall.actual_units != self.evaluated_target_units:
            raise ValueError("overall actual units must equal evaluated target units")
        for values, label in (
            (self.horizons, "horizon"),
            (self.countries, "country"),
            (self.identity_confidence, "identity-confidence"),
            (self.history_lengths, "history-length"),
            (self.volume_deciles, "volume-decile"),
        ):
            if not values:
                raise ValueError(f"{label} validation slices are required")
            names = tuple(value.name for value in values)
            if len(names) != len(set(names)):
                raise ValueError(f"{label} validation slice names must be unique")
            if sum((item.actual_units for item in values), start=_ZERO) != (
                self.overall.actual_units
            ):
                raise ValueError(f"{label} validation slices must partition actual units")

    @property
    def evaluated_volume_coverage(self) -> Decimal:
        return self.evaluated_target_units / self.eligible_target_units


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """Predeclared accuracy, coverage, interval, and regression thresholds."""

    minimum_relative_wape_improvement_percent: Decimal = Decimal("2")
    minimum_evaluated_volume_coverage: Decimal = Decimal("0.80")
    maximum_material_slice_regression_percent: Decimal = Decimal("2")
    target_interval_coverage: Decimal = Decimal("0.80")
    interval_coverage_tolerance: Decimal = Decimal("0.05")
    minimum_material_slice_units_share: Decimal = Decimal("0.005")

    def __post_init__(self) -> None:
        for value, label in (
            (
                self.minimum_relative_wape_improvement_percent,
                "minimum relative WAPE improvement",
            ),
            (
                self.maximum_material_slice_regression_percent,
                "maximum material slice regression",
            ),
        ):
            _bounded(value, label, _ZERO, _HUNDRED)
        for value, label in (
            (self.minimum_evaluated_volume_coverage, "minimum volume coverage"),
            (self.target_interval_coverage, "target interval coverage"),
            (self.interval_coverage_tolerance, "interval coverage tolerance"),
            (
                self.minimum_material_slice_units_share,
                "minimum material slice units share",
            ),
        ):
            _bounded(value, label, _ZERO, Decimal(1))
        if (
            self.target_interval_coverage - self.interval_coverage_tolerance < 0
            or self.target_interval_coverage + self.interval_coverage_tolerance > 1
        ):
            raise ValueError("interval coverage policy must stay between zero and one")


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    eligible: bool
    blockers: tuple[str, ...]


def evaluate_promotion(
    evidence: PromotionEvidence,
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    """Evaluate a challenger without silently relaxing any promotion gate."""

    policy = policy or PromotionPolicy()
    blockers: list[str] = []
    if evidence.holdout_snapshot_id in evidence.development_snapshot_ids:
        blockers.append("holdout_snapshot_used_during_development")
    if not evidence.unseen_holdout:
        blockers.append("holdout_not_frozen_and_unseen")
    if not evidence.observed_targets_only:
        blockers.append("targets_include_estimated_or_forecast_values")
    if not evidence.publication_vintages_as_of_origin:
        blockers.append("publication_vintages_do_not_reconstruct_forecast_origin")
    if evidence.evaluated_volume_coverage < policy.minimum_evaluated_volume_coverage:
        blockers.append("evaluated_volume_coverage_below_policy")
    if (
        evidence.overall.relative_improvement_percent
        < policy.minimum_relative_wape_improvement_percent
    ):
        blockers.append("overall_wape_improvement_below_policy")
    if any(item.challenger_wape >= item.baseline_wape for item in evidence.horizons):
        blockers.append("challenger_does_not_improve_every_horizon")
    if _material_regressions(evidence.countries, evidence.overall, policy):
        blockers.append("material_country_regression")
    if _material_regressions(evidence.identity_confidence, evidence.overall, policy):
        blockers.append("material_identity_confidence_regression")
    if _material_regressions(evidence.history_lengths, evidence.overall, policy):
        blockers.append("material_history_length_regression")
    if _material_regressions(evidence.volume_deciles, evidence.overall, policy):
        blockers.append("material_volume_decile_regression")
    coverage = evidence.overall.interval_coverage
    if coverage is None:
        blockers.append("prediction_intervals_not_evaluated")
    elif abs(coverage - policy.target_interval_coverage) > policy.interval_coverage_tolerance:
        blockers.append("prediction_interval_coverage_outside_policy")
    return PromotionDecision(not blockers, tuple(blockers))


def _material_regressions(
    slices: tuple[ValidationSlice, ...],
    overall: ValidationSlice,
    policy: PromotionPolicy,
) -> tuple[str, ...]:
    minimum_units = (
        overall.actual_units * policy.minimum_material_slice_units_share
    )
    return tuple(
        item.name
        for item in slices
        if item.actual_units >= minimum_units
        and item.challenger_wape - item.baseline_wape
        > policy.maximum_material_slice_regression_percent / _HUNDRED
    )


def _non_negative(value: Decimal | None, label: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(f"{label} must be a finite non-negative Decimal")


def _bounded(value: Decimal, label: str, minimum: Decimal, maximum: Decimal) -> None:
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label} is outside the supported policy range")

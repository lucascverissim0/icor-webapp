"""One ranked row per Eurocode: the units certainly its own, and what it may share.

The opportunity ranking can be read by Eurocode, ICOR's own unit. A row is one
code. Its demand is what the code certainly carries — registrations allocated to
it alone — so a score built on it is a conservative lower bound. Units a code
shares with others (a launch year, a label naming no body, a vehicle without a
code) are reported beside it as "up to", never split, and are counted once in
the view's totals. Units no code explains are totals of their own, so the rows
and the totals together account for the whole view.

Both opportunity sources feed the same accumulator, so the snapshot ranking and
the in-memory one cannot disagree about a row.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field

from icor.domain.opportunities import CoverageStatus
from icor.domain.planner import DemandRange
from icor.eurocodes.allocation import Allocation
from icor.eurocodes.annotation import EurocodeBreakdown
from icor.eurocodes.runtime import EurocodeMap
from icor.evidence.normalization import source_vehicle_display_label


@dataclass(frozen=True, slots=True)
class EurocodeVehicle:
    """One vehicle label whose registrations may carry the code."""

    brand: str
    model: str
    certain_units: int
    shared_units: int


@dataclass(frozen=True, slots=True)
class EvidenceMix:
    """A code's certain units by what their cohorts rest on; they add up to its demand.

    ``stock`` was counted by a national register, ``registrations`` are published
    new-car registrations aged by the survival curve, ``interpolated`` are
    registrations estimated between published years and ``forecast`` are
    registrations not published yet. ``unknown`` comes from a source that does
    not record it.
    """

    stock: int = 0
    registrations: int = 0
    interpolated: int = 0
    forecast: int = 0
    unknown: int = 0

    @property
    def total(self) -> int:
        return self.stock + self.registrations + self.interpolated + self.forecast + self.unknown


@dataclass(frozen=True, slots=True)
class EurocodeGroup:
    """What a Eurocode row shows beside its certain demand."""

    code: str
    label: str
    #: The display make of the code's first family; a rebadged code has several.
    make: str
    first_year: int
    last_year: int | None
    drive: str
    #: Units the code may carry but shares with other codes, beyond its certain ones.
    shared_units: int
    shared_with: tuple[str, ...]
    shared_with_uncoded: bool
    vehicles: tuple[EurocodeVehicle, ...]
    #: What the certain units rest on.
    evidence: EvidenceMix


@dataclass(frozen=True, slots=True)
class RankedCell:
    """Forecast units of one estimate in one registration year, with what labels it."""

    identity: str
    year: int
    geography: str
    raw_model: str
    #: The canonical labels the vehicle is ranked under elsewhere.
    brand: str
    model: str
    horizon: int
    configuration: Hashable
    demand: DemandRange
    coverage_status: str
    icor_worked: bool
    #: The cohort's evidence kind (an ``EvidenceMix`` field); None where it is not recorded.
    evidence: str | None = None


@dataclass(frozen=True, slots=True)
class EurocodeTotals:
    """Everything one Eurocode row is built from."""

    group: EurocodeGroup
    #: Certain units only.
    demand: DemandRange
    exact_units: int
    fallback_units: int
    uncovered_units: int
    icor_worked_units: int
    configuration_count: int
    #: Certain units by market and horizon, ordered by horizon then market.
    contributions: tuple[tuple[str, int, DemandRange], ...]


@dataclass(frozen=True, slots=True)
class EurocodeRanking:
    groups: Mapping[str, EurocodeTotals]
    #: The view's units by outcome, shared units counted once; no codes listed.
    totals: EurocodeBreakdown


@dataclass(slots=True)
class _Accumulator:
    downside: int = 0
    base: int = 0
    upside: int = 0
    exact: int = 0
    fallback: int = 0
    uncovered: int = 0
    worked: int = 0
    shared: int = 0
    partners: set[str] = field(default_factory=set)
    uncoded: bool = False
    configurations: set[Hashable] = field(default_factory=set)
    vehicles: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    contributions: dict[tuple[str, int], list[int]] = field(default_factory=dict)
    evidence: dict[str, int] = field(default_factory=dict)


def rank_by_eurocode(eurocodes: EurocodeMap, cells: Iterable[RankedCell]) -> EurocodeRanking:
    """Fold forecast cells into one total per code and the view's outcome totals."""

    allocations: dict[tuple[str, int, str, str], Allocation] = {}
    codes: dict[str, _Accumulator] = {}
    outcomes = dict.fromkeys(("single", "shared", "outside", "unassignable", "unreviewed"), 0)
    for cell in cells:
        units = cell.demand.base_units
        key = (cell.identity, cell.year, cell.geography, cell.raw_model)
        allocation = allocations.get(key)
        if allocation is None:
            allocation = allocations[key] = eurocodes.allocate(
                cell.identity, cell.year, cell.geography, raw_model=cell.raw_model
            )
        outcomes[allocation.status] += units
        if allocation.status == "single":
            entry = codes.setdefault(allocation.codes[0], _Accumulator())
            entry.downside += cell.demand.downside_units
            entry.base += units
            entry.upside += cell.demand.upside_units
            if cell.coverage_status == CoverageStatus.EXACT_COVERED:
                entry.exact += units
            elif cell.coverage_status == CoverageStatus.FALLBACK_ONLY:
                entry.fallback += units
            else:
                entry.uncovered += units
            if cell.icor_worked:
                entry.worked += units
            kind = cell.evidence or "unknown"
            entry.evidence[kind] = entry.evidence.get(kind, 0) + units
            entry.configurations.add(cell.configuration)
            entry.vehicles.setdefault((cell.brand, cell.model), [0, 0])[0] += units
            market = entry.contributions.setdefault((cell.geography, cell.horizon), [0, 0, 0])
            market[0] += cell.demand.downside_units
            market[1] += units
            market[2] += cell.demand.upside_units
        elif allocation.status == "shared" and units:
            for code in allocation.codes:
                entry = codes.setdefault(code, _Accumulator())
                entry.shared += units
                entry.partners.update(other for other in allocation.codes if other != code)
                entry.uncoded = entry.uncoded or allocation.uncoded
                entry.vehicles.setdefault((cell.brand, cell.model), [0, 0])[1] += units
    groups: dict[str, EurocodeTotals] = {}
    for code, entry in codes.items():
        if not entry.base and not entry.shared:
            continue
        summary = eurocodes.summary(code)
        assert summary is not None, "an allocated code is a reviewed code"
        evidence = EvidenceMix(**entry.evidence)
        assert evidence.total == entry.base, "every certain unit has one evidence kind"
        groups[code] = EurocodeTotals(
            group=EurocodeGroup(
                code=code,
                label=summary.label,
                make=source_vehicle_display_label(
                    eurocodes.decisions.families[summary.families[0]].make
                ),
                first_year=summary.first_year,
                last_year=summary.last_year,
                drive=summary.drive,
                shared_units=entry.shared,
                shared_with=tuple(sorted(entry.partners)),
                shared_with_uncoded=entry.uncoded,
                vehicles=tuple(
                    EurocodeVehicle(brand, model, certain, shared)
                    for (brand, model), (certain, shared) in sorted(
                        entry.vehicles.items(),
                        key=lambda item: (-item[1][0], -item[1][1], item[0]),
                    )
                ),
                evidence=evidence,
            ),
            demand=DemandRange(entry.downside, entry.base, entry.upside),
            exact_units=entry.exact,
            fallback_units=entry.fallback,
            uncovered_units=entry.uncovered,
            icor_worked_units=entry.worked,
            configuration_count=len(entry.configurations),
            contributions=tuple(
                (market, horizon, DemandRange(*units))
                for (market, horizon), units in sorted(
                    entry.contributions.items(), key=lambda item: (item[0][1], item[0][0])
                )
                if any(units)
            ),
        )
    return EurocodeRanking(
        groups=groups,
        totals=EurocodeBreakdown(
            codes=(),
            single_units=outcomes["single"],
            shared_units=outcomes["shared"],
            outside_units=outcomes["outside"],
            unassignable_units=outcomes["unassignable"],
            unreviewed_units=outcomes["unreviewed"],
        ),
    )

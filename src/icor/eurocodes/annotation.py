"""Which Eurocodes a ranked row's registrations may carry, without guessing a split.

A row sums registrations of several publisher spellings, markets and
registration years. Each such cell is allocated on its own, because a reviewed
variant is keyed on the spelling, a drive side on the market, and a code's window
on the year. A cell then belongs to one code, is shared by several (a launch
year, a label that names no body, a vehicle without a code), or is not assigned.

Shared units are never divided between codes. Each code reports the units that
are certainly its own and, apart from them, the units it may share and with
which codes: "certain X, up to Y more". Unassigned units are reported by reason,
so the row's units are always accounted for in full.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from icor.eurocodes.runtime import EurocodeMap


@dataclass(frozen=True, slots=True)
class EurocodeCell:
    """Registrations of one spelling of one identity, in one market and year."""

    identity: str
    year: int
    geography: str
    raw_model: str
    units: int


@dataclass(frozen=True, slots=True)
class EurocodeShare:
    """One code's part of a row."""

    code: str
    label: str
    first_year: int
    last_year: int | None
    drive: str
    #: Units only this code can explain.
    certain_units: int
    #: Units this code may explain, shared with ``shared_with`` and never split.
    shared_units: int
    shared_with: tuple[str, ...]
    #: Some shared units may be a vehicle the catalogue has no code for.
    shared_with_uncoded: bool


@dataclass(frozen=True, slots=True)
class EurocodeBreakdown:
    """A row's units by Eurocode; every unit appears in exactly one total."""

    codes: tuple[EurocodeShare, ...]
    #: Units in cells with exactly one code.
    single_units: int
    #: Units in cells shared by several codes or with a vehicle without a code.
    shared_units: int
    #: A reviewed vehicle in a year no code of its family covers.
    outside_units: int
    #: Reviewed, but the label cannot name one catalogued vehicle.
    unassignable_units: int
    #: Not reviewed.
    unreviewed_units: int

    @property
    def total_units(self) -> int:
        return (
            self.single_units
            + self.shared_units
            + self.outside_units
            + self.unassignable_units
            + self.unreviewed_units
        )


def eurocode_breakdown(eurocodes: EurocodeMap, cells: Iterable[EurocodeCell]) -> EurocodeBreakdown:
    totals = {"single": 0, "shared": 0, "outside": 0, "unassignable": 0, "unreviewed": 0}
    certain: dict[str, int] = {}
    shared: dict[str, int] = {}
    partners: dict[str, set[str]] = {}
    uncoded: set[str] = set()
    families: dict[str, tuple[str, ...]] = {}
    for cell in cells:
        if not cell.units:
            continue
        allocation = eurocodes.allocate(
            cell.identity, cell.year, cell.geography, raw_model=cell.raw_model
        )
        totals[allocation.status] += cell.units
        for code in allocation.codes:
            families.setdefault(code, allocation.families)
            if allocation.status == "single":
                certain[code] = certain.get(code, 0) + cell.units
            else:
                shared[code] = shared.get(code, 0) + cell.units
                partners.setdefault(code, set()).update(
                    other for other in allocation.codes if other != code
                )
                if allocation.uncoded:
                    uncoded.add(code)
    shares = []
    for code in families:
        summary = eurocodes.describe(code, families[code])
        assert summary is not None, "an allocated code is always a reviewed family's code"
        shares.append(
            EurocodeShare(
                code=code,
                label=summary.label,
                first_year=summary.first_year,
                last_year=summary.last_year,
                drive=summary.drive,
                certain_units=certain.get(code, 0),
                shared_units=shared.get(code, 0),
                shared_with=tuple(sorted(partners.get(code, ()))),
                shared_with_uncoded=code in uncoded,
            )
        )
    shares.sort(key=lambda share: (-share.certain_units, -share.shared_units, share.code))
    return EurocodeBreakdown(
        codes=tuple(shares),
        single_units=totals["single"],
        shared_units=totals["shared"],
        outside_units=totals["outside"],
        unassignable_units=totals["unassignable"],
        unreviewed_units=totals["unreviewed"],
    )

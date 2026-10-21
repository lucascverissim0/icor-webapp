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

Cells may carry fractional units (estimated registrations). The totals are then
rounded once, by largest remainder, so they still sum to the rounded whole.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from icor.eurocodes.runtime import EurocodeMap


@dataclass(frozen=True, slots=True)
class EurocodeCell:
    """Registrations of one spelling of one identity, in one market and year."""

    identity: str
    year: int
    geography: str
    raw_model: str
    units: int | Decimal


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


_STATUSES = ("single", "shared", "outside", "unassignable", "unreviewed")


def _whole(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _apportion(values: Sequence[Decimal], total: int | None = None) -> list[int]:
    """Whole numbers summing to ``total`` (the rounded sum), by largest remainder."""

    target = _whole(sum(values, Decimal(0))) if total is None else total
    floors = [int(value) for value in values]
    order = sorted(range(len(values)), key=lambda index: (floors[index] - values[index], index))
    for index in order[: target - sum(floors)]:
        floors[index] += 1
    return floors


def eurocode_breakdown(eurocodes: EurocodeMap, cells: Iterable[EurocodeCell]) -> EurocodeBreakdown:
    totals = {status: Decimal(0) for status in _STATUSES}
    certain: dict[str, Decimal] = {}
    shared: dict[str, Decimal] = {}
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
                certain[code] = certain.get(code, Decimal(0)) + cell.units
            else:
                shared[code] = shared.get(code, Decimal(0)) + cell.units
                partners.setdefault(code, set()).update(
                    other for other in allocation.codes if other != code
                )
                if allocation.uncoded:
                    uncoded.add(code)
    single, shared_total, outside, unassignable, unreviewed = _apportion(
        [totals[status] for status in _STATUSES]
    )
    certain_whole = dict(
        zip(
            families,
            _apportion([certain.get(code, Decimal(0)) for code in families], single),
            strict=True,
        )
    )
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
                certain_units=certain_whole[code],
                shared_units=min(_whole(shared.get(code, Decimal(0))), shared_total),
                shared_with=tuple(sorted(partners.get(code, ()))),
                shared_with_uncoded=code in uncoded,
            )
        )
    shares.sort(key=lambda share: (-share.certain_units, -share.shared_units, share.code))
    return EurocodeBreakdown(
        codes=tuple(shares),
        single_units=single,
        shared_units=shared_total,
        outside_units=outside,
        unassignable_units=unassignable,
        unreviewed_units=unreviewed,
    )


@dataclass(frozen=True, slots=True)
class EurocodeSearch:
    """A search text that is also a Eurocode, which is then read both ways.

    A four-character code can also be part of a vehicle name, or look like a
    year, so a search is never narrowed to one reading: vehicles whose name
    contains the text and vehicles that may carry the code are both listed, and
    this names the code reading so the interface can say so.
    """

    code: str
    #: The code as reviewed vehicles carry it; None when no reviewed vehicle
    #: does (a commercial vehicle's code, or one not reviewed yet).
    label: str | None = None
    first_year: int | None = None
    last_year: int | None = None


def eurocode_search(eurocodes: EurocodeMap | None, text: str) -> EurocodeSearch | None:
    """The code reading of a search text, if the text is exactly a catalogue code."""

    if eurocodes is None:
        return None
    code = eurocodes.searched_code(text)
    if code is None:
        return None
    summary = eurocodes.summary(code)
    if summary is None or not eurocodes.identities_carrying(code):
        return EurocodeSearch(code)
    return EurocodeSearch(code, summary.label, summary.first_year, summary.last_year)

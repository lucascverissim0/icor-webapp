"""A Eurocode row's certain units, what they rest on, and what it only may share.

All codes and vehicles are fictional (`Q` prefix, which the real catalogue does
not use). In the demo map the A1 Horizon carries Q201 alone up to 2024, shares
2025 with Q202, and carries Q202 alone from 2026.
"""

from __future__ import annotations

from icor.domain.planner import DemandRange
from icor.eurocodes.demo import demo_eurocodes
from icor.eurocodes.grouping import EvidenceMix, RankedCell, rank_by_eurocode

IDENTITY = "aurora mobility|a1 horizon"


def _cell(
    year: int,
    units: int,
    evidence: str | None,
    *,
    identity: str = IDENTITY,
    geography: str = "DE",
    configuration: str = "c1",
) -> RankedCell:
    return RankedCell(
        identity=identity,
        year=year,
        geography=geography,
        raw_model=identity.split("|")[-1],
        brand="Aurora Mobility",
        model="A1 Horizon",
        horizon=2028,
        configuration=configuration,
        demand=DemandRange(units - 1, units, units + 2),
        coverage_status="uncovered",
        icor_worked=False,
        evidence=evidence,
    )


def test_certain_units_are_split_by_what_they_rest_on_and_add_up_to_demand() -> None:
    ranking = rank_by_eurocode(
        demo_eurocodes(),
        (
            _cell(2022, 30, "stock"),
            _cell(2023, 50, "registrations", geography="FR"),
            _cell(2024, 20, "forecast"),
            _cell(2024, 4, "interpolated", configuration="c2"),
            _cell(2021, 5, None),
        ),
    )

    totals = ranking.groups["Q201"]
    assert totals.group.evidence == EvidenceMix(
        stock=30, registrations=50, interpolated=4, forecast=20, unknown=5
    )
    assert totals.group.evidence.total == totals.demand.base_units == 109


def test_shared_and_unassigned_units_rest_on_nothing_of_a_code() -> None:
    ranking = rank_by_eurocode(
        demo_eurocodes(),
        (
            _cell(2024, 10, "stock"),
            _cell(2025, 40, "stock"),
            _cell(2027, 7, "forecast"),
            _cell(2024, 9, "stock", identity="nobody|reviewed"),
        ),
    )

    first, second = ranking.groups["Q201"], ranking.groups["Q202"]
    assert first.group.evidence == EvidenceMix(stock=10)
    assert first.group.shared_units == 40
    assert second.group.evidence == EvidenceMix(forecast=7)
    assert (ranking.totals.shared_units, ranking.totals.unreviewed_units) == (40, 9)


def test_a_code_with_only_shared_units_rests_on_nothing() -> None:
    ranking = rank_by_eurocode(demo_eurocodes(), (_cell(2025, 40, "registrations"),))

    for code in ("Q201", "Q202"):
        totals = ranking.groups[code]
        assert totals.demand.base_units == 0
        assert totals.group.evidence == EvidenceMix()
        assert totals.group.evidence.total == 0

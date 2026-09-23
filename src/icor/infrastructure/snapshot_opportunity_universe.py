"""The whole-market population a demand percentile is measured against.

A score must describe the vehicle, not the view. Until now the percentile was
computed by window functions that ran after the caller's market, horizon and
search clauses had already removed rows, so searching "golf" ranked a vehicle
against sixty-one others instead of against the market and every score on the
screen quietly changed meaning.

This module builds that population once — every group the snapshot holds at one
grouping level, with no clause of the caller's applied — and ships it to a
connection as a temp table the ranking can join. Building it costs seconds on a
nine-gigabyte snapshot and cannot be afforded per request; shipping it costs a
tenth of a second, which is a twentieth of what the identity table beside it
already costs.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from icor.application.opportunities import OpportunityGroupBy
from icor.application.ranking import WHOLE_MARKET_BASIS, DemandPopulation

TABLE = "opportunity_universe"


@dataclass(frozen=True, slots=True)
class UniverseGroup:
    """One vehicle group's standing in the market, before any filtering."""

    brand: str
    model: str | None
    model_year: int | None
    base_units: int
    percentile: float
    rank: int | None
    readiness_ratio: float


@dataclass(frozen=True, slots=True)
class OpportunityUniverse:
    group_by: OpportunityGroupBy
    needs_model_year: bool
    groups: tuple[UniverseGroup, ...] = field(repr=False)
    population: int
    basis: str = WHOLE_MARKET_BASIS

    @classmethod
    def read(
        cls,
        connection: sqlite3.Connection,
        grouped_cte: str,
        *,
        group_by: OpportunityGroupBy,
        needs_model_year: bool,
    ) -> OpportunityUniverse:
        """Group the whole snapshot at one level and rank what it holds.

        ``grouped_cte`` is the same common table expression the ranking itself
        uses, built with no filter, so the group keys here cannot disagree with
        the group keys there — including the canonical identity folding that
        makes ten spellings of Volkswagen one brand.
        """

        rows = connection.execute(
            f"""{grouped_cte}
            SELECT brand, model, model_year, base_units, exact_units,
                fallback_units
            FROM grouped"""
        ).fetchall()
        demand = DemandPopulation.of(int(row["base_units"]) for row in rows)
        groups = tuple(
            UniverseGroup(
                brand=row["brand"],
                model=row["model"],
                model_year=(
                    None if row["model_year"] is None else int(row["model_year"])
                ),
                base_units=int(row["base_units"]),
                percentile=demand.percentile(int(row["base_units"])),
                rank=demand.rank(int(row["base_units"])),
                readiness_ratio=_readiness_ratio(row),
            )
            for row in rows
        )
        return cls(
            group_by=group_by,
            needs_model_year=needs_model_year,
            groups=groups,
            population=demand.size,
        )

    def materialize(self, connection: sqlite3.Connection) -> None:
        """Put this population where SQL can join, sort and summarise on it.

        The index is not a ``WITHOUT ROWID`` primary key on purpose: ``model``
        and ``model_year`` are NULL at the brand and model levels, and SQLite
        makes every primary-key column of such a table implicitly NOT NULL, so
        the insert would fail.
        """

        connection.execute(
            f"""CREATE TEMP TABLE {TABLE} (
                brand TEXT NOT NULL,
                model TEXT,
                model_year INTEGER,
                universe_base_units INTEGER NOT NULL,
                universe_rank INTEGER,
                universe_percentile REAL NOT NULL,
                universe_readiness_ratio REAL NOT NULL
            )"""
        )
        connection.executemany(
            f"INSERT INTO temp.{TABLE} VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    group.brand,
                    group.model,
                    group.model_year,
                    group.base_units,
                    group.rank,
                    group.percentile,
                    group.readiness_ratio,
                )
                for group in self.groups
            ),
        )
        connection.execute(
            f"CREATE INDEX temp.{TABLE}_identity "
            f"ON {TABLE} (brand, model, model_year)"
        )


def _readiness_ratio(row: sqlite3.Row) -> float:
    base = int(row["base_units"])
    if base == 0:
        return 0.0
    exact = int(row["exact_units"])
    fallback = int(row["fallback_units"])
    return (exact + fallback * 0.5) / base

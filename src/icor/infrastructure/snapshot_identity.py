"""One canonical make/model identity per raw vehicle id, shared by both channels.

Model search already collapses the snapshot's publisher spellings — 557 makes to
54, 22,892 make/model pairs to 4,082 — through `VehicleIdentityIndex`. The
opportunity ranking read `canonical_vehicle` raw, so the same car appeared in it
once per spelling: `VOLKSWAGEN`, `VOLKSWAGEN VW`, `VOLKSWAGEB`, `VOLKSWAGWN` and
`volks wagen , vw` were five brands. Both channels now resolve through this
module, so they cannot disagree about which car a row describes.

The resolved identity is materialized into a temporary table rather than applied
in Python, because the ranking groups, ranks and paginates in SQL and a grouping
key the database cannot see would have to be applied after pagination.

A vehicle the index cannot fold keeps its own raw labels instead of being
dropped. Losing a row from the ranking is a worse failure than leaving a
duplicate in it, and it matches the merge asymmetry the index already follows:
under-merging leaves a duplicate in a list, over-merging sums two different
windshields into one forecast.
"""

from __future__ import annotations

import sqlite3

from icor.domain.evidence import STOCK_LOWER_BOUND_REASON
from icor.evidence.vehicle_identity import VehicleIdentityIndex

_LOWER_BOUND_CODE = f'"{STOCK_LOWER_BOUND_REASON}"'

_RANKABLE_VEHICLES = """FROM canonical_vehicle v
    JOIN (
        SELECT DISTINCT canonical_vehicle_id AS vehicle_id
        FROM opportunity_estimate
    ) o ON o.vehicle_id = v.vehicle_id"""

_VOLUME_JOIN = """LEFT JOIN (
        SELECT canonical_vehicle_id AS vehicle_id,
            SUM(CAST(registrations AS REAL)) AS registrations
        FROM cohort_estimate GROUP BY canonical_vehicle_id
    ) c ON c.vehicle_id = v.vehicle_id"""

_VOLUME_COLUMNS = frozenset({"canonical_vehicle_id", "registrations"})


def _has_volume(connection: sqlite3.Connection) -> bool:
    """Whether this snapshot's `cohort_estimate` can weight the vocabulary.

    Volume decides which labels are base models and which are trim, so the
    weighted form is always preferred. It is not required: a snapshot that
    cannot supply it still resolves identities, just without that ordering.
    Demanding a column the schema may not carry is what locked the application
    out of its own snapshot once already.
    """

    columns = {row[1] for row in connection.execute("PRAGMA table_info(cohort_estimate)")}
    return columns >= _VOLUME_COLUMNS


def load_identity_index(connection: sqlite3.Connection) -> VehicleIdentityIndex:
    """Build the canonical identity index from one snapshot's own labels.

    Both subqueries are aggregated before they meet, because joining
    `opportunity_estimate` straight onto `cohort_estimate` fans out to one row
    per opportunity-cohort pair and multiplies the registration volume the
    identity vocabulary is weighted by.
    """

    if _has_volume(connection):
        rows = (
            "SELECT v.vehicle_id, v.make, v.model, "
            "COALESCE(c.registrations, 0) "
            f"{_RANKABLE_VEHICLES} {_VOLUME_JOIN}"
        )
    else:
        rows = f"SELECT v.vehicle_id, v.make, v.model, 0 {_RANKABLE_VEHICLES}"
    fetched = connection.execute(rows).fetchall()
    return VehicleIdentityIndex.from_rows(
        fetched, vocabulary_volume=_vocabulary_volume(connection, fetched)
    )


def _vocabulary_volume(
    connection: sqlite3.Connection, rows: list[tuple[str, str, str, float]]
) -> dict[str, float] | None:
    """Volume that may teach the model vocabulary, without lower-bound stock.

    A register that masks names counts some labels alone (`SERIE` for many old
    BMWs); counted in, it raised the share a two-word name needs and taught
    the bare word, merging different cars. A vehicle known only from such a
    register teaches nothing. Without lower-bound cohorts this is every row's
    own volume, so other snapshots fold exactly as before.
    """

    if not _has_reason_codes(connection):
        return None
    lower_bound = {
        vehicle_id: (float(volume or 0.0), only_lower_bound)
        for vehicle_id, volume, only_lower_bound in connection.execute(
            """SELECT canonical_vehicle_id,
                SUM(CASE WHEN instr(reason_codes, ?) THEN 0
                    ELSE CAST(registrations AS REAL) END),
                MIN(instr(reason_codes, ?) > 0)
            FROM cohort_estimate GROUP BY canonical_vehicle_id
            HAVING MAX(instr(reason_codes, ?) > 0)""",
            (_LOWER_BOUND_CODE,) * 3,
        )
    }
    if not lower_bound:
        return None
    teaching: dict[str, float] = {}
    for vehicle_id, _make, _model, volume in rows:
        if vehicle_id not in lower_bound:
            teaching[vehicle_id] = float(volume or 0.0)
            continue
        own, only_lower_bound = lower_bound[vehicle_id]
        if not only_lower_bound:
            teaching[vehicle_id] = own
    return teaching


def _has_reason_codes(connection: sqlite3.Connection) -> bool:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(cohort_estimate)")}
    return {"canonical_vehicle_id", "registrations", "reason_codes"} <= columns


def materialize_identity_table(connection: sqlite3.Connection, index: VehicleIdentityIndex) -> None:
    """Write `temp.canonical_identity` for every vehicle the snapshot can rank."""

    connection.execute("DROP TABLE IF EXISTS temp.canonical_identity")
    connection.execute(
        """CREATE TEMP TABLE canonical_identity (
            vehicle_id TEXT PRIMARY KEY,
            brand TEXT NOT NULL,
            model TEXT NOT NULL
        )"""
    )
    rows = connection.execute(
        f"SELECT v.vehicle_id, v.make, v.model {_RANKABLE_VEHICLES}"
    ).fetchall()
    resolved: list[tuple[str, str, str]] = []
    for vehicle_id, make, model in rows:
        key = index.identity_for(vehicle_id)
        identity = index.identity(*key) if key is not None else None
        if identity is None:
            resolved.append((vehicle_id, make, model))
        else:
            resolved.append((vehicle_id, identity.display_make, identity.display_model))
    connection.executemany(
        "INSERT INTO canonical_identity (vehicle_id, brand, model) VALUES (?, ?, ?)",
        resolved,
    )

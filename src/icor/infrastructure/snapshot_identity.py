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

from icor.evidence.vehicle_identity import VehicleIdentityIndex

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

    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(cohort_estimate)")
    }
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
    return VehicleIdentityIndex.from_rows(connection.execute(rows).fetchall())


def materialize_identity_table(
    connection: sqlite3.Connection, index: VehicleIdentityIndex
) -> None:
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
            resolved.append(
                (vehicle_id, identity.display_make, identity.display_model)
            )
    connection.executemany(
        "INSERT INTO canonical_identity (vehicle_id, brand, model) VALUES (?, ?, ?)",
        resolved,
    )

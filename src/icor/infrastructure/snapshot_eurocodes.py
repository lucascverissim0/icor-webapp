"""What the running app can claim about its Eurocodes on the active snapshot.

The decisions were reviewed for one snapshot and one version of the identity
rules. Serving them against another is allowed but said out loud: the warnings
name the mismatch, and orphaned identities (reviewed, no longer present) are
counted. Coverage is measured the way the review measured it, by allocating
every registration cell, so the share the app reports is the share it serves.

Measuring reads every forecast cell once, a few seconds on a full snapshot, so
it runs once per process and is cached; the application starts it in the
background at boot.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from icor.eurocodes.runtime import EurocodeMap, EurocodeState, identity_key
from icor.evidence.vehicle_identity import identity_rules_sha256
from icor.infrastructure.snapshot_identity import load_identity_index
from icor.infrastructure.snapshot_planner_repository import _cohort_attribution_ctes


@dataclass(frozen=True, slots=True)
class EurocodeCoverage:
    """One forecast horizon's base units by allocation outcome."""

    horizon: int
    total_units: int
    single_units: int
    shared_units: int
    outside_units: int
    unassignable_units: int
    unreviewed_units: int
    #: Part of the shared units: those that may be a vehicle without a code.
    shared_with_uncoded_units: int


@dataclass(frozen=True, slots=True)
class EurocodeStatus:
    state: EurocodeState
    coverage: tuple[EurocodeCoverage, ...] = ()


def measure(
    eurocodes: EurocodeMap, connection: sqlite3.Connection, snapshot_id: str
) -> EurocodeStatus:
    """Allocate every forecast cell of the snapshot and count the outcomes."""

    index = load_identity_index(connection)
    labels = {
        vehicle_id: (make, model)
        for vehicle_id, make, model in connection.execute(
            "SELECT vehicle_id, make, model FROM canonical_vehicle "
            "WHERE vehicle_id IN (SELECT canonical_vehicle_id FROM opportunity_estimate)"
        )
    }
    keys = {
        vehicle_id: identity_key(index, vehicle_id, make, model)
        for vehicle_id, (make, model) in labels.items()
    }
    materialized = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'opportunity_cohort_attribution'"
    ).fetchone()
    prefix, attribution = (
        ("", "opportunity_cohort_attribution")
        if materialized
        else (f"WITH {_cohort_attribution_ctes('1 = 1')} ", "cohort_attribution")
    )
    totals: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for horizon, vehicle_id, geography, year, units in connection.execute(
        f"""{prefix}SELECT o.horizon_year, o.canonical_vehicle_id, o.geography,
            a.registration_cohort_year, SUM(a.base_units)
        FROM {attribution} a
        JOIN opportunity_estimate o ON o.opportunity_id = a.opportunity_id
        GROUP BY 1, 2, 3, 4"""
    ):
        if not units:
            continue
        allocation = eurocodes.allocate(
            keys[vehicle_id], int(year), geography, raw_model=labels[vehicle_id][1]
        )
        counts = totals[int(horizon)]
        counts[allocation.status] += int(units)
        counts["total"] += int(units)
        if allocation.uncoded:
            counts["uncoded"] += int(units)
    coverage = tuple(
        EurocodeCoverage(
            horizon=horizon,
            total_units=counts["total"],
            single_units=counts["single"],
            shared_units=counts["shared"],
            outside_units=counts["outside"],
            unassignable_units=counts["unassignable"],
            unreviewed_units=counts["unreviewed"],
            shared_with_uncoded_units=counts["uncoded"],
        )
        for horizon, counts in sorted(totals.items())
    )
    state = eurocodes.state(
        snapshot_id=snapshot_id,
        identity_rules_sha256=identity_rules_sha256(),
        present_identities=set(keys.values()),
    )
    return EurocodeStatus(state, coverage)


class SnapshotEurocodeStatus:
    """The active snapshot's Eurocode status, measured once and then cached."""

    def __init__(self, eurocodes: EurocodeMap, snapshot_path: Path, snapshot_id: str) -> None:
        self._eurocodes = eurocodes
        self._snapshot_path = snapshot_path
        self._snapshot_id = snapshot_id
        self._lock = threading.Lock()
        self._status: EurocodeStatus | None = None

    def status(self) -> EurocodeStatus:
        with self._lock:
            if self._status is None:
                connection = sqlite3.connect(
                    f"{self._snapshot_path.resolve().as_uri()}?mode=ro", uri=True
                )
                try:
                    self._status = measure(self._eurocodes, connection, self._snapshot_id)
                finally:
                    connection.close()
            return self._status

    def warm_in_background(self) -> None:
        threading.Thread(target=self.status, name="eurocode-status", daemon=True).start()

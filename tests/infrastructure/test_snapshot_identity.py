"""The identity vocabulary a snapshot teaches, with and without lower-bound stock."""

from __future__ import annotations

import json
import sqlite3

from icor.infrastructure.snapshot_identity import load_identity_index

_SCHEMA = """
CREATE TABLE canonical_vehicle (vehicle_id TEXT, make TEXT, model TEXT);
CREATE TABLE opportunity_estimate (canonical_vehicle_id TEXT);
CREATE TABLE cohort_estimate (
    canonical_vehicle_id TEXT, registrations TEXT, reason_codes TEXT);
"""
_REGISTERED = json.dumps(["reconciled-registration-cohort", "survival"])
_LOWER_BOUND = json.dumps(
    ["observed-stock-anchored-cohort", "survival", "observed-stock-lower-bound"]
)


def _snapshot(cohorts: list[tuple[str, str, str]]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(_SCHEMA)
    vehicles = [("x5", "BMW", "X5"), ("s3", "BMW", "SERIE 3"), ("serie", "BMW", "SERIE")]
    connection.executemany("INSERT INTO canonical_vehicle VALUES (?, ?, ?)", vehicles)
    connection.executemany(
        "INSERT INTO opportunity_estimate VALUES (?)", [(item[0],) for item in vehicles]
    )
    connection.executemany("INSERT INTO cohort_estimate VALUES (?, ?, ?)", cohorts)
    return connection


def test_a_vehicle_known_only_from_lower_bound_stock_does_not_teach_the_vocabulary():
    index = load_identity_index(
        _snapshot(
            [
                ("x5", "60000", _REGISTERED),
                ("x5", "1000", _LOWER_BOUND),
                ("s3", "400", _REGISTERED),
                ("serie", "40000", _LOWER_BOUND),
            ]
        )
    )

    assert index.identity_for("s3") == ("bmw", "serie 3")
    assert index.identity_for("serie") == ("bmw", "serie")


def test_registered_volume_still_teaches_as_before():
    index = load_identity_index(
        _snapshot(
            [
                ("x5", "60000", _REGISTERED),
                ("s3", "400", _REGISTERED),
                ("serie", "40000", _REGISTERED),
            ]
        )
    )

    assert index.identity_for("s3") == ("bmw", "serie")


_UNTAUGHT = json.dumps(
    ["observed-stock-anchored-cohort", "survival", "observed-stock-labels-do-not-teach-vocabulary"]
)


def test_a_vehicle_known_only_from_an_untaught_register_does_not_teach_the_vocabulary():
    index = load_identity_index(
        _snapshot(
            [
                ("x5", "60000", _REGISTERED),
                ("s3", "400", _REGISTERED),
                ("serie", "40000", _UNTAUGHT),
                ("x5", "9000", _UNTAUGHT),
            ]
        )
    )

    assert index.identity_for("s3") == ("bmw", "serie 3")
    assert index.identity_for("serie") == ("bmw", "serie")


_ARCHIVED = json.dumps(
    [
        "reconciled-registration-cohort",
        "survival",
        "archive-registrations-not-forecast",
        "registrations-labels-do-not-teach-vocabulary",
    ]
)


def test_archived_registrations_whose_labels_do_not_teach_change_no_fold():
    index = load_identity_index(
        _snapshot(
            [
                ("x5", "60000", _REGISTERED),
                ("s3", "400", _REGISTERED),
                ("serie", "40000", _ARCHIVED),
                ("x5", "9000", _ARCHIVED),
            ]
        )
    )

    assert index.identity_for("s3") == ("bmw", "serie 3")
    assert index.identity_for("serie") == ("bmw", "serie")


def test_extra_vehicles_fold_onto_the_vocabulary_without_teaching_it():
    cohorts = [
        ("x5", "60000", _REGISTERED),
        ("s3", "400", _REGISTERED),
        ("serie", "40000", _LOWER_BOUND),
    ]
    extras = [("dk-x5", "BMW", "X5"), ("dk-serie", "BMW", "SERIE"), ("x5", "BMW", "IGNORED")]

    index = load_identity_index(_snapshot(cohorts), extra_vehicles=extras)

    # Every ranked vehicle folds as it does without extras.
    assert index.identity_for("s3") == ("bmw", "serie 3")
    assert index.identity_for("x5") == ("bmw", "x5")
    # The extras take the existing identities; one already ranked keeps its own labels.
    assert index.identity_for("dk-x5") == ("bmw", "x5")
    assert index.identity_for("dk-serie") == ("bmw", "serie")

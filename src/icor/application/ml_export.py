"""Deterministic, temporally safe training export from one snapshot."""

from __future__ import annotations

import csv
import sqlite3
from datetime import date
from io import StringIO
from pathlib import Path


class MLExportService:
    def __init__(self, repository, snapshot_id: str) -> None:
        self._repository = repository
        self.snapshot_id = snapshot_id

    def render_csv(self, cutoff: date) -> str:
        path = getattr(self._repository, "path", None)
        if isinstance(path, Path):
            return self._render_sqlite(path, cutoff)
        releases = {
            item.release_id: item for item in self._repository.list_releases()
        }
        vehicles = {
            item.vehicle_id: item for item in self._repository.list_vehicles()
        }
        generations = {
            item.generation_id: item for item in self._repository.list_generations()
        }
        assignments = {
            item.observation_id: item
            for item in self._repository.list_generation_assignments()
        }
        output = StringIO(newline="")
        writer = csv.writer(output, lineterminator="\r\n")
        _write_header(writer)
        for observation in sorted(
            self._repository.list_observations(), key=lambda item: item.observation_id
        ):
            release = releases[observation.release_id]
            assignment = assignments.get(observation.observation_id)
            if (
                assignment is None
                or observation.canonical_vehicle_id is None
                or release.published_at.date() > cutoff
                or observation.period_end > cutoff
            ):
                continue
            generation = generations[assignment.selected_generation_id]
            vehicle = vehicles[observation.canonical_vehicle_id]
            writer.writerow(
                (
                    observation.observation_id,
                    self.snapshot_id,
                    observation.release_id,
                    observation.geography,
                    observation.registration_cohort_year or observation.period_end.year,
                    vehicle.make,
                    vehicle.model,
                    generation.generation_id,
                    generation.identity_kind.value,
                    observation.measure.value,
                    str(observation.value),
                    assignment.confidence.value,
                    str(assignment.training_weight),
                    assignment.resolver_version,
                    assignment.registry_version,
                )
            )
        return output.getvalue()

    def _render_sqlite(self, path: Path, cutoff: date) -> str:
        output = StringIO(newline="")
        writer = csv.writer(output, lineterminator="\r\n")
        _write_header(writer)
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """SELECT observation.observation_id, observation.release_id,
                observation.geography, observation.registration_cohort_year,
                observation.period_end, observation.measure, observation.value,
                vehicle.make, vehicle.model, assignment.selected_generation_id,
                generation.identity_kind, assignment.confidence,
                assignment.training_weight, assignment.resolver_version,
                assignment.registry_version
                FROM observation
                JOIN source_release release
                    ON release.release_id = observation.release_id
                JOIN generation_assignment assignment
                    ON assignment.observation_id = observation.observation_id
                JOIN generation_entry generation
                    ON generation.generation_id = assignment.selected_generation_id
                JOIN canonical_vehicle vehicle
                    ON vehicle.vehicle_id = observation.canonical_vehicle_id
                WHERE SUBSTR(release.published_at, 1, 10) <= ?
                AND observation.period_end <= ?
                ORDER BY observation.observation_id""",
                (cutoff.isoformat(), cutoff.isoformat()),
            )
            for row in rows:
                writer.writerow(
                    (
                        row["observation_id"], self.snapshot_id, row["release_id"],
                        row["geography"],
                        row["registration_cohort_year"]
                        or int(row["period_end"][:4]),
                        row["make"], row["model"],
                        row["selected_generation_id"], row["identity_kind"],
                        row["measure"], row["value"], row["confidence"],
                        row["training_weight"], row["resolver_version"],
                        row["registry_version"],
                    )
                )
        finally:
            connection.close()
        return output.getvalue()


def _write_header(writer) -> None:
    writer.writerow(
        (
            "observation_id", "snapshot_id", "release_id", "geography",
            "registration_cohort_year", "make", "model", "generation_id",
            "generation_identity_kind", "measure", "value", "confidence",
            "training_weight", "resolver_version", "registry_version",
        )
    )

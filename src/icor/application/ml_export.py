"""Deterministic, temporally safe training export from one snapshot."""

from __future__ import annotations

import csv
from datetime import date
from io import StringIO


class MLExportService:
    def __init__(self, repository, snapshot_id: str) -> None:
        self._repository = repository
        self.snapshot_id = snapshot_id

    def render_csv(self, cutoff: date) -> str:
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
        writer.writerow(
            (
                "observation_id", "snapshot_id", "release_id", "geography",
                "registration_cohort_year", "make", "model", "generation_id",
                "generation_identity_kind", "measure", "value", "confidence",
                "training_weight", "resolver_version", "registry_version",
            )
        )
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

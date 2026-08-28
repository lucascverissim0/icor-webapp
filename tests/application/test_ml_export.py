from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace

from icor.application.ml_export import MLExportService


class Ledger:
    def list_releases(self):
        return (
            SimpleNamespace(
                release_id="release-known",
                published_at=datetime(2024, 6, 1, tzinfo=UTC),
            ),
            SimpleNamespace(
                release_id="release-future",
                published_at=datetime(2025, 6, 1, tzinfo=UTC),
            ),
        )

    def list_vehicles(self):
        return (
            SimpleNamespace(vehicle_id="vehicle-alpha", make="Example", model="Alpha"),
        )

    def list_generations(self):
        return (
            SimpleNamespace(
                generation_id="generation-alpha",
                canonical_vehicle_id="vehicle-alpha",
                identity_kind=SimpleNamespace(value="estimated"),
            ),
        )

    def list_generation_assignments(self):
        return (
            SimpleNamespace(
                observation_id="observation-known",
                selected_generation_id="generation-alpha",
                confidence=SimpleNamespace(value="low"),
                training_weight=Decimal("0.5"),
                resolver_version="resolver-v1",
                registry_version="registry-v1",
            ),
            SimpleNamespace(
                observation_id="observation-future",
                selected_generation_id="generation-alpha",
                confidence=SimpleNamespace(value="low"),
                training_weight=Decimal("0.5"),
                resolver_version="resolver-v1",
                registry_version="registry-v1",
            ),
        )

    def list_observations(self):
        return (
            SimpleNamespace(
                observation_id="observation-known",
                release_id="release-known",
                canonical_vehicle_id="vehicle-alpha",
                geography="DE",
                registration_cohort_year=2023,
                period_end=date(2023, 12, 31),
                measure=SimpleNamespace(value="new_registrations"),
                value=Decimal("100"),
            ),
            SimpleNamespace(
                observation_id="observation-future",
                release_id="release-future",
                canonical_vehicle_id="vehicle-alpha",
                geography="DE",
                registration_cohort_year=2024,
                period_end=date(2024, 12, 31),
                measure=SimpleNamespace(value="new_registrations"),
                value=Decimal("120"),
            ),
        )


def test_export_excludes_evidence_not_known_at_cutoff() -> None:
    result = MLExportService(Ledger(), "snapshot-real-v1").render_csv(
        cutoff=date(2024, 12, 31)
    )

    assert "observation-known" in result
    assert "observation-future" not in result
    assert result.startswith("observation_id,snapshot_id,release_id")
    assert result.endswith("\r\n")


def test_export_is_deterministic() -> None:
    service = MLExportService(Ledger(), "snapshot-real-v1")

    assert service.render_csv(date(2026, 1, 1)) == service.render_csv(
        date(2026, 1, 1)
    )

"""Reviewed production composition for official evidence source adapters."""

from __future__ import annotations

from collections.abc import Mapping

from icor.application.snapshot_build import EvidenceLoader
from icor.domain.snapshots import SnapshotVersions
from icor.evidence.sources.eea import EEAPassengerCarLoader
from icor.evidence.sources.kba import KBAFZ10Loader
from icor.evidence.sources.uk_dft import UKActiveFleetLoader, UKFirstRegistrationLoader

OFFICIAL_SOURCE_VERSIONS = SnapshotVersions(
    source_registry="official-sources-v1",
    identity_registry="unresolved-source-labels-v1",
    reconciliation_method="not-applied-v1",
    confidence_method="source-evidence-v1",
    estimation_method="not-applied-v1",
    survival_method="not-applied-v1",
    hazard_method="not-applied-v1",
    forecast_method="not-applied-v1",
)


def official_loader_registry() -> Mapping[str, EvidenceLoader]:
    """Return a fresh fail-closed map of parser names reviewed for local use."""

    return {
        "eea_co2_cars_zip_v1": EEAPassengerCarLoader(),
        "kba_fz10_xlsx_v1": KBAFZ10Loader(),
        "uk_dft_veh0120_csv_v1": UKActiveFleetLoader(),
        "uk_dft_veh0160_csv_v1": UKFirstRegistrationLoader(),
    }

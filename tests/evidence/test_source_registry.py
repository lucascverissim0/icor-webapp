from icor.evidence.source_registry import OFFICIAL_SOURCE_VERSIONS, official_loader_registry
from icor.evidence.sources.eea import EEAAnnualAggregateLoader, EEAPassengerCarLoader
from icor.evidence.sources.kba import KBAFZ10Loader
from icor.evidence.sources.uk_dft import (
    UKActiveFleetLoader,
    UKFirstRegistrationLoader,
    UKVehicleAgeLoader,
)


def test_official_registry_contains_only_reviewed_parser_contracts() -> None:
    registry = official_loader_registry()

    assert set(registry) == {
        "eea_co2_cars_zip_v1",
        "eea_co2_cars_annual_aggregate_csv_v1",
        "kba_fz10_xlsx_v1",
        "uk_dft_veh0120_csv_v1",
        "uk_dft_veh0160_csv_v1",
        "uk_dft_veh0124_csv_v1",
    }
    assert isinstance(registry["eea_co2_cars_zip_v1"], EEAPassengerCarLoader)
    assert isinstance(
        registry["eea_co2_cars_annual_aggregate_csv_v1"], EEAAnnualAggregateLoader
    )
    assert isinstance(registry["kba_fz10_xlsx_v1"], KBAFZ10Loader)
    assert isinstance(registry["uk_dft_veh0120_csv_v1"], UKActiveFleetLoader)
    assert isinstance(registry["uk_dft_veh0160_csv_v1"], UKFirstRegistrationLoader)
    assert isinstance(registry["uk_dft_veh0124_csv_v1"], UKVehicleAgeLoader)


def test_official_source_snapshot_versions_do_not_claim_unimplemented_methods() -> None:
    assert OFFICIAL_SOURCE_VERSIONS.source_registry == "official-sources-v1"
    assert OFFICIAL_SOURCE_VERSIONS.identity_registry == "exact-normalized-model-family-v1"
    assert OFFICIAL_SOURCE_VERSIONS.generation_registry == "generation-registry-v1"
    assert OFFICIAL_SOURCE_VERSIONS.generation_resolver == "generation-resolver-v1"
    assert OFFICIAL_SOURCE_VERSIONS.reconciliation_method == "not-applied-v1"
    assert OFFICIAL_SOURCE_VERSIONS.estimation_method == "not-applied-v1"
    assert OFFICIAL_SOURCE_VERSIONS.forecast_method == "not-applied-v1"

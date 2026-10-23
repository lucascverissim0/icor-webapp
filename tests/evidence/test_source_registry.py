from icor.evidence.source_registry import OFFICIAL_SOURCE_VERSIONS, official_loader_registry
from icor.evidence.sources.dgt import DGTParqueLoader
from icor.evidence.sources.eea import EEAAnnualAggregateLoader, EEAPassengerCarLoader
from icor.evidence.sources.kba import KBAFZ10Loader
from icor.evidence.sources.kba_archive import KBAFZ10ArchiveLoader
from icor.evidence.sources.national_register import NationalRegisterLoader
from icor.evidence.sources.rdw import RDWRegisteredCarsLoader
from icor.evidence.sources.traficom import TraficomVehicleLoader
from icor.evidence.sources.uk_dft import (
    UKActiveFleetLoader,
    UKFirstRegistrationLoader,
    UKVehicleAgeLoader,
)
from icor.forecasting.survival import load_age_anchored_survival_model


def test_official_registry_contains_only_reviewed_parser_contracts() -> None:
    registry = official_loader_registry()

    assert set(registry) == {
        "eea_co2_cars_zip_v1",
        "eea_co2_cars_annual_aggregate_csv_v1",
        "kba_fz10_xlsx_v1",
        "kba_fz10_xls_v1",
        "uk_dft_veh0120_csv_v1",
        "uk_dft_veh0160_csv_v1",
        "uk_dft_veh0124_csv_v1",
        "rdw_registered_cars_csv_v1",
        "traficom_vehicles_zip_v1",
        "dgt_parque_cohorts_csv_v1",
        "national_register_cohorts_csv_v1",
    }
    assert isinstance(registry["eea_co2_cars_zip_v1"], EEAPassengerCarLoader)
    assert isinstance(
        registry["eea_co2_cars_annual_aggregate_csv_v1"], EEAAnnualAggregateLoader
    )
    assert isinstance(registry["kba_fz10_xlsx_v1"], KBAFZ10Loader)
    assert isinstance(registry["kba_fz10_xls_v1"], KBAFZ10ArchiveLoader)
    assert isinstance(registry["uk_dft_veh0120_csv_v1"], UKActiveFleetLoader)
    assert isinstance(registry["uk_dft_veh0160_csv_v1"], UKFirstRegistrationLoader)
    assert isinstance(registry["uk_dft_veh0124_csv_v1"], UKVehicleAgeLoader)
    assert isinstance(registry["rdw_registered_cars_csv_v1"], RDWRegisteredCarsLoader)
    assert isinstance(registry["traficom_vehicles_zip_v1"], TraficomVehicleLoader)
    assert isinstance(registry["dgt_parque_cohorts_csv_v1"], DGTParqueLoader)
    assert isinstance(registry["national_register_cohorts_csv_v1"], NationalRegisterLoader)


def test_official_source_snapshot_versions_do_not_claim_unimplemented_methods() -> None:
    assert OFFICIAL_SOURCE_VERSIONS.source_registry == "official-sources-v3"
    assert OFFICIAL_SOURCE_VERSIONS.identity_registry == "exact-normalized-model-family-v1"
    assert (
        OFFICIAL_SOURCE_VERSIONS.generation_registry
        == "public-generation-registry-v2"
    )
    assert OFFICIAL_SOURCE_VERSIONS.generation_resolver == "generation-resolver-v1"
    assert (
        OFFICIAL_SOURCE_VERSIONS.reconciliation_method
        == "single-coverage-corroboration-v2"
    )
    assert (
        OFFICIAL_SOURCE_VERSIONS.estimation_method
        == "register-years-archive-stock-anchored-cohorts-v6"
    )
    assert (
        OFFICIAL_SOURCE_VERSIONS.survival_method
        == "uk-dft-licensed-stock-band-v1+eurostat-net-retention-v1+eurostat-age-anchor-v1"
        == load_age_anchored_survival_model().method
    )
    assert (
        OFFICIAL_SOURCE_VERSIONS.hazard_method
        == "france-insurance-windshield-hazard-v2"
    )
    assert (
        OFFICIAL_SOURCE_VERSIONS.forecast_method
        == "validated-recency-damped-ensemble-v2"
    )

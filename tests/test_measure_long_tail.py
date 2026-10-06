"""The long-tail report counts a car once, at its full size, whatever its spelling."""

from __future__ import annotations

from scripts.measure_long_tail import summarize


def test_spellings_fold_into_one_vehicle_and_lost_volume_is_reported() -> None:
    vehicles = [
        ("golf-a", "VOLKSWAGEN", "GOLF"),
        ("golf-b", "VW", "Golf"),
        ("tonale", "ALFA ROMEO", "TONALE"),
        ("mito", "ALFA ROMEO", "MITO"),
    ]
    volume = {"golf-a": 900.0, "golf-b": 100.0, "tonale": 40.0, "mito": 10.0}
    # Only one Golf spelling and the Tonale were forecast.
    demand = {"golf-a": 50.0, "tonale": 0.0}

    report = summarize(vehicles, volume, demand)

    assert report["identities_observed"] == 3
    assert report["registrations_observed"] == 1050
    # 900 of the Golf's 1,000 and all 40 of the Tonale's are forecast.
    assert report["forecast_registration_share"] == round(940 / 1050, 4)
    deciles = {item["max_registrations"]: item for item in report["deciles"]}
    assert deciles[1000]["forecast_registration_share"] == 0.9
    assert deciles[1000]["ranked_identity_share"] == 1.0
    assert deciles[40]["ranked_identity_share"] == 0.0
    assert deciles[10]["forecast_identity_share"] == 0.0
    panel = {item["vehicle"]: item for item in report["tail_panel"]}
    assert panel["Alfa Romeo Tonale"]["observed"] is True
    assert panel["Alfa Romeo Tonale"]["searchable_with_forecast"] is True
    assert panel["Alfa Romeo MiTo"]["observed"] is True
    assert panel["Alfa Romeo MiTo"]["searchable_with_forecast"] is False
    assert panel["Saab 9-3"]["observed"] is False

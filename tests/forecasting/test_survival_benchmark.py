from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.benchmark_survival_calibration import _load_registrations


def _write_dft_csv(path: Path, *, replacement: str | None = None) -> None:
    quarters = [f"{year} Q{quarter}" for year in range(2015, 2026) for quarter in range(1, 5)]
    fieldnames = ["BodyType", "Make", *quarters]
    row = {field: "1" for field in quarters}
    row.update({"BodyType": "Cars", "Make": "CITROËN"})
    if replacement is not None:
        row["2020 Q2"] = replacement
    with path.open("w", encoding="cp1252", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow({"BodyType": "Motorcycles", "Make": "OTHER", **{q: "99" for q in quarters}})


def test_registration_loader_preserves_cp1252_and_sums_only_cars(tmp_path: Path) -> None:
    path = tmp_path / "df_VEH0160_UK.csv"
    _write_dft_csv(path)

    registrations = _load_registrations(path)

    assert len(registrations) == 11
    assert registrations[0].cohort_year == 2015
    assert registrations[-1].cohort_year == 2025
    assert {item.registrations for item in registrations} == {Decimal("4")}


def test_registration_loader_rejects_suppressed_values(tmp_path: Path) -> None:
    path = tmp_path / "df_VEH0160_UK.csv"
    _write_dft_csv(path, replacement="[c]")

    with pytest.raises(ValueError, match="non-numeric UK registration value"):
        _load_registrations(path)

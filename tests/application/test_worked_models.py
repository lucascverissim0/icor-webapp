from pathlib import Path

import pytest

from icor.application.worked_models import IcorWorkedModelCatalog

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_catalog_preserves_recorded_model_years_without_extrapolation() -> None:
    catalog = IcorWorkedModelCatalog.from_path(ROOT / "data" / "icor_supported_models.txt")

    assert catalog.matches("Volkswagen", "Golf", 2020) is True
    assert catalog.matches("VOLKSWAGEN VW", "GOLF", 2020) is True
    assert catalog.matches("Volkswagen", "Golf", 2021) is False
    assert catalog.matches("Ford", "Kuga", 2012) is True
    assert catalog.matches("Ford", "Kuga", 2020) is True


def test_legacy_catalog_rejects_unparseable_nonblank_rows(tmp_path: Path) -> None:
    source = tmp_path / "worked.txt"
    source.write_text('{\n  "vw golf": {2020: "G8"},\n  broken\n}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 3"):
        IcorWorkedModelCatalog.from_path(source)

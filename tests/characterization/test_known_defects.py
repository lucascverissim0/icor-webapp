import re
from pathlib import Path

import pytest

from source_loader import load_function


ROOT = Path(__file__).resolve().parents[2]
SCRIPT1 = ROOT / "scripts" / "script1.py"


@pytest.mark.xfail(strict=True, reason="ICOR-001: Script 1 cannot parse the catalog mapping")
def test_icor_001_supported_audi_a3_is_recognized() -> None:
    parser = load_function(SCRIPT1, "parse_icor_models", {"re": re})
    catalog = (ROOT / "data" / "icor_supported_models.txt").read_text(encoding="utf-8")
    assert "Audi A3" in parser(catalog)


@pytest.mark.xfail(strict=True, reason="ICOR-006: generated filenames retain path separators")
def test_icor_006_model_filename_has_no_separator() -> None:
    generated = "Nissan X-Trail / Rogue".replace(" ", "_")
    assert "/" not in generated and "\\" not in generated


@pytest.mark.xfail(strict=True, reason="ICOR-009: NaN wins over a valid regional value")
def test_icor_009_valid_year_count_wins_over_nan() -> None:
    assert max(float("nan") or 0, 5 or 0) == 5


@pytest.mark.xfail(strict=True, reason="ICOR-030: fleet survival conventions disagree")
def test_icor_030_one_year_old_cohort_has_one_survival_value() -> None:
    decay = 0.0556
    script1_survival = (1 - decay) ** 1
    script2_survival = 1.0
    assert script1_survival == script2_survival

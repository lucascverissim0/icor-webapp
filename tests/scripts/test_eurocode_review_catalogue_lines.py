import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eurocode_review.py"


def _module():
    spec = importlib.util.spec_from_file_location("eurocode_review_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(number, start, end):
    return SimpleNamespace(
        row=number,
        makes=("suzuki",),
        start=start,
        end=end,
        flags=(),
        brand="SUZUKI",
        description="model",
        sop=None,
        eop=None,
        scope="",
    )


def test_old_open_ended_and_recently_ended_codes_are_listed_but_long_ended_ones_are_not():
    catalogue = SimpleNamespace(
        codes=lambda: {
            "OPEN": [_row(1, 1982, None)],
            "ENDED1986": [_row(2, 1980, 1986)],
            "ENDED1975": [_row(3, 1970, 1975)],
        }
    )

    lines = _module()._catalogue_lines(catalogue, {"suzuki"})

    listed = {line.split(" | ")[1] for line in lines}
    assert listed == {"OPEN", "ENDED1986"}

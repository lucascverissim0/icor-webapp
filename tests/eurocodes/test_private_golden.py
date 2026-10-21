"""The real reviewed Eurocodes against hand-verified expectations (ICOR-private).

The catalogue, the decisions and the golden cases are ICOR-private and never in
the repository, so this test reads them from the private folder and skips where
they are absent (CI, other machines). No code or catalogue text appears here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from icor.eurocodes.runtime import EurocodeMap

PRIVATE = Path(
    os.environ.get(
        "ICOR_EUROCODE_PRIVATE",
        Path(__file__).resolve().parents[2] / ".local" / "private" / "eurocode",
    )
)
CATALOGUE = PRIVATE / "catalog.json"
DECISIONS = PRIVATE / "decisions.json"
GOLDEN = PRIVATE / "golden.json"

pytestmark = pytest.mark.skipif(
    not (CATALOGUE.is_file() and DECISIONS.is_file() and GOLDEN.is_file()),
    reason="ICOR-private Eurocode files are not on this machine",
)


@pytest.fixture(scope="module")
def eurocodes() -> EurocodeMap:
    return EurocodeMap.from_paths(CATALOGUE, DECISIONS)


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    document = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert document["format"] == "icor-eurocode-golden"
    return document["cases"]


def test_the_golden_set_is_large_enough_to_mean_something(cases: list[dict]) -> None:
    assert len(cases) >= 100
    assert len({case["identity"] for case in cases}) >= 50


def test_every_golden_case_allocates_as_verified(eurocodes: EurocodeMap, cases: list[dict]) -> None:
    mismatches = []
    for case in cases:
        allocation = eurocodes.decisions.allocate(
            case["identity"], case["year"], case["geography"], raw_model=case["raw_model"]
        )
        actual = (sorted(allocation.codes), allocation.status, allocation.uncoded)
        expected = (sorted(case["codes"]), case["status"], case.get("uncoded", False))
        if actual != expected:
            mismatches.append(
                f"{case['identity']} {case['year']} {case['geography']} {case['raw_model']!r}: "
                f"expected {expected}, got {actual} ({case['why']})"
            )
    assert not mismatches, "\n".join(mismatches)

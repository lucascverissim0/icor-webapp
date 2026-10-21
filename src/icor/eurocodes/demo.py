"""A fictional Eurocode map for the demo planner data and the browser tests.

Every code uses the ``Q`` prefix, which ICOR's catalogue does not use, and every
vehicle is one of the demo data's invented brands, so nothing here is ICOR data.
The map exercises each outcome a row can show: a single code, a launch span
shared by two generations, a year before the first code (outside), and a vehicle
nobody reviewed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from icor.eurocodes.catalog import parse_rows, to_document
from icor.eurocodes.runtime import EurocodeMap

_SHA256 = "0" * 64

_ROWS = (
    ("Q201", "AURORA MOBILITY", "A1 HORIZON 5D HBK 2020-2025", 2025, 2020),
    ("Q202", "AURORA MOBILITY", "A1 HORIZON II 5D HBK 2025-", None, 2025),
    ("Q301", "MERIDIAN MOTORS", "M5 VECTOR SUV 2024-", None, 2024),
    ("Q401", "VELORA WORKS", "V3 CURRENT 5D HBK 2024-", None, 2024),
)


def _decisions() -> dict[str, Any]:
    return {
        "format": "icor-eurocode-decisions",
        "version": 1,
        "metadata": {
            "catalogue_sha256": _SHA256,
            "snapshot_id": "demo",
            "identity_rules_sha256": "demo",
        },
        "families": {
            "aurora/a1-horizon": {
                "label": "Aurora Mobility A1 Horizon",
                "make": "aurora mobility",
                "codes": [
                    {
                        "code": "Q201",
                        "label": "A1 Horizon I",
                        "first_year": 2020,
                        "last_year": 2025,
                        "rows": [2],
                    },
                    {
                        "code": "Q202",
                        "label": "A1 Horizon II",
                        "first_year": 2025,
                        "first_year_latest": 2026,
                        "replaces": ["Q201"],
                        "rows": [3],
                    },
                ],
            },
            "meridian/m5-vector": {
                "label": "Meridian Motors M5 Vector",
                "make": "meridian motors",
                "codes": [{"code": "Q301", "label": "M5 Vector", "first_year": 2024, "rows": [4]}],
            },
            "velora/v3-current": {
                "label": "Velora Works V3 Current",
                "make": "velora works",
                "codes": [{"code": "Q401", "label": "V3 Current", "first_year": 2024, "rows": [5]}],
            },
        },
        "identities": {
            "aurora mobility|a1 horizon": {"family": "aurora/a1-horizon"},
            "meridian motors|m5 vector": {"family": "meridian/m5-vector"},
            "velora works|v3 current": {"family": "velora/v3-current"},
        },
    }


def demo_eurocodes() -> EurocodeMap:
    catalogue = parse_rows(_ROWS, make_aliases={}, source_sha256=_SHA256)
    return EurocodeMap.from_documents(catalogue, _decisions())


def write_demo_eurocodes(directory: Path) -> Path:
    """Write the demo map as the two files a process loads with ``ICOR_EUROCODE_DIR``."""

    directory.mkdir(parents=True, exist_ok=True)
    catalogue = parse_rows(_ROWS, make_aliases={}, source_sha256=_SHA256)
    (directory / "catalog.json").write_text(json.dumps(to_document(catalogue)), encoding="utf-8")
    (directory / "decisions.json").write_text(json.dumps(_decisions()), encoding="utf-8")
    return directory

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

import pytest

from icor.evidence.serialization import canonical_json_bytes, sha256_file


class ExampleStatus(StrEnum):
    READY = "ready"


@dataclass(frozen=True)
class ExampleRecord:
    recorded_on: date
    recorded_at: datetime
    amount: Decimal
    status: ExampleStatus
    labels: tuple[str, ...]
    location: Path


def test_canonical_json_is_stable_across_mapping_order() -> None:
    left = canonical_json_bytes({"b": 2, "a": 1})
    right = canonical_json_bytes({"a": 1, "b": 2})

    assert left == right == b'{"a":1,"b":2}\n'


def test_canonical_json_converts_contract_values_explicitly() -> None:
    value = ExampleRecord(
        recorded_on=date(2026, 8, 26),
        recorded_at=datetime(2026, 8, 26, 10, 0, tzinfo=UTC),
        amount=Decimal("12.50"),
        status=ExampleStatus.READY,
        labels=("source", "verified"),
        location=Path("releases/example.csv"),
    )

    assert canonical_json_bytes(value) == (
        b'{"amount":"12.50","labels":["source","verified"],'
        b'"location":"releases/example.csv","recorded_at":"2026-08-26T10:00:00+00:00",'
        b'"recorded_on":"2026-08-26","status":"ready"}\n'
    )


def test_canonical_json_rejects_floats() -> None:
    with pytest.raises(TypeError, match="float"):
        canonical_json_bytes({"value": 1.5})


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "release.csv"
    artifact.write_bytes(b"make,model,count\nA,B,1\n")

    assert sha256_file(artifact) == (
        "5ba45a928128f18ed081de659501374802968f3fc00d37ec9158bab5dd210777"
    )

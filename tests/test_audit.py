import json
from pathlib import Path

from icor.audit import audit_repository

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_audit_reports_duplicate_identity_and_null_units(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    write_json(
        data / "Top100_2025.txt",
        [
            {"model": "Example One", "generation": "I", "units_sold": 100},
            {"model": "Example One", "generation": "I", "units_sold": None},
        ],
    )
    (data / "icor_supported_models.txt").write_text(
        '{"example one": {2025: "G1"}}', encoding="utf-8"
    )

    report = audit_repository(tmp_path)

    assert report.file_count == 1
    assert report.record_count == 2
    assert report.catalog_model_count == 1
    assert report.catalog_match_count == 1
    assert [finding.code for finding in report.findings] == [
        "duplicate_identity",
        "incomplete_top100",
        "null_units",
        "unsupported_icor_catalog_format",
    ]


def test_audit_dictionary_is_deterministic_and_relative(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    write_json(
        data / "Top100_World_2025.txt",
        [{"model": "Example Two", "generation": "II", "units_sold": 20}],
    )
    (data / "icor_supported_models.txt").write_text(
        "example two\t2025", encoding="utf-8"
    )

    first = audit_repository(tmp_path).to_dict()
    second = audit_repository(tmp_path).to_dict()

    assert first == second
    assert str(tmp_path) not in json.dumps(first)


def test_repository_audit_has_expected_source_inventory() -> None:
    report = audit_repository(ROOT)

    assert report.file_count == 22
    assert report.record_count == 2_190
    assert sum(finding.code == "null_units" for finding in report.findings) == 50
    assert dict(report.forecast_constants)["script1.DECAY_RATE"] == 0.0556
    assert dict(report.forecast_constants)["script2.REPAIR_RATE"] == 0.021

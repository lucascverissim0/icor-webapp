from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EU_FILE = re.compile(r"^Top100_(\d{4})\.txt$")
WORLD_FILE = re.compile(r"^Top100_World_(\d{4})\.txt$")
CATALOG_KEY = re.compile(r'^\s*\{?\s*"([^"]+)"\s*:', re.MULTILINE)
FORECAST_CONSTANTS = {
    "DEADLINE_YEAR",
    "DECAY_RATE",
    "REPAIR_RATE",
    "REPL_RATE_MEAN",
    "SELECTED_YEAR",
    "YEARS_TO_PROJECT",
}


@dataclass(frozen=True, slots=True)
class AuditFinding:
    code: str
    path: str
    record_index: int | None
    field: str | None
    message: str


@dataclass(frozen=True, slots=True)
class AuditReport:
    file_count: int
    record_count: int
    catalog_model_count: int
    catalog_match_count: int
    runtime_python: str
    forecast_constants: tuple[tuple[str, int | float | str], ...]
    findings: tuple[AuditFinding, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "file_count": self.file_count,
            "record_count": self.record_count,
            "catalog_model_count": self.catalog_model_count,
            "catalog_match_count": self.catalog_match_count,
            "runtime_python": self.runtime_python,
            "forecast_constants": dict(self.forecast_constants),
            "findings": [asdict(finding) for finding in self.findings],
        }


def _normalize_identity(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _finding_sort_key(finding: AuditFinding) -> tuple[str, str, int, str, str]:
    return (
        finding.code,
        finding.path,
        -1 if finding.record_index is None else finding.record_index,
        finding.field or "",
        finding.message,
    )


def _source_constants(root: Path) -> tuple[tuple[str, int | float | str], ...]:
    values: list[tuple[str, int | float | str]] = []
    for script_name in ("script1", "script2"):
        path = root / "scripts" / f"{script_name}.py"
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in FORECAST_CONSTANTS:
                continue
            try:
                value: Any = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                value = ast.unparse(node.value)
            if isinstance(value, (int, float, str)):
                values.append((f"{script_name}.{target.id}", value))
    return tuple(sorted(values))


def _catalog_details(path: Path) -> tuple[set[str], bool]:
    if not path.is_file():
        return set(), False
    text = path.read_text(encoding="utf-8")
    names = {_normalize_identity(match) for match in CATALOG_KEY.findall(text)}
    try:
        parsed = json.loads(text)
        supported = isinstance(parsed, (dict, list))
    except json.JSONDecodeError:
        lines = [line for line in text.splitlines() if line.strip()]
        supported = bool(lines) and all("\t" in line for line in lines)
    return names, supported


def audit_repository(root: Path) -> AuditReport:
    resolved_root = root.resolve()
    data_dir = resolved_root / "data"
    findings: list[AuditFinding] = []
    record_count = 0
    market_models: set[str] = set()
    seen: dict[tuple[str, int, str, str], tuple[str, int]] = {}

    market_files: list[tuple[Path, str, int]] = []
    if data_dir.is_dir():
        for path in sorted(data_dir.glob("Top100*.txt"), key=lambda item: item.name):
            eu_match = EU_FILE.fullmatch(path.name)
            world_match = WORLD_FILE.fullmatch(path.name)
            if eu_match:
                market_files.append((path, "EU", int(eu_match.group(1))))
            elif world_match:
                market_files.append((path, "World", int(world_match.group(1))))

    for path, region, year in market_files:
        relative_path = _relative(path, resolved_root)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            findings.append(
                AuditFinding("invalid_json", relative_path, None, None, type(exc).__name__)
            )
            continue
        if not isinstance(document, list):
            findings.append(
                AuditFinding(
                    "non_list_document", relative_path, None, None, "Expected a JSON list"
                )
            )
            continue

        record_count += len(document)
        if len(document) != 100:
            findings.append(
                AuditFinding(
                    "incomplete_top100",
                    relative_path,
                    None,
                    None,
                    f"Expected 100 records, found {len(document)}",
                )
            )

        for index, record in enumerate(document):
            if not isinstance(record, dict):
                findings.append(
                    AuditFinding(
                        "non_object_record", relative_path, index, None, "Expected an object"
                    )
                )
                continue
            missing = [field for field in ("model", "generation") if field not in record]
            for field in missing:
                findings.append(
                    AuditFinding(
                        "missing_field", relative_path, index, field, f"Missing {field}"
                    )
                )
            if missing:
                continue

            model = _normalize_identity(record["model"])
            generation = _normalize_identity(record["generation"])
            market_models.add(model)
            key = (region, year, model, generation)
            if key in seen:
                findings.append(
                    AuditFinding(
                        "duplicate_identity",
                        relative_path,
                        index,
                        None,
                        f"Repeated identity first seen at record {seen[key][1]}",
                    )
                )
            else:
                seen[key] = (relative_path, index)

            units = record.get("units_sold", record.get("projected_units_2025"))
            if not isinstance(units, (int, float)) or isinstance(units, bool):
                findings.append(
                    AuditFinding(
                        "null_units", relative_path, index, "units_sold", "No numeric units"
                    )
                )

    catalog_path = data_dir / "icor_supported_models.txt"
    catalog_models, supported_catalog = _catalog_details(catalog_path)
    if catalog_path.is_file() and not supported_catalog:
        findings.append(
            AuditFinding(
                "unsupported_icor_catalog_format",
                _relative(catalog_path, resolved_root),
                None,
                None,
                "Catalog is neither valid JSON nor tab-delimited data",
            )
        )

    return AuditReport(
        file_count=len(market_files),
        record_count=record_count,
        catalog_model_count=len(catalog_models),
        catalog_match_count=len(catalog_models & market_models),
        runtime_python=f"{sys.version_info.major}.{sys.version_info.minor}",
        forecast_constants=_source_constants(resolved_root),
        findings=tuple(sorted(findings, key=_finding_sort_key)),
    )

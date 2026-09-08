"""Conservative read-only adapter for ICOR's legacy worked-model list."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from icor.evidence.normalization import normalize_vehicle_label

_ROW = re.compile(r'^\s*"([^"]+)"\s*:\s*(\{.*\})\s*,?\s*$')
_MAKES = tuple(
    sorted(
        (
            "land rover",
            "range rover",
            "mercedes",
            "mitsubishi",
            "citroen",
            "hyundai",
            "jaguar",
            "nissan",
            "renault",
            "skoda",
            "volvo",
            "honda",
            "mazda",
            "opel",
            "ford",
            "fiat",
            "audi",
            "bmw",
            "kia",
            "vw",
        ),
        key=len,
        reverse=True,
    )
)
_MAKE_ALIASES = {
    "vw": ("vw", "volkswagen", "volkswagen vw", "volkswagen. vw"),
}


@dataclass(frozen=True, slots=True)
class WorkedModelRecord:
    brand: str
    model: str
    model_year: int
    legacy_generation: str


@dataclass(frozen=True, slots=True)
class IcorWorkedModelCatalog:
    records: tuple[WorkedModelRecord, ...]

    @classmethod
    def empty(cls) -> IcorWorkedModelCatalog:
        return cls(())

    @classmethod
    def from_path(cls, path: Path) -> IcorWorkedModelCatalog:
        records: set[WorkedModelRecord] = set()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped in {"{", "}"}:
                continue
            match = _ROW.fullmatch(line)
            if match is None:
                raise ValueError(f"legacy ICOR worked-model line {line_number} is invalid")
            identity = _split_identity(match.group(1))
            if identity is None:
                continue
            try:
                generations = ast.literal_eval(match.group(2))
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"legacy ICOR worked-model line {line_number} is invalid"
                ) from error
            if not isinstance(generations, dict) or any(
                type(year) is not int or type(generation) is not str or not generation.strip()
                for year, generation in generations.items()
            ):
                raise ValueError(f"legacy ICOR worked-model line {line_number} is invalid")
            make, model = identity
            for alias in _MAKE_ALIASES.get(make, (make,)):
                for year, generation in generations.items():
                    records.add(WorkedModelRecord(alias, model, year, generation.strip()))
        return cls(
            tuple(
                sorted(records, key=lambda item: (item.brand, item.model, item.model_year))
            )
        )

    def matches(self, brand: str, model: str, model_year: int) -> bool:
        normalized_brand = normalize_vehicle_label(brand)
        normalized_model = normalize_vehicle_label(model)
        return any(
            record.brand == normalized_brand
            and record.model == normalized_model
            and record.model_year == model_year
            for record in self.records
        )


def _split_identity(value: str) -> tuple[str, str] | None:
    normalized = normalize_vehicle_label(value)
    if normalized is None or normalized.startswith("support "):
        return None
    for make in _MAKES:
        prefix = f"{make} "
        if normalized.startswith(prefix):
            model = normalized[len(prefix) :].strip()
            return (make, model) if model else None
    return None

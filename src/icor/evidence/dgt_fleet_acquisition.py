"""Deterministic acquisition of the DGT passenger-car stock by make, model and cohort."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import unicodedata
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.parse import urlsplit

from icor.evidence.sources.dgt import SCHEMA, classify_group

ARCHIVE_URL = "https://www.dgt.es/microdatos/Parque/2025/parque_consolidado_2025.zip"
_ALLOWED_HOST = "www.dgt.es"
# The 2025 edition (HTTP Last-Modified 2026-04-10). DGT publishes one file per
# year and has not replaced earlier years, so re-acquisition reproduces it.
ARCHIVE_BYTES = 1_645_095_393
ARCHIVE_SHA256 = "12959c31c3debae2135142372bd3fc4c88fceac998eac5e5701d218ca3ad1c65"
ARCHIVE_MEMBER = "parque_vehiculos_202512.txt"
#: The 45 variables of DGT's "Interfaz de salida, Fichero de Parque Anual".
SOURCE_HEADER = [
    "PROVINCIA",
    "MUNICIPIO",
    "FABRICANTE",
    "MARCA",
    "MODELO",
    "TIPO",
    "VARIANTE",
    "VERSION",
    "PROVINCIA_MATR",
    "FECHA_MATR",
    "FEC_PRIM_MATR",
    "CLASE_MATR",
    "PROCEDENCIA",
    "NUEVO_USADO",
    "TIPO_TITULAR",
    "NUM_TITULARES",
    "SUBTIPO_DGT",
    "TIPO_DGT",
    "CAT_EURO",
    "CLAS_CONSTRUCCION",
    "CLAS_UTILIZACION",
    "SERVICIO",
    "RENTING",
    "TARA",
    "PESO_MAX",
    "MOM",
    "MMTA",
    "CILINDRADA",
    "POTENCIA",
    "KW",
    "PROPULSION",
    "CATELECT",
    "CONSUMO",
    "AUTONOMIA",
    "ALIMENTACION",
    "TIPO_DISTINTIVO",
    "EMISIONES_EURO",
    "EMISIONES_CO2",
    "CARROCERIA",
    "DISTANCIA_EJES",
    "EJE_ANTERIOR",
    "EJE_POSTERIOR",
    "PLAZAS",
    "PLAZAS_MAX",
    "PLAZAS_PIE",
]
PASSENGER_CATEGORIES = frozenset({"M1", "M1G"})
# Cars homologated before EU categories carry none; DGT still types them as
# TURISMOS. With them the stock matches Eurostat's Spanish total (27.27M vs 27.28M).
UNCATEGORISED_PASSENGER_TYPE = "TURISMOS"
_DATE = re.compile(r"\d{2}/\d{2}/(\d{4})\Z")


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    path: Path
    artifact_bytes: int
    sha256: str
    source_rows: int
    passenger_cars: int
    group_count: int
    accepted_group_count: int
    rejected_group_count: int
    accepted_vehicle_count: int
    masked_vehicle_count: int


class _Response(Protocol):
    def __enter__(self) -> _Response: ...

    def __exit__(self, *args: object) -> object: ...

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...


def download_archive(
    destination: Path,
    *,
    opener: Callable[..., _Response] = urllib.request.urlopen,
    url: str = ARCHIVE_URL,
    expected_bytes: int = ARCHIVE_BYTES,
) -> Path:
    """Download the allowlisted archive atomically, enforcing its byte ceiling."""

    _require_allowlisted(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ICOR-evidence/1"})
        with opener(request, timeout=120) as response:
            _require_allowlisted(response.geturl())
            with NamedTemporaryFile("wb", delete=False, dir=destination.parent) as output:
                temporary = Path(output.name)
                copied = 0
                while chunk := response.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > expected_bytes:
                        raise ValueError("DGT archive exceeds the pinned byte limit")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def acquire_fleet(
    destination: Path,
    archive: Path,
    *,
    expected_bytes: int = ARCHIVE_BYTES,
    expected_sha256: str = ARCHIVE_SHA256,
    member: str = ARCHIVE_MEMBER,
) -> AcquisitionResult:
    """Aggregate the pinned archive into a sorted canonical CSV of counts only.

    Only make, model, first-registration year and a count leave the archive; no
    row-level field is written. Masked and otherwise unusable groups stay in the
    artifact so that the loader reports them against the manifest.
    """

    if archive.stat().st_size != expected_bytes or _sha256(archive) != expected_sha256:
        raise ValueError("DGT archive does not match the pinned release")
    counts: dict[tuple[str, str, str], int] = {}
    source_rows = passenger_cars = 0
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist() != [member]:
            raise ValueError("DGT archive members are unsupported")
        with bundle.open(member) as raw:
            stream = io.TextIOWrapper(raw, encoding="latin-1", newline="")
            header = stream.readline().rstrip("\r\n").split("|")
            if header != SOURCE_HEADER:
                raise ValueError("DGT archive schema is unsupported")
            position = {name: index for index, name in enumerate(header)}
            for line in stream:
                fields = line.rstrip("\r\n").split("|")
                if len(fields) != len(header):
                    raise ValueError("DGT archive row is malformed")
                source_rows += 1
                category = fields[position["CAT_EURO"]]
                if category not in PASSENGER_CATEGORIES and not (
                    category == "" and fields[position["TIPO_DGT"]] == UNCATEGORISED_PASSENGER_TYPE
                ):
                    continue
                passenger_cars += 1
                first = _DATE.match(fields[position["FEC_PRIM_MATR"]])
                key = (
                    _collapse(fields[position["MARCA"]]),
                    _collapse(fields[position["MODELO"]]),
                    first.group(1) if first else fields[position["FEC_PRIM_MATR"]].strip(),
                )
                counts[key] = counts.get(key, 0) + 1
    if not counts:
        raise ValueError("DGT archive holds no passenger cars")
    records = sorted(((*key, str(count)) for key, count in counts.items()), key=_sort_key)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    accepted = rejected = accepted_vehicles = masked = 0
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=destination.parent
        ) as output:
            temporary = Path(output.name)
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(SCHEMA)
            for record in records:
                writer.writerow(record)
                group = classify_group(dict(zip(SCHEMA, record, strict=True)))
                if group is None:
                    rejected += 1
                else:
                    accepted += 1
                    accepted_vehicles += group.vehicles
                if "¡" in record[0] or "¡" in record[1]:
                    masked += int(record[3])
            output.flush()
            os.fsync(output.fileno())
        artifact_bytes = temporary.stat().st_size
        sha256 = _sha256(temporary)
        os.replace(temporary, destination)
        temporary = None
        return AcquisitionResult(
            destination,
            artifact_bytes,
            sha256,
            source_rows,
            passenger_cars,
            len(records),
            accepted,
            rejected,
            accepted_vehicles,
            masked,
        )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _require_allowlisted(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != _ALLOWED_HOST:
        raise ValueError("DGT archive URL is outside its allowlisted origin")


def _collapse(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _sort_key(record: tuple[str, ...]) -> tuple[object, ...]:
    make, model, cohort, vehicles = record
    numeric = cohort.isascii() and cohort.isdigit()
    return (make, model, 0 if numeric else 1, int(cohort) if numeric else 0, cohort, vehicles)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

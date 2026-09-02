"""Deterministic model aggregate export from the official EEA cars viewer."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import unicodedata
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from urllib.parse import urlencode, urlsplit

from icor.evidence.normalization import normalize_vehicle_label
from icor.evidence.sources.eea import ANNUAL_AGGREGATE_SCHEMA

API_URL = "https://co2cars.apps.eea.europa.eu/tools/api"
YEAR = 2025
VERSION = "v31"
STATUS = "P"
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class _Response(Protocol):
    def __enter__(self) -> _Response: ...
    def __exit__(self, *args: object) -> object: ...
    def read(self) -> bytes: ...
    def geturl(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ProvisionalAcquisitionResult:
    path: Path
    artifact_bytes: int
    sha256: str
    group_count: int
    source_row_count: int
    accepted_row_count: int
    rejected_row_count: int
    registration_count: int


def acquire_2025_provisional(
    destination: Path,
    *,
    opener: Callable[..., _Response] = urllib.request.urlopen,
    page_size: int = 10_000,
) -> ProvisionalAcquisitionResult:
    if type(page_size) is not int or not 1 <= page_size <= 10_000:
        raise ValueError("EEA provisional page size must be between 1 and 10000")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    groups = source_rows = accepted = rejected = registrations = 0
    after: dict[str, object] | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=destination.parent
        ) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(
                output, ANNUAL_AGGREGATE_SCHEMA, delimiter=";", lineterminator="\n"
            )
            writer.writeheader()
            while True:
                buckets, next_after = _fetch_page(page_size, after, opener)
                if not buckets:
                    break
                for bucket in buckets:
                    row = _canonical_row(bucket)
                    writer.writerow(row)
                    groups += 1
                    count = int(row["SourceRows"])
                    source_rows += count
                    registrations += int(row["Registrations"])
                    if all(
                        normalize_vehicle_label(row[field]) is not None
                        for field in ("MS", "Mk", "Cn")
                    ):
                        accepted += count
                    else:
                        rejected += count
                if next_after is None:
                    break
                after = next_after
            output.flush()
            os.fsync(output.fileno())
        assert temporary is not None
        size = temporary.stat().st_size
        digest = _sha256(temporary)
        os.replace(temporary, destination)
        temporary = None
        return ProvisionalAcquisitionResult(
            destination, size, digest, groups, source_rows, accepted, rejected, registrations
        )
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _fetch_page(
    page_size: int,
    after: dict[str, object] | None,
    opener: Callable[..., _Response],
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    composite: dict[str, object] = {
        "size": page_size,
        "sources": [
            {field: {"terms": {"field": field, "missing_bucket": True}}}
            for field in ("MS", "Mk", "Cn")
        ],
    }
    if after is not None:
        composite["after"] = after
    query = {
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {"term": {"year": YEAR}},
                    {"term": {"scStatus": "Provisional"}},
                ]
            }
        },
        "aggs": {
            "groups": {
                "composite": composite,
                "aggs": {"registrations": {"sum": {"field": "r"}}},
            }
        },
    }
    url = f"{API_URL}?{urlencode({'source': json.dumps(query, separators=(',', ':'))})}"
    request = urllib.request.Request(url, headers={"User-Agent": "ICOR-evidence/1"})
    with opener(request, timeout=120) as response:
        actual = urlsplit(response.geturl())
        if actual.scheme != "https" or actual.netloc != "co2cars.apps.eea.europa.eu":
            raise ValueError("EEA provisional API redirected outside its allowlisted origin")
        payload_bytes = response.read()
    if len(payload_bytes) > _MAX_RESPONSE_BYTES:
        raise ValueError("EEA provisional API response exceeds the byte limit")
    payload = json.loads(payload_bytes)
    if not isinstance(payload, dict) or payload.get("timed_out") is True or "errors" in payload:
        raise ValueError("EEA provisional API returned an error response")
    groups = payload.get("aggregations", {}).get("groups", {})
    buckets = groups.get("buckets")
    after_key = groups.get("after_key")
    if not isinstance(buckets, list) or any(not isinstance(row, dict) for row in buckets):
        raise ValueError("EEA provisional API response schema is unsupported")
    if after_key is not None and not isinstance(after_key, dict):
        raise ValueError("EEA provisional API pagination is unsupported")
    return buckets, after_key


def _canonical_row(bucket: dict[str, object]) -> dict[str, object]:
    key = bucket.get("key")
    count = bucket.get("doc_count")
    registration = bucket.get("registrations")
    if not isinstance(key, dict) or type(count) is not int or not isinstance(registration, dict):
        raise ValueError("EEA provisional aggregate bucket is invalid")
    value = registration.get("value")
    if not isinstance(value, (int, float)) or value < 0 or int(value) != value or count <= 0:
        raise ValueError("EEA provisional aggregate counts are invalid")
    labels = {
        field: "" if key.get(field) is None else " ".join(
            unicodedata.normalize("NFC", str(key[field])).split()
        )
        for field in ("MS", "Mk", "Cn")
    }
    return {
        "Year": str(YEAR), "Status": STATUS, "Version_file": VERSION,
        **labels, "TAN": "", "T": "", "Va": "", "Ve": "", "Ft": "",
        "Registrations": str(int(value)), "SourceRows": str(count),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

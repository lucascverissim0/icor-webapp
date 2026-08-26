from __future__ import annotations

import csv
import json
import socket
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from icor.domain.evidence import (
    CanonicalVehicle,
    EvidenceConfidence,
    IdentityMapping,
    MappingStatus,
    Measure,
    Observation,
    PeriodPrecision,
    PublicationStatus,
)
from icor.evidence.serialization import canonical_json_bytes
from icor.infrastructure.release_store import StoredRelease
from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository
from scripts.build_evidence_snapshot import main

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources"
SAMPLE_ARTIFACT = FIXTURES / "sample-registration.csv"
SAMPLE_MANIFEST = FIXTURES / "sample-registration.manifest.json"
BUILD_AS_OF = "2026-08-26T12:00:00+00:00"
RELEASE_ID = "sample-registration-2024"


class SampleRegistrationLoader:
    """Contract-test adapter retained only in this integration composition root."""

    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        vehicle = CanonicalVehicle(
            vehicle_id="vehicle-example-motors-alpha-2024",
            make="Example Motors",
            model="Alpha",
            model_year=2024,
            market="EU",
        )
        repository.add_vehicle(vehicle)
        for release in releases:
            with release.artifact_path.open(encoding="utf-8", newline="") as artifact:
                rows = tuple(csv.DictReader(artifact))
            observations = tuple(
                self._observation(release, vehicle, row, position)
                for position, row in enumerate(rows, start=2)
            )
            repository.add_observations(observations)
            for observation in observations:
                repository.add_mapping(
                    IdentityMapping(
                        mapping_id=f"mapping-{observation.observation_id}",
                        observation_id=observation.observation_id,
                        canonical_vehicle_id=vehicle.vehicle_id,
                        status=MappingStatus.NORMALIZED_LABEL,
                        reason="Fictional labels match the contract-test registry.",
                        reviewed_at=datetime(2026, 8, 26, 11, 0, tzinfo=UTC),
                    )
                )

    @staticmethod
    def _observation(
        release: StoredRelease,
        vehicle: CanonicalVehicle,
        row: dict[str, str],
        position: int,
    ) -> Observation:
        country = row["reporting_country"]
        year = int(row["registration_year"])
        return Observation(
            observation_id=f"observation-{country.casefold()}-{year}",
            release_id=release.release_id,
            original_row_locator=f"row:{position}",
            geography=country,
            geography_version="eu-2024",
            period_start=date(year, 1, 1),
            period_end=date(year, 12, 31),
            period_precision=PeriodPrecision.YEAR,
            measure=Measure.NEW_REGISTRATIONS,
            value=Decimal(row["new_registrations"]),
            unit="vehicles",
            publication_status=PublicationStatus.FINAL,
            original_make=row["make"],
            original_model=row["model"],
            original_model_year=row["registration_year"],
            original_type=None,
            source_make_identifier=None,
            source_model_identifier=None,
            normalized_make=row["make"],
            normalized_model=row["model"],
            normalized_model_year=year,
            canonical_vehicle_id=vehicle.vehicle_id,
            mapping_status=MappingStatus.NORMALIZED_LABEL,
            transformation_notes=("Fictional labels normalized for contract testing.",),
            validation_flags=(),
            evidence_confidence=EvidenceConfidence(
                authority=10,
                publication_status=10,
                coverage=25,
                identity=15,
                independent_agreement=0,
                reasons=("Fictional single-source contract evidence.",),
            ),
        )


class RawMessageFailingLoader:
    def load(
        self,
        releases: tuple[StoredRelease, ...],
        repository: SQLiteEvidenceRepository,
    ) -> None:
        del releases, repository
        raise RuntimeError("Example Motors,Alpha,secret-row")


def _run_cli(
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    *,
    loaders: dict[str, object] | None = None,
) -> tuple[int, dict[str, Any], str]:
    code = main(arguments, loader_registry=loaders)
    captured = capsys.readouterr()
    assert captured.out
    payload = json.loads(captured.out)
    assert captured.out == canonical_json_bytes(payload).decode("utf-8")
    return code, payload, captured.err


def _root_arguments(root: Path) -> list[str]:
    return ["--root", str(root), "--allow-external-root"]


def _stage(
    capsys: pytest.CaptureFixture[str], root: Path
) -> tuple[int, dict[str, Any], str]:
    return _run_cli(
        capsys,
        [
            "stage-release",
            *_root_arguments(root),
            "--manifest",
            str(SAMPLE_MANIFEST),
            "--artifact",
            str(SAMPLE_ARTIFACT),
        ],
    )


def _build(
    capsys: pytest.CaptureFixture[str],
    root: Path,
    *,
    loaders: dict[str, object] | None,
) -> tuple[int, dict[str, Any], str]:
    return _run_cli(
        capsys,
        [
            "build",
            *_root_arguments(root),
            "--release",
            RELEASE_ID,
            "--build-as-of",
            BUILD_AS_OF,
            "--deterministic-seed",
            "17",
        ],
        loaders=loaders,
    )


def test_clean_room_build_is_reproducible_promotable_and_offline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_socket(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("clean-room CLI attempted to create a network socket")

    monkeypatch.setattr(socket, "socket", forbid_socket)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    fixture_bytes = {
        SAMPLE_ARTIFACT: SAMPLE_ARTIFACT.read_bytes(),
        SAMPLE_MANIFEST: SAMPLE_MANIFEST.read_bytes(),
    }
    outputs: list[dict[str, Any]] = []

    for root in (first_root, second_root):
        code, payload, error = _stage(capsys, root)
        assert code == 0
        assert error == ""
        assert payload == {
            "accepted_record_count": 2,
            "release_id": RELEASE_ID,
            "source_id": "sample-registration",
            "state": "staged",
        }
        outputs.append(payload)

    code, unavailable, error = _run_cli(
        capsys, ["status", *_root_arguments(first_root)]
    )
    assert code == 4
    assert unavailable == {"active_snapshot_id": None, "state": "unavailable"}
    assert error == "No active snapshot is available.\n"

    registry = {"sample_registration_csv": SampleRegistrationLoader()}
    first_code, first, first_error = _build(capsys, first_root, loaders=registry)
    second_code, second, second_error = _build(capsys, second_root, loaders=registry)
    assert first_code == second_code == 0
    assert first_error == second_error == ""
    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["database_sha256"] == second["database_sha256"]
    assert first["observation_count"] == second["observation_count"] == 2
    assert first["published_value_count"] == second["published_value_count"] == 0
    outputs.extend((first, second))

    code, promoted, error = _run_cli(
        capsys,
        [
            "promote",
            *_root_arguments(first_root),
            "--snapshot",
            first["snapshot_id"],
        ],
    )
    assert code == 0
    assert error == ""
    assert promoted["active_snapshot_id"] == first["snapshot_id"]
    outputs.append(promoted)

    code, status, error = _run_cli(capsys, ["status", *_root_arguments(first_root)])
    assert code == 0
    assert error == ""
    assert status["active_snapshot_id"] == first["snapshot_id"]
    outputs.append(status)

    code, verified, error = _run_cli(capsys, ["verify", *_root_arguments(first_root)])
    assert code == 0
    assert error == ""
    assert verified["snapshot_id"] == first["snapshot_id"]
    assert verified["state"] == "verified"
    outputs.append(verified)

    rendered = json.dumps(outputs, sort_keys=True)
    for raw_value in ("Example Motors", "Alpha", '"DE"', '"FR"'):
        assert raw_value not in rendered
    assert set(tmp_path.iterdir()) == {first_root, second_root}
    assert all(path.read_bytes() == content for path, content in fixture_bytes.items())


def test_build_rejects_unregistered_parser_with_typed_safe_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "evidence"
    assert _stage(capsys, root)[0] == 0

    code, payload, error = _build(capsys, root, loaders=None)

    assert code == 2
    assert payload == {
        "error": {"code": "unsupported_parser"},
        "state": "rejected",
    }
    assert error == "The requested release parser is not supported.\n"


def test_failed_snapshot_validation_exits_three(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "evidence"
    assert _stage(capsys, root)[0] == 0
    build_code, build, build_error = _build(
        capsys,
        root,
        loaders={"sample_registration_csv": SampleRegistrationLoader()},
    )
    assert build_code == 0
    assert build_error == ""
    database = root / "candidates" / build["snapshot_id"] / "evidence.sqlite3"
    database.write_bytes(database.read_bytes() + b"tampered-candidate")

    code, payload, error = _run_cli(
        capsys,
        [
            "promote",
            *_root_arguments(root),
            "--snapshot",
            build["snapshot_id"],
        ],
    )

    assert code == 3
    assert payload == {
        "error": {"code": "snapshot_validation_failed"},
        "state": "rejected",
    }
    assert error == "Snapshot validation failed.\n"


def test_external_root_requires_explicit_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "outside-workspace"

    code, payload, error = _run_cli(
        capsys,
        [
            "stage-release",
            "--root",
            str(root),
            "--manifest",
            str(SAMPLE_MANIFEST),
            "--artifact",
            str(SAMPLE_ARTIFACT),
        ],
    )

    assert code == 2
    assert payload == {"error": {"code": "invalid_root"}, "state": "rejected"}
    assert error == "Evidence root must be contained in the workspace.\n"
    assert not root.exists()


def test_root_and_subcommand_inputs_are_required_and_sanitized(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, payload, error = _run_cli(capsys, ["build", "--release", "secret-row"])

    assert code == 2
    assert payload == {"error": {"code": "invalid_input"}, "state": "rejected"}
    assert error == "Invalid command input.\n"
    assert "secret-row" not in error


def test_loader_failure_is_sanitized_without_raw_row_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "evidence"
    assert _stage(capsys, root)[0] == 0

    code, payload, error = _build(
        capsys,
        root,
        loaders={"sample_registration_csv": RawMessageFailingLoader()},
    )

    assert code == 3
    assert payload == {"error": {"code": "operation_failed"}, "state": "rejected"}
    assert error == "Evidence snapshot operation failed.\n"
    assert "Example Motors" not in error
    assert "secret-row" not in error

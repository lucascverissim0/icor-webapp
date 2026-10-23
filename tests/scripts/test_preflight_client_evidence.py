from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from icor.eurocodes.demo import write_demo_eurocodes
from scripts import build_client_eurocodes as builder
from scripts import preflight_client_evidence as preflight

FREE_TEXT = "free text"


@pytest.fixture
def client_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A client Eurocode map built from the demo one, whose snapshot checks pass."""

    builder.build(write_demo_eurocodes(tmp_path / "private"), tmp_path / "client")
    monkeypatch.setattr(
        "icor.evidence.vehicle_identity.identity_rules_sha256", lambda: "demo", raising=True
    )
    return tmp_path / "client"


def _problems(directory: Path) -> list[str]:
    snapshot = SimpleNamespace(scope="full", snapshot_id="demo")
    return preflight._eurocode_failures(directory, snapshot, Path("unused.sqlite3"))


def test_a_map_built_for_the_snapshot_passes(client_map: Path) -> None:
    assert _problems(client_map) == []


@pytest.mark.parametrize("level", ["document", "family", "identity"])
def test_a_key_the_loader_does_not_read_fails_the_preflight(client_map: Path, level: str) -> None:
    path = client_map / "decisions.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    target = {
        "document": document,
        "family": next(iter(document["families"].values())),
        "identity": next(iter(document["identities"].values())),
    }[level]
    target["note"] = FREE_TEXT
    path.write_text(json.dumps(document), encoding="utf-8")

    problems = _problems(client_map)

    assert len(problems) == 1
    assert "outside the allowlist" in problems[0]
    assert level in problems[0]
    assert FREE_TEXT not in problems[0] and "note" not in problems[0]


def test_a_reviewers_rationale_fails_the_preflight(client_map: Path) -> None:
    path = client_map / "decisions.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    next(iter(document["identities"].values()))["why"] = FREE_TEXT
    path.write_text(json.dumps(document), encoding="utf-8")

    assert any("outside the allowlist" in problem for problem in _problems(client_map))

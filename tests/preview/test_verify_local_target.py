"""`verify_client_release.py --local` proves the client map, not whichever one the shell names.

The proof is only worth filing if it ran against what the image carries. An
`ICOR_EUROCODE_DIR` left in the environment (the private map, say) must not be
what it loaded. The real artifacts are not opened: the runner preflight is
replaced by a probe that records the environment it was given.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from icor.api.app import ROOT
from scripts import verify_client_release as verify


class _Probe(Exception):
    pass


@pytest.fixture
def given(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The environment `_local_app` hands the container runner's preflight."""

    seen: dict[str, Any] = {}

    def validate_runner(environment: dict[str, str], **_: object) -> None:
        seen.update(environment)
        raise _Probe

    monkeypatch.setattr("icor.preview.runner.validate_runner", validate_runner)
    return seen


def test_an_ambient_eurocode_dir_is_ignored(
    given: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ICOR_EUROCODE_DIR", str(tmp_path / "private-map"))

    with pytest.raises(_Probe):
        verify._local_app("client-reviewer")

    assert given["ICOR_EUROCODE_DIR"] == str(ROOT / ".local" / "client-eurocode")


def test_an_explicit_eurocode_dir_wins_over_the_environment(
    given: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ICOR_EUROCODE_DIR", str(tmp_path / "private-map"))

    with pytest.raises(_Probe):
        verify._local_app("client-reviewer", tmp_path / "chosen")

    assert given["ICOR_EUROCODE_DIR"] == str(tmp_path / "chosen")


def test_the_command_line_chooses_the_map_only_for_a_local_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, Path | None]] = []

    def local_app(username: str, eurocode_dir: Path | None = None) -> None:
        calls.append((username, eurocode_dir))
        raise _Probe

    monkeypatch.setattr(verify, "_local_app", local_app)

    with pytest.raises(_Probe):
        verify.main(["--local", "--username", "u", "--eurocode-dir", str(tmp_path)])
    with pytest.raises(_Probe):
        verify.main(["--local", "--username", "u"])
    with pytest.raises(SystemExit):
        verify.main(["--url", "https://x.example", "--username", "u", "--eurocode-dir", "y"])

    assert calls == [("u", tmp_path), ("u", None)]
    assert "--eurocode-dir" in capsys.readouterr().err

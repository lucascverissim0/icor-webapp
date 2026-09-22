from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest


def test_browser_fixture_builds_a_reusable_sealed_candidate(tmp_path: Path) -> None:
    from icor.application.evidence_review import EvidenceReviewService
    from icor.application.registrations import RegistrationQuery, RegistrationService
    from scripts.e2e_fixture import prepare_e2e_fixture

    first = prepare_e2e_fixture(tmp_path)
    second = prepare_e2e_fixture(tmp_path)

    assert first == second
    evidence = EvidenceReviewService.from_candidate(first)
    registrations = RegistrationService.from_candidate(first)
    assert evidence.summary().observation_count == 3
    summary = registrations.summary()
    assert summary.model_count == 3
    assert summary.total_registrations == Decimal("600")
    tesla = registrations.ranking(RegistrationQuery(search="Tesla"))
    assert [(row.make, row.model, row.registrations) for row in tesla.items] == [
        ("TESLA", "MODEL Y", Decimal("200"))
    ]


def test_browser_runner_prepares_both_candidates_or_rejects_partial_configuration(
    tmp_path: Path,
) -> None:
    from scripts.run_e2e_dev import fixture_root_for, prepare_environment

    prepared = prepare_environment({}, fixture_root=tmp_path)

    evidence = Path(prepared["ICOR_E2E_EVIDENCE_CANDIDATE"])
    generation = Path(prepared["ICOR_E2E_GENERATION_CANDIDATE"])
    assert evidence == generation
    assert (evidence / "snapshot.json").is_file()
    for partial in (
        {"ICOR_E2E_EVIDENCE_CANDIDATE": str(evidence)},
        {"ICOR_E2E_GENERATION_CANDIDATE": str(generation)},
    ):
        with pytest.raises(ValueError, match="both be configured"):
            prepare_environment(partial, fixture_root=tmp_path)

    explicit = {
        "ICOR_E2E_EVIDENCE_CANDIDATE": "evidence-candidate",
        "ICOR_E2E_GENERATION_CANDIDATE": "generation-candidate",
    }
    assert prepare_environment(explicit, fixture_root=tmp_path) == explicit
    empty = prepare_environment(
        {
            "ICOR_E2E_EVIDENCE_CANDIDATE": "",
            "ICOR_E2E_GENERATION_CANDIDATE": "",
        },
        fixture_root=tmp_path,
    )
    assert Path(empty["ICOR_E2E_EVIDENCE_CANDIDATE"]) == evidence
    assert fixture_root_for(18001, 19001) != fixture_root_for(18002, 19002)


def test_browser_runner_never_shares_its_own_stdio_with_the_children(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The property that actually prevents the CI teardown hang.

    Playwright kills the webServer process group, which does not reach children
    that started their own session. If those children hold the write end of the
    pipe Playwright is reading, its `close` event never fires and teardown hangs
    forever with every test already green.
    """

    import subprocess

    from scripts import run_e2e_dev

    recorded: list[dict[str, object]] = []

    class _FakeProcess:
        pid = 4321

        def __init__(self) -> None:
            self.stdout = None

        def poll(self) -> int | None:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(command, **kwargs):  # type: ignore[no-untyped-def]
        recorded.append(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(run_e2e_dev.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(run_e2e_dev.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        run_e2e_dev, "prepare_environment", lambda environment, **_: dict(environment)
    )
    monkeypatch.setattr(run_e2e_dev, "_install_termination_handlers", lambda: None)
    monkeypatch.setattr(run_e2e_dev, "_pump", lambda process, label: None)

    run_e2e_dev.run(api_port=18999, web_port=19999)

    assert len(recorded) == 2
    for kwargs in recorded:
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.STDOUT


def test_browser_runner_escalates_when_a_child_ignores_the_first_signal() -> None:
    """A child that will not stop must not be left running after teardown."""

    import subprocess

    from scripts import run_e2e_dev

    class _StubbornProcess:
        pid = 777

        def __init__(self) -> None:
            self.killed = False
            self._waits = 0

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            self._waits += 1
            if self._waits == 1:
                raise subprocess.TimeoutExpired(cmd="child", timeout=timeout or 0)
            return 0

        def kill(self) -> None:
            self.killed = True

    process = _StubbornProcess()
    run_e2e_dev._stop(process)  # type: ignore[arg-type]

    assert process.killed or process._waits >= 2

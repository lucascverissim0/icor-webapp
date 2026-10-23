from __future__ import annotations

from pathlib import Path

import pytest


def test_a_relative_out_dir_is_reported_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The files were written and then the closing print raised ValueError."""

    from scripts import report_eurocode_data as report

    monkeypatch.chdir(report.ROOT)
    relative = Path(".local/private/eurocode/reports")

    assert report._written_label(relative / "data-coverage-x") == (
        ".local/private/eurocode/reports/data-coverage-x.{json,csv,html}"
    )

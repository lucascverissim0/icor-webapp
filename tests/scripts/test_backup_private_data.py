from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import backup_private_data as backup


@pytest.fixture
def folders(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "private"
    (source / "eurocode" / "review").mkdir(parents=True)
    (source / "eurocode" / "catalog.json").write_text('{"rows": []}\n', encoding="utf-8")
    (source / "eurocode" / "review" / "batch.bin").write_bytes(bytes(range(256)) * 10)
    (source / "backtests").mkdir()
    (source / "backtests" / "report.json").write_text("{}\n", encoding="utf-8")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    return source, str(remote)


def _private(_: str) -> bool:
    return True


def _commits(remote: str) -> int:
    completed = subprocess.run(
        ["git", f"--git-dir={remote}", "rev-list", "--count", "main"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(completed.stdout)


def test_a_backup_round_trips_every_file_and_leaves_no_git_in_the_folder(
    folders: tuple[Path, str],
) -> None:
    source, remote = folders

    result = backup.back_up(source, remote, is_private=_private)

    assert result["pushed"] is True
    assert result["files"] == 3
    assert not (source / ".git").exists()
    verified = backup.verify(source, remote)
    assert verified["matches"] is True
    assert verified["manifest_sha256"] == result["manifest_sha256"]


def test_an_unchanged_folder_adds_no_commit_and_a_change_adds_one_on_top(
    folders: tuple[Path, str],
) -> None:
    source, remote = folders
    backup.back_up(source, remote, is_private=_private)

    assert backup.back_up(source, remote, is_private=_private)["pushed"] is False
    assert _commits(remote) == 1

    (source / "backtests" / "report.json").write_text('{"v": 2}\n', encoding="utf-8")
    (source / "eurocode" / "review" / "batch.bin").unlink()
    assert backup.back_up(source, remote, is_private=_private)["pushed"] is True
    assert _commits(remote) == 2
    assert backup.verify(source, remote)["matches"] is True


def test_verify_reports_a_mismatch(folders: tuple[Path, str]) -> None:
    source, remote = folders
    backup.back_up(source, remote, is_private=_private)
    (source / "eurocode" / "catalog.json").write_text('{"rows": [1]}\n', encoding="utf-8")

    verified = backup.verify(source, remote)

    assert verified["matches"] is False
    assert verified["different"] == 1


def test_a_public_or_unconfirmed_repository_is_refused_before_anything_is_pushed(
    folders: tuple[Path, str],
) -> None:
    source, remote = folders

    with pytest.raises(backup.BackupError, match="not private"):
        backup.back_up(source, remote, is_private=lambda _: False)
    with pytest.raises(backup.BackupError, match="https://github.com/"):
        backup.github_repository_is_private(remote)
    assert (
        subprocess.run(
            ["git", f"--git-dir={remote}", "rev-parse", "--verify", "-q", "main"],
            capture_output=True,
            check=False,
        ).returncode
        != 0
    )


def test_files_github_would_reject_are_refused(
    folders: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, remote = folders
    monkeypatch.setattr(backup, "MAX_FILE_BYTES", 100)

    with pytest.raises(backup.BackupError, match="larger"):
        backup.back_up(source, remote, is_private=_private)

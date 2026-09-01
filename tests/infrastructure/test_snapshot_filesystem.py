from __future__ import annotations

import os
from pathlib import Path

import pytest

from icor.infrastructure.snapshot_filesystem import SnapshotFilesystem


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor-root contract")
def test_fsync_directory_reuses_the_pinned_root_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    filesystem = SnapshotFilesystem()

    with filesystem.pin_root(tmp_path, create=False) as pinned_root:
        original_open = os.open

        def reject_pinned_root_reopen(
            path: os.PathLike[str] | str,
            flags: int,
            mode: int = 0o777,
        ) -> int:
            if Path(path) == pinned_root:
                raise AssertionError("pinned descriptor root was reopened")
            return original_open(path, flags, mode)

        monkeypatch.setattr(os, "open", reject_pinned_root_reopen)

        filesystem.fsync_directory(pinned_root)


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor-root contract")
def test_nested_pin_restores_the_outer_root_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    filesystem = SnapshotFilesystem()
    inner_root = tmp_path / "inner"
    inner_root.mkdir()

    with filesystem.pin_root(tmp_path, create=False) as outer_pinned_root:
        with filesystem.pin_root(inner_root, create=False) as inner_pinned_root:
            filesystem.fsync_directory(inner_pinned_root)

        original_open = os.open

        def reject_outer_root_reopen(
            path: os.PathLike[str] | str,
            flags: int,
            mode: int = 0o777,
        ) -> int:
            if Path(path) == outer_pinned_root:
                raise AssertionError("outer pinned descriptor root was reopened")
            return original_open(path, flags, mode)

        monkeypatch.setattr(os, "open", reject_outer_root_reopen)

        filesystem.fsync_directory(outer_pinned_root)

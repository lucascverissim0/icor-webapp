#!/usr/bin/env python3
"""Back up the git-ignored ``.local/private`` folder to a private GitHub repository.

The folder holds ICOR's proprietary data, so nothing about it is printed except
counts and a manifest digest, and the push is refused unless GitHub reports the
target repository as private. The backup's history lives only on GitHub: each
run builds a throwaway git directory in a temporary folder, fetches the remote's
last commit, commits the folder's current state on top of it, pushes and deletes
the temporary folder, so no second copy of the data is kept on this machine.

    uv run python scripts/backup_private_data.py           # back up
    uv run python scripts/backup_private_data.py --verify  # compare with GitHub

Before pushing, the committed tree is compared with the folder's own file list:
a ``.gitignore`` or ``.gitattributes`` inside the folder can make git drop or
rewrite a file, and the throwaway index starts empty, so the push would delete it
from the backup. The counts reported are those of the committed tree.

``--verify`` clones the remote's last commit into a temporary folder, compares
every file's sha256 with the local folder and deletes the clone.

Credentials come from the repository's own non-interactive credential helper;
the visibility check uses the ``gh`` login in ``GH_CONFIG_DIR`` (default
``~/.config/gh-icor``). Restore with
``git clone https://github.com/lucascverissim0/icor-private-data .local/private``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".local" / "private"
REMOTE = "https://github.com/lucascverissim0/icor-private-data"
BRANCH = "main"
# GitHub rejects files over 100 MB; stop before a push half-succeeds.
MAX_FILE_BYTES = 95 * 1024 * 1024


class BackupError(RuntimeError):
    """A backup or verification that must not be reported as done."""


def manifest(folder: Path) -> dict[str, str]:
    """Every file under ``folder`` by relative POSIX path, with its sha256."""

    files: dict[str, str] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise BackupError("the folder contains a symbolic link")
        if not path.is_file():
            continue
        relative = path.relative_to(folder)
        if relative.parts[0] == ".git":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        files[relative.as_posix()] = digest.hexdigest()
    return files


def manifest_digest(files: dict[str, str]) -> str:
    lines = "".join(f"{path}\t{digest}\n" for path, digest in sorted(files.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def github_repository_is_private(remote: str) -> bool:
    """Ask GitHub, through the repository's gh login, whether ``remote`` is private."""

    prefix = "https://github.com/"
    if not remote.startswith(prefix):
        raise BackupError("only a https://github.com/ remote can be checked for privacy")
    name = remote.removeprefix(prefix).removesuffix(".git")
    environment = dict(os.environ)
    environment.setdefault("GH_CONFIG_DIR", str(Path.home() / ".config" / "gh-icor"))
    completed = subprocess.run(
        ["gh", "repo", "view", name, "--json", "isPrivate"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise BackupError("GitHub did not confirm the backup repository's visibility")
    return json.loads(completed.stdout).get("isPrivate") is True


def _credential_options() -> list[str]:
    """The repository's own GitHub credential helpers, as ``-c`` options."""

    completed = subprocess.run(
        ["git", "-C", str(ROOT), "config", "--get-all", "credential.https://github.com.helper"],
        capture_output=True,
        text=True,
        check=False,
    )
    helpers = completed.stdout.splitlines() if completed.returncode == 0 else []
    options: list[str] = []
    for helper in helpers:
        options += ["-c", f"credential.https://github.com.helper={helper}"]
    return options


def _identity_options() -> list[str]:
    options: list[str] = []
    for key in ("user.name", "user.email"):
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "config", "--get", key],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            options += ["-c", f"{key}={completed.stdout.strip()}"]
    return options


def _git(
    arguments: Sequence[str],
    *,
    git_dir: Path,
    work_tree: Path | None = None,
    options: Sequence[str] = (),
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    # A global excludes or attributes file must not silently drop or rewrite a file.
    command = [
        "git",
        "-c",
        "core.excludesFile=",
        "-c",
        "core.attributesFile=",
        "-c",
        "core.autocrlf=false",
        *options,
        f"--git-dir={git_dir}",
    ]
    if work_tree is not None:
        command.append(f"--work-tree={work_tree}")
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GCM_INTERACTIVE"] = "never"
    completed = subprocess.run(
        [*command, *arguments],
        capture_output=True,
        text=True,
        env=environment,
        cwd=work_tree,
        check=False,
    )
    if check and completed.returncode != 0:
        # Git's own message can name a path from the folder; keep it off stdout.
        raise BackupError(f"git {arguments[0]} failed (exit {completed.returncode})")
    return completed


def _committed_tree(git_dir: Path) -> dict[str, int]:
    """Every file in the last commit by path, with the size git stored for it."""

    listing = _git(["ls-tree", "-r", "-l", "-z", "HEAD"], git_dir=git_dir).stdout
    tree: dict[str, int] = {}
    for entry in filter(None, listing.split("\0")):
        meta, path = entry.split("\t", 1)
        _, kind, _, size = meta.split()
        # A submodule entry has no size and is never a file of the folder.
        tree[path] = int(size) if kind == "blob" else -1
    return tree


def _check_committed_tree(source: Path, files: dict[str, str], tree: dict[str, int]) -> None:
    """Refuse a commit that does not hold exactly the folder's files, at their sizes.

    Counts only: a path or its name is the folder's own data.
    """

    missing = set(files) - set(tree)
    extra = set(tree) - set(files)
    altered = {
        path for path in set(files) & set(tree) if tree[path] != (source / path).stat().st_size
    }
    if missing or extra or altered:
        raise BackupError(
            "refusing to push: the committed tree is not the folder "
            f"({len(missing)} missing, {len(extra)} extra, {len(altered)} altered); "
            "a .gitignore or .gitattributes inside the folder may be hiding or rewriting files"
        )


def _remote_has_branch(remote: str, git_dir: Path, options: Sequence[str]) -> bool:
    completed = _git(["ls-remote", "--heads", remote, BRANCH], git_dir=git_dir, options=options)
    return bool(completed.stdout.strip())


def back_up(
    source: Path,
    remote: str,
    *,
    is_private: Callable[[str], bool] = github_repository_is_private,
    message: str = "Back up private working data",
) -> dict[str, object]:
    if not source.is_dir():
        raise BackupError("the folder to back up does not exist")
    if not is_private(remote):
        raise BackupError("refusing to push: the backup repository is not private")
    files = manifest(source)
    if not files:
        raise BackupError("refusing to back up an empty folder")
    largest = max((source / path).stat().st_size for path in files)
    if largest > MAX_FILE_BYTES:
        raise BackupError("a file is larger than GitHub accepts")
    options = [*_credential_options(), *_identity_options()]
    with tempfile.TemporaryDirectory(prefix="icor-private-backup-") as scratch:
        git_dir = Path(scratch) / "backup.git"
        _git(["init", "-q", "--bare", str(git_dir)], git_dir=git_dir)
        _git(["config", "core.bare", "false"], git_dir=git_dir)
        _git(["symbolic-ref", "HEAD", f"refs/heads/{BRANCH}"], git_dir=git_dir)
        if _remote_has_branch(remote, git_dir, options):
            _git(["fetch", "-q", "--depth", "1", remote, BRANCH], git_dir=git_dir, options=options)
            _git(["update-ref", f"refs/heads/{BRANCH}", "FETCH_HEAD"], git_dir=git_dir)
        _git(["add", "-A"], git_dir=git_dir, work_tree=source)
        has_parent = (
            _git(["rev-parse", "--verify", "-q", "HEAD"], git_dir=git_dir, check=False).returncode
            == 0
        )
        unchanged = (
            has_parent
            and _git(
                ["diff", "--cached", "--quiet", "HEAD"],
                git_dir=git_dir,
                work_tree=source,
                check=False,
            ).returncode
            == 0
        )
        if not unchanged:
            _git(
                ["commit", "-q", "--no-verify", "-m", message],
                git_dir=git_dir,
                work_tree=source,
                options=options,
            )
        tree = _committed_tree(git_dir)
        _check_committed_tree(source, files, tree)
        if not unchanged:
            _git(
                ["push", "-q", remote, f"HEAD:refs/heads/{BRANCH}"],
                git_dir=git_dir,
                options=options,
            )
        commit = _git(["rev-parse", "--short", "HEAD"], git_dir=git_dir).stdout.strip()
    return {
        "files": len(tree),
        "bytes": sum(tree.values()),
        "manifest_sha256": manifest_digest(files),
        "commit": commit,
        "pushed": not unchanged,
    }


def verify(source: Path, remote: str) -> dict[str, object]:
    """Clone the remote's last commit into a temporary folder and compare every file."""

    local = manifest(source)
    options = _credential_options()
    with tempfile.TemporaryDirectory(prefix="icor-private-verify-") as scratch:
        clone = Path(scratch) / "clone"
        environment = dict(os.environ)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        completed = subprocess.run(
            [
                "git",
                *options,
                "clone",
                "-q",
                "--depth",
                "1",
                "--branch",
                BRANCH,
                remote,
                str(clone),
            ],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            raise BackupError(f"git clone failed (exit {completed.returncode})")
        remote_files = manifest(clone)
        shutil.rmtree(clone)
    missing = sorted(set(local) - set(remote_files))
    extra = sorted(set(remote_files) - set(local))
    different = sorted(
        path for path in set(local) & set(remote_files) if local[path] != remote_files[path]
    )
    return {
        "files": len(local),
        "remote_files": len(remote_files),
        "missing_on_remote": len(missing),
        "only_on_remote": len(extra),
        "different": len(different),
        "manifest_sha256": manifest_digest(local),
        "remote_manifest_sha256": manifest_digest(remote_files),
        "matches": not (missing or extra or different),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        result = (
            verify(arguments.source, arguments.remote)
            if arguments.verify
            else back_up(arguments.source, arguments.remote)
        )
    except BackupError as error:
        print(f"backup refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1, sort_keys=True))
    if arguments.verify and not result["matches"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

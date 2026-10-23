"""MAC-COOL (Lucas 2026-10-23): keep the shared Mac cool. PreToolUse hook on Bash.

Modelled on Friday's ~/PEBLE/tools/hooks/mac_load_guard.py (MAC-COOL-1). It refuses with
exit 2 and puts the reason on stderr, where the agent sees it, in two cases:
  1. A HEAVY command that is not wrapped in tools/heavy.sh. heavy.sh queues the job behind
     the ONE Mac-wide lock that Friday shares, waits until the Mac is cool, and runs it under
     nice. The heavy commands are:
     - the full pytest suite and the full Vitest suite;
     - the web build and any Playwright run;
     - ICOR's long data jobs (snapshot builds, backtests, benchmarks, calibrations).
     Targeted tests stay free.
  2. `git worktree add` into a folder that Spotlight indexes. Spotlight skips dot folders,
     `*.noindex` folders and temp folders.
The hook must run on the system /usr/bin/python3 (3.9) as well as on 3.12. It fails open
on any error, so a defect in it can never block a session.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys

HEAVY_HINT = (
    "MAC-COOL: this is a heavy job. Run it as `tools/heavy.sh <command>` in the background. "
    "That runs one heavy job at a time on the whole Mac, shared with Friday, waits until the "
    "Mac is cool, and uses nice. Or target the test files you changed."
)
WORKTREE_HINT = (
    "MAC-COOL: Spotlight would index this worktree and its node_modules. Put it under a dot "
    "folder (e.g. .claude/worktrees/<name>), a *.noindex folder or /tmp, not {path}."
)

# Redirections and heredocs go before tokenising, so `2>&1` or `> out.log` never look like a
# test target. The redirection must start a word.
_REDIRECT = re.compile(
    r"""(?:(?<=[\s;|(&])|^)(?:\d+|&)?(?:>>|>\||>&|<&|<<<|<<-|<<|>|<)\s*"""
    r"""(?:"[^"]*"|'[^']*'|[^\s;&|()<>]+)?"""
)
_SEPARATOR = re.compile(r"^[();<>|&]+$")

_WRAPPERS = {"nice", "time", "env", "timeout", "caffeinate", "command", "exec", "nohup"}
_WRAPPER_VALUE_FLAGS = {"-n", "-s", "-k", "-u", "-t", "-w", "--signal", "--kill-after", "--unset"}
_SHELLS = {"bash", "sh", "zsh", "dash"}
_PYTHON = re.compile(r"^(python|python3|python3\.\d+|py)$")
_PYTHON_VALUE_FLAGS = {"-X", "-W", "-Q"}
_UV_RUN_VALUE_FLAGS = {
    "--project",
    "--directory",
    "--group",
    "--only-group",
    "--no-group",
    "--extra",
    "--no-extra",
    "--with",
    "--with-editable",
    "--with-requirements",
    "--python",
    "-p",
    "--package",
    "--env-file",
    "--index",
    "--default-index",
    "--index-url",
    "--extra-index-url",
    "--config-file",
    "--cache-dir",
    "--color",
}
_NPM_VALUE_FLAGS = {"--prefix", "-w", "--workspace", "--loglevel", "--userconfig", "--cache"}
_NPX_VALUE_FLAGS = {"-p", "--package", "-c", "--call"}
_PYTEST_VALUE_FLAGS = {
    "-k",
    "-m",
    "-p",
    "-c",
    "-o",
    "-W",
    "--rootdir",
    "--basetemp",
    "--deselect",
    "--ignore",
    "--ignore-glob",
    "--junitxml",
    "--tb",
    "--maxfail",
    "--durations",
    "--override-ini",
    "--confcutdir",
    "--log-level",
    "--cov",
    "--import-mode",
    "--capture",
    "-r",
}
_PYTEST_TARGETED_FLAGS = {"-k", "--lf", "--last-failed", "--co", "--collect-only"}
_PYTEST_WHOLE_TREE = {"tests", "."}
_VITEST_VALUE_FLAGS = {
    "--reporter",
    "--config",
    "-c",
    "--project",
    "--dir",
    "--root",
    "-r",
    "-t",
    "--testNamePattern",
    "--maxWorkers",
    "--pool",
    "--environment",
    "--outputFile",
}
_HEAVY_SCRIPT = re.compile(
    r"^(build_|rebuild_|benchmark_|backtest_|calibrate_|measure_)\w*\.py$"
    r"|^(verify_client_release|compare_snapshot_cohorts|report_snapshot_completeness)\.py$"
)


def _segments(command):
    """Simple commands as token lists, split on ; & && || | ( ) and newlines."""
    text = _REDIRECT.sub(" ", command.replace("\\\n", " ").replace("\n", " ; "))
    try:
        lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = re.sub(r"([();|&])", r" \1 ", text).split()
    segment = []
    for token in tokens:
        if _SEPARATOR.match(token):
            if segment:
                yield segment
            segment = []
        else:
            segment.append(token)
    if segment:
        yield segment


def _unwrap(tokens):
    """Drop leading NAME=value assignments and wrappers (nice -n 10, timeout 600, ...)."""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            i += 1
        elif os.path.basename(token) in _WRAPPERS:
            i += 1
            while i < len(tokens) and (
                tokens[i].startswith("-")
                or re.match(r"^\d+(\.\d+)?[smhd]?$", tokens[i])
                or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[i])
            ):
                i += 2 if tokens[i] in _WRAPPER_VALUE_FLAGS else 1
        else:
            break
    return tokens[i:]


def _positionals(args, value_flags):
    """Arguments that are not flags or flag values; a bare `--` is skipped."""
    found, skip = [], False
    for arg in args:
        if skip:
            skip = False
        elif arg == "--":
            continue
        elif arg.startswith("-"):
            skip = arg in value_flags
        else:
            found.append(arg)
    return found


def _pytest_heavy(args):
    if any(arg.split("=")[0] in _PYTEST_TARGETED_FLAGS for arg in args):
        return False
    targets = [os.path.normpath(p) for p in _positionals(args, _PYTEST_VALUE_FLAGS)]
    return all(t in _PYTEST_WHOLE_TREE for t in targets)


def _vitest_heavy(args):
    if args[:1] == ["related"]:
        return False
    if args[:1] in (["run"], ["watch"], ["dev"]):
        args = args[1:]
    return not _positionals(args, _VITEST_VALUE_FLAGS)


def _script_heavy(path, args):
    return bool(_HEAVY_SCRIPT.match(os.path.basename(path))) and not (
        "--help" in args or "-h" in args
    )


def _python_heavy(args):
    i = 0
    while i < len(args) and args[i].startswith("-") and args[i] not in ("-m", "-c"):
        i += 2 if args[i] in _PYTHON_VALUE_FLAGS else 1
    rest = args[i:]
    if rest[:1] == ["-c"]:
        return False
    if rest[:1] == ["-m"]:
        return len(rest) > 1 and rest[1] == "pytest" and _pytest_heavy(rest[2:])
    return bool(rest) and rest[0].endswith(".py") and _script_heavy(rest[0], rest[1:])


def _uv_run(args):
    """The command after `uv run [options]`, as tokens; None when it is not `uv run`."""
    if args[:1] != ["run"]:
        return None
    i = 1
    while i < len(args) and args[i].startswith("-"):
        if args[i] == "--":
            i += 1
            break
        if args[i] == "-m":
            return ["python", *args[i:]]
        i += 2 if args[i] in _UV_RUN_VALUE_FLAGS else 1
    return args[i:]


def _npm_heavy(args):
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in _NPM_VALUE_FLAGS else 1
    rest = args[i:]
    if not rest:
        return False
    if rest[0] in ("test", "t", "tst"):
        script, extra = "test", rest[1:]
    elif rest[0] in ("run", "run-script", "rum", "urn"):
        named = _positionals(rest[1:], _NPM_VALUE_FLAGS)
        if not named:
            return False
        script = named[0]
        extra = rest[rest.index(script, 1) + 1 :]
    elif rest[0] in ("exec", "x"):
        return _npx_heavy(rest[1:])
    else:
        return False
    if script == "test":
        return not _positionals(extra, _VITEST_VALUE_FLAGS)
    return script in ("build", "e2e")


def _npx_heavy(args):
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in _NPX_VALUE_FLAGS else 1
    rest = args[i:]
    return bool(rest) and _tool_heavy(os.path.basename(rest[0]), rest[1:])


def _tool_heavy(tool, args):
    if tool == "vitest":
        return _vitest_heavy(args)
    if tool == "playwright":
        return args[:1] == ["test"]
    if tool == "vite":
        return args[:1] == ["build"]
    return False


def _is_heavy(tokens):
    tokens = _unwrap(tokens)
    if not tokens:
        return False
    cmd, args = os.path.basename(tokens[0]), tokens[1:]
    if cmd == "uv":
        inner = _uv_run(args)
        return bool(inner) and _is_heavy(inner)
    if _PYTHON.match(cmd):
        return _python_heavy(args)
    if cmd == "pytest":
        return _pytest_heavy(args)
    if cmd in ("npm", "pnpm", "yarn"):
        return _npm_heavy(args)
    if cmd == "npx":
        return _npx_heavy(args)
    if cmd.endswith(".py"):
        return _script_heavy(cmd, args)
    return _tool_heavy(cmd, args)


def _shell_script(tokens):
    """The script of `bash|sh|zsh [-flags]c '<script>'`, or None."""
    tokens = _unwrap(tokens)
    if not tokens or os.path.basename(tokens[0]) not in _SHELLS:
        return None
    for i, arg in enumerate(tokens[1:-1], start=1):
        if re.match(r"^-[a-zA-Z]*c[a-zA-Z]*$", arg):
            return tokens[i + 1]
        if not arg.startswith("-"):
            return None
    return None


def _hidden_from_spotlight(path):
    parts = path.split(os.sep)
    temp_roots = ("/tmp/", "/private/tmp/", "/var/folders/", "/private/var/folders/")
    return (
        any(p.startswith(".") and p not in (".", "..") for p in parts)
        or any(p.endswith(".noindex") for p in parts)
        or any((path + os.sep).startswith(root) for root in temp_roots)
    )


def _indexed_worktree(tokens, cwd):
    """The path a `git worktree add` would create, when Spotlight indexes it."""
    if not tokens or os.path.basename(tokens[0]) != "git":
        return None
    args, base = tokens[1:], cwd
    while len(args) > 1 and args[0] == "-C":
        base = os.path.join(base, os.path.expanduser(args[1]))
        args = args[2:]
    if args[:2] != ["worktree", "add"]:
        return None
    targets = _positionals(args[2:], {"-b", "-B", "--reason"})
    if not targets:
        return None
    path = os.path.normpath(os.path.join(base, os.path.expanduser(targets[0])))
    return None if _hidden_from_spotlight(path) else path


def _through_heavy_sh(tokens):
    """`tools/heavy.sh <cmd>` or `bash tools/heavy.sh <cmd>`: the job is already queued."""
    tokens = _unwrap(tokens)
    if tokens and os.path.basename(tokens[0]) in _SHELLS:
        tokens = tokens[1:]
    return bool(tokens) and os.path.basename(tokens[0]) == "heavy.sh"


def check(command, cwd):
    """A refusal message, or None to allow."""
    for tokens in _segments(command):
        if _through_heavy_sh(tokens):
            continue
        script = _shell_script(tokens)
        if script is not None:
            refused = check(script, cwd)
            if refused:
                return refused
            continue
        if _is_heavy(tokens):
            return HEAVY_HINT + "\n  refused: " + " ".join(tokens)[:160]
        path = _indexed_worktree(_unwrap(tokens), cwd)
        if path:
            return WORKTREE_HINT.format(path=path)
    return None


def main():
    try:
        payload = json.load(sys.stdin)
        if payload.get("tool_name") not in (None, "Bash"):
            return 0
        command = (payload.get("tool_input") or {}).get("command") or ""
        message = check(command, payload.get("cwd") or os.getcwd())
        if message:
            print(message, file=sys.stderr)
            return 2
        return 0
    except Exception:
        return 0  # fail open: never block on our own defects


if __name__ == "__main__":
    sys.exit(main())

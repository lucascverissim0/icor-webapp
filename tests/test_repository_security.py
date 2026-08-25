from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CREDENTIAL_SHAPE = re.compile(rb"sk-[A-Za-z0-9_-]{20,}")
BINARY_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pyc",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}


def test_tracked_text_files_contain_no_openai_credential_shape() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    violations: list[str] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative_path = raw_path.decode("utf-8")
        path = ROOT / relative_path
        if path.suffix.casefold() in BINARY_EXTENSIONS:
            continue
        if CREDENTIAL_SHAPE.search(path.read_bytes()):
            violations.append(relative_path)

    assert not violations, "Credential-like token in tracked file(s): " + ", ".join(violations)

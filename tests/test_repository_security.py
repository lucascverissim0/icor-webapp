import re
import subprocess
from pathlib import Path

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


def test_local_secret_template_is_safe_and_real_secrets_remain_ignored() -> None:
    example = (ROOT / ".streamlit" / "secrets.example.toml").read_text(encoding="utf-8")
    assert "sk-" not in example

    result = subprocess.run(
        ["git", "check-ignore", ".streamlit/secrets.toml"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_development_guide_documents_reproducible_local_commands() -> None:
    guide = (ROOT / "docs" / "DEVELOPMENT.md").read_text(encoding="utf-8")
    for required_text in (
        "icor-webapp-development",
        "uv sync --locked --all-groups",
        "uv run pytest",
        "uv run python scripts/audit_baseline.py",
        "uv run streamlit run ui/app.py",
    ):
        assert required_text in guide

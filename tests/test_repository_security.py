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


def credential_shape_violations(
    repository_root: Path, relative_paths: tuple[str, ...] | None = None
) -> list[str]:
    """Scan repository text files with the same credential boundary used by this suite."""
    if relative_paths is None:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repository_root,
            check=True,
            capture_output=True,
        )
        relative_paths = tuple(
            raw_path.decode("utf-8") for raw_path in result.stdout.split(b"\0") if raw_path
        )

    violations: list[str] = []
    for relative_path in relative_paths:
        path = repository_root / relative_path
        if path.suffix.casefold() in BINARY_EXTENSIONS:
            continue
        if CREDENTIAL_SHAPE.search(path.read_bytes()):
            violations.append(relative_path)
    return violations


def git_check_ignore(repository_root: Path, relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--", relative_path],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def test_tracked_text_files_contain_no_openai_credential_shape() -> None:
    violations = credential_shape_violations(ROOT)

    assert not violations, "Credential-like token in tracked file(s): " + ", ".join(violations)


def test_local_secret_template_is_safe_and_real_secrets_remain_ignored() -> None:
    example = (ROOT / ".streamlit" / "secrets.example.toml").read_text(encoding="utf-8")
    assert "sk-" not in example

    assert git_check_ignore(ROOT, ".streamlit/secrets.toml")


def test_runtime_evidence_and_candidate_snapshots_are_ignored() -> None:
    assert git_check_ignore(
        ROOT, ".local/evidence/candidates/example/evidence.sqlite3"
    )


def test_fictional_source_fixture_is_clear_to_repository_credential_scanner() -> None:
    findings = credential_shape_violations(
        ROOT, ("tests/fixtures/sources/sample-registration.csv",)
    )

    assert findings == []


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

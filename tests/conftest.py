import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ANNOTATION_UNSAFE = re.compile(r"[\r\n]+")


@pytest.fixture(autouse=True)
def isolate_integration_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep every test offline and independent from machine credentials."""
    for name in (
        "OPENAI_API_KEY",
        "SERPAPI_KEY",
        "POSTHOG_API_KEY",
        "ICOR_EXTERNAL_NETWORK",
        "ICOR_PREVIEW_USERS",
        "ICOR_PREVIEW_SESSION_SECRET",
        "ICOR_PREVIEW_SESSION_TTL_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):  # noqa: ANN201
    """Emit a GitHub annotation for each failure, so a CI failure names itself.

    The windows-latest job has failed for weeks without anyone being able to say
    which test failed: the workflow log endpoint needs a token this machine does
    not have. Check-run annotations are readable without one, so a failure that
    carries its own name and assertion text is diagnosable from here.
    """

    del item, call
    outcome = yield
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    report = outcome.get_result()
    if report.when not in {"call", "teardown"} or not report.failed:
        return

    location, line, _ = report.location
    candidate = Path(location)
    try:
        resolved = (
            candidate if candidate.is_absolute() else _REPOSITORY_ROOT / candidate
        ).resolve()
        path = resolved.relative_to(_REPOSITORY_ROOT)
    except (ValueError, OSError):
        path = candidate

    message = _ANNOTATION_UNSAFE.sub(" ", str(report.longrepr))[:900]
    print(
        f"::error file={path.as_posix()},line={(line or 0) + 1},"
        f"title={report.nodeid}::{message}",
        flush=True,
    )

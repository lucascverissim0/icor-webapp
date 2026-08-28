from collections.abc import Iterator

import pytest


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

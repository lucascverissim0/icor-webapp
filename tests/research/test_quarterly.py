import json
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from icor.research.quarterly import (
    BATCHES,
    BUDGET_USD,
    MAX_OUTPUT_TOKENS_PER_BATCH,
    MAX_TOOL_CALLS_PER_BATCH,
    TARGETS,
    ResearchError,
    conservative_run_ceiling_usd,
    run_quarterly_research,
)


def _candidate(**changes):
    value = {
        "target_key": "uk-dft-vehicle-model",
        "publisher": "UK Department for Transport / DVLA",
        "source_url": "https://www.gov.uk/government/statistics/vehicle-licensing-statistics",
        "release_title": "Vehicle licensing statistics",
        "period_start": "2026-Q1",
        "period_end": "2026-Q2",
        "measure": "first registrations",
        "geography": "Great Britain",
        "model_detail": True,
        "publication_status": "provisional",
        "evidence_note": "Candidate release; verify table definitions before ingestion.",
    }
    value.update(changes)
    return value


class _FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payload if "uk-dft-vehicle-model" in kwargs["input"] else {"candidates": []}
        return SimpleNamespace(
            id=f"resp_test_{len(self.calls)}",
            output_text=json.dumps(payload),
            output=[SimpleNamespace(type="web_search_call")],
            usage=SimpleNamespace(input_tokens=2_000, output_tokens=500),
        )


def test_request_is_bounded_and_report_requires_human_review(tmp_path):
    responses = _FakeResponses({"candidates": [_candidate()]})
    destination = run_quarterly_research(
        SimpleNamespace(responses=responses),
        tmp_path,
        now=datetime(2026, 10, 1, tzinfo=UTC),
    )

    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["review_status"] == "pending-human-review"
    assert report["automatic_promotion"] is False
    assert report["budget"]["estimated_metered_cost_usd"] == "0.0576"
    assert len(responses.calls) == len(BATCHES)
    assert len(report["response_ids"]) == len(BATCHES)
    assert len(report["targets"]) == 33
    assert {target.batch for target in TARGETS} == set(BATCHES)
    for request in responses.calls:
        assert request["model"] == "o4-mini"
        assert request["max_tool_calls"] == MAX_TOOL_CALLS_PER_BATCH
        assert request["max_output_tokens"] == MAX_OUTPUT_TOKENS_PER_BATCH
        assert request["store"] is False
        assert request["tools"] == [{"type": "web_search"}]


@pytest.mark.parametrize(
    "url",
    [
        "http://www.gov.uk/not-https",
        "https://gov.uk.evil.example/release",
        "https://user:password@gov.uk/release",
        "https://example.com/release",
    ],
)
def test_non_official_candidate_urls_fail_closed(tmp_path, url):
    responses = _FakeResponses({"candidates": [_candidate(source_url=url)]})
    with pytest.raises(ResearchError, match="official allowlist"):
        run_quarterly_research(
            SimpleNamespace(responses=responses),
            tmp_path,
            now=datetime(2026, 10, 1, tzinfo=UTC),
        )
    assert list(tmp_path.iterdir()) == []


def test_duplicate_candidates_fail_closed(tmp_path):
    candidate = _candidate()
    responses = _FakeResponses({"candidates": [candidate, candidate]})
    with pytest.raises(ResearchError, match="duplicate"):
        run_quarterly_research(SimpleNamespace(responses=responses), tmp_path)


def test_existing_report_is_never_replaced(tmp_path):
    now = datetime(2026, 10, 1, tzinfo=UTC)
    responses = _FakeResponses({"candidates": [_candidate()]})
    destination = run_quarterly_research(SimpleNamespace(responses=responses), tmp_path, now=now)
    original = destination.read_bytes()

    with pytest.raises(ResearchError, match="already exists"):
        run_quarterly_research(SimpleNamespace(responses=responses), tmp_path, now=now)

    assert destination.read_bytes() == original


def test_conservative_multi_batch_ceiling_stays_below_budget():
    assert conservative_run_ceiling_usd() == Decimal("2.0512")
    assert conservative_run_ceiling_usd() < BUDGET_USD

from __future__ import annotations

from types import SimpleNamespace

from icor.domain.evidence import ConfidenceBand
from icor.domain.generations import GenerationIdentityKind
from icor.domain.snapshots import SnapshotVersions
from scripts.report_snapshot_completeness import report


class Repository:
    def list_completeness_records(self):
        return (
            SimpleNamespace(
                geography="DE", year=2023, evidence_only_count=2,
                forecastable_count=7, rejected_record_count=1,
            ),
        )

    def list_generations(self):
        return (
            SimpleNamespace(identity_kind=GenerationIdentityKind.ESTIMATED),
        )

    def list_generation_assignments(self):
        return (SimpleNamespace(confidence=ConfidenceBand.LOW),)

    def list_cohort_estimates(self):
        return (object(), object())

    def list_opportunity_estimates(self):
        return (object(),)


def test_report_is_path_free_and_exposes_exact_completeness() -> None:
    versions = SnapshotVersions(
        "sources-v1", "identity-v1", "reconcile-v1", "confidence-v1",
        "estimate-v1", "survival-v1", "hazard-v1", "forecast-v1",
        "registry-v1", "resolver-v1",
    )
    manifest = SimpleNamespace(
        snapshot_id="snapshot-real-v1", status=SimpleNamespace(value="candidate"),
        observation_count=10, release_ids=("release-a",), warnings=(),
        versions=versions,
    )

    result = report(manifest, Repository())

    assert result["counts"]["forecastable"] == 7
    assert result["counts"]["evidence_only"] == 2
    assert result["generation_assignment_confidence"] == {"low": 1}
    assert "path" not in str(result).casefold()

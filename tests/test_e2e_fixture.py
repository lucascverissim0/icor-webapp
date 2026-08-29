from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest


def test_browser_fixture_builds_a_reusable_sealed_candidate(tmp_path: Path) -> None:
    from icor.application.evidence_review import EvidenceReviewService
    from icor.application.registrations import RegistrationQuery, RegistrationService
    from scripts.e2e_fixture import prepare_e2e_fixture

    first = prepare_e2e_fixture(tmp_path)
    second = prepare_e2e_fixture(tmp_path)

    assert first == second
    evidence = EvidenceReviewService.from_candidate(first)
    registrations = RegistrationService.from_candidate(first)
    assert evidence.summary().observation_count == 3
    summary = registrations.summary()
    assert summary.model_count == 3
    assert summary.total_registrations == Decimal("600")
    tesla = registrations.ranking(RegistrationQuery(search="Tesla"))
    assert [(row.make, row.model, row.registrations) for row in tesla.items] == [
        ("TESLA", "MODEL Y", Decimal("200"))
    ]


def test_browser_runner_prepares_both_candidates_or_rejects_partial_configuration(
    tmp_path: Path,
) -> None:
    from scripts.run_e2e_dev import fixture_root_for, prepare_environment

    prepared = prepare_environment({}, fixture_root=tmp_path)

    evidence = Path(prepared["ICOR_E2E_EVIDENCE_CANDIDATE"])
    generation = Path(prepared["ICOR_E2E_GENERATION_CANDIDATE"])
    assert evidence == generation
    assert (evidence / "snapshot.json").is_file()
    for partial in (
        {"ICOR_E2E_EVIDENCE_CANDIDATE": str(evidence)},
        {"ICOR_E2E_GENERATION_CANDIDATE": str(generation)},
    ):
        with pytest.raises(ValueError, match="both be configured"):
            prepare_environment(partial, fixture_root=tmp_path)

    explicit = {
        "ICOR_E2E_EVIDENCE_CANDIDATE": "evidence-candidate",
        "ICOR_E2E_GENERATION_CANDIDATE": "generation-candidate",
    }
    assert prepare_environment(explicit, fixture_root=tmp_path) == explicit
    empty = prepare_environment(
        {
            "ICOR_E2E_EVIDENCE_CANDIDATE": "",
            "ICOR_E2E_GENERATION_CANDIDATE": "",
        },
        fixture_root=tmp_path,
    )
    assert Path(empty["ICOR_E2E_EVIDENCE_CANDIDATE"]) == evidence
    assert fixture_root_for(18001, 19001) != fixture_root_for(18002, 19002)

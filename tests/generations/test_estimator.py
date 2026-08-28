from icor.domain.generations import GenerationIdentityKind
from icor.generations.estimator import EstimatedGenerationBuilder


def test_sparse_history_creates_one_stable_broad_estimated_generation() -> None:
    builder = EstimatedGenerationBuilder("estimated-generation-v1")

    first = builder.build(
        canonical_vehicle_id="vehicle-example-alpha-eu",
        market="EU",
        observed_years=(2016, 2018, 2020),
        evidence_ids=("observation-2016", "observation-2018", "observation-2020"),
    )
    second = builder.build(
        canonical_vehicle_id="vehicle-example-alpha-eu",
        market="EU",
        observed_years=(2020, 2016, 2018),
        evidence_ids=("observation-2020", "observation-2016", "observation-2018"),
    )

    assert first == second
    assert len(first) == 1
    assert first[0].display_name == "estimated-generation-1 (2016-2020)"
    assert first[0].identity_kind is GenerationIdentityKind.ESTIMATED

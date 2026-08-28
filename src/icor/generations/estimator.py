"""Conservative stable generation fallback for sparse model histories."""

from __future__ import annotations

from datetime import date

from icor.domain.generations import GenerationEntry, GenerationIdentityKind
from icor.evidence.normalization import stable_evidence_id


class EstimatedGenerationBuilder:
    def __init__(self, version: str) -> None:
        if not version.strip():
            raise ValueError("estimated generation version is required")
        self.version = version

    def build(
        self,
        *,
        canonical_vehicle_id: str,
        market: str,
        observed_years: tuple[int, ...],
        evidence_ids: tuple[str, ...],
    ) -> tuple[GenerationEntry, ...]:
        if not observed_years or any(type(year) is not int for year in observed_years):
            raise ValueError("estimated generation years are required")
        years = tuple(sorted(set(observed_years)))
        evidence = tuple(sorted(set(evidence_ids)))
        if not evidence:
            raise ValueError("estimated generation evidence is required")
        start, end = years[0], years[-1]
        identifier = stable_evidence_id(
            "generation-estimated",
            canonical_vehicle_id,
            market,
            str(start),
            str(end),
            self.version,
        )
        return (
            GenerationEntry(
                generation_id=identifier,
                canonical_vehicle_id=canonical_vehicle_id,
                display_name=f"estimated-generation-1 ({start}-{end})",
                market=market,
                start_month=date(start, 1, 1),
                end_month=date(end, 12, 1),
                identity_kind=GenerationIdentityKind.ESTIMATED,
                body_style=None,
                facelift=None,
                platform=None,
                evidence_ids=evidence,
                dependency_groups=("algorithmic-estimation",),
                confidence_reasons=(
                    "Sparse evidence supports one broad chronological generation only.",
                ),
                registry_version=self.version,
            ),
        )

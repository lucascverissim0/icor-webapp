"""Emit a canonical, path-free completeness report for one verified snapshot."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path

from icor.application.evidence_review import EvidenceReviewService
from icor.domain.generations import GenerationIdentityKind
from icor.evidence.serialization import canonical_json_bytes
from icor.infrastructure.snapshot_store import SnapshotStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report ICOR snapshot completeness.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate", type=Path)
    source.add_argument("--root", type=Path)
    return parser


def _open(args: argparse.Namespace):
    if args.candidate is not None:
        service = EvidenceReviewService.from_candidate(args.candidate)
        from icor.infrastructure.sqlite_evidence_repository import SQLiteEvidenceRepository

        return service.manifest, SQLiteEvidenceRepository(service.database_path)
    return SnapshotStore(args.root).open_active_snapshot()


def report(manifest, repository) -> dict[str, object]:
    completeness = repository.list_completeness_records()
    path = getattr(repository, "path", None)
    if isinstance(path, Path):
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro", uri=True
        )
        connection.row_factory = sqlite3.Row
        try:
            table_counts = {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in (
                    "generation_entry", "generation_assignment", "cohort_estimate",
                    "opportunity_estimate",
                )
            }
            estimated_generations = int(
                connection.execute(
                    "SELECT COUNT(*) FROM generation_entry WHERE identity_kind = ?",
                    (GenerationIdentityKind.ESTIMATED.value,),
                ).fetchone()[0]
            )
            confidence = {
                row["confidence"]: row["count"]
                for row in connection.execute(
                    """SELECT confidence, COUNT(*) AS count
                    FROM generation_assignment GROUP BY confidence ORDER BY confidence"""
                )
            }
        finally:
            connection.close()
        generation_count = table_counts["generation_entry"]
        assignment_count = table_counts["generation_assignment"]
        cohort_count = table_counts["cohort_estimate"]
        opportunity_count = table_counts["opportunity_estimate"]
    else:
        generations = repository.list_generations()
        assignments = repository.list_generation_assignments()
        confidence = Counter(item.confidence.value for item in assignments)
        estimated_generations = sum(
            item.identity_kind is GenerationIdentityKind.ESTIMATED
            for item in generations
        )
        generation_count = len(generations)
        assignment_count = len(assignments)
        cohort_count = len(repository.list_cohort_estimates())
        opportunity_count = len(repository.list_opportunity_estimates())
    return {
        "counts": {
            "assigned_observations": assignment_count,
            "cohorts": cohort_count,
            "completeness_records": len(completeness),
            "estimated_generations": estimated_generations,
            "evidence_only": sum(item.evidence_only_count for item in completeness),
            "forecastable": sum(item.forecastable_count for item in completeness),
            "generations": generation_count,
            "observations": manifest.observation_count,
            "opportunities": opportunity_count,
            "rejected_source_records": sum(
                item.rejected_record_count for item in completeness
            ),
            "sourced_generations": generation_count - estimated_generations,
        },
        "generation_assignment_confidence": dict(sorted(confidence.items())),
        "geographies": sorted({item.geography for item in completeness}),
        "limitations": [
            "Generation labels may be estimated and do not claim manufacturer naming.",
            "Opportunity values are generation-level baselines, not exact fitment demand.",
            "Sparse series remain evidence-only and are excluded from forecasting.",
        ],
        "release_ids": list(manifest.release_ids),
        "snapshot_id": manifest.snapshot_id,
        "status": manifest.status.value,
        "versions": {
            name: getattr(manifest.versions, name)
            for name in (field.name for field in fields(manifest.versions))
        },
        "warnings": list(manifest.warnings),
        "years": sorted({item.year for item in completeness}),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest, repository = _open(args)
        sys.stdout.buffer.write(canonical_json_bytes(report(manifest, repository)))
    except (OSError, RuntimeError, ValueError):
        sys.stderr.write("Verified snapshot completeness is unavailable.\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

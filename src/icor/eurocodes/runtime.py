"""Reviewed Eurocodes for the running app: load, verify and describe them.

The catalogue and the review decisions are ICOR-private files kept outside the
repository. Loading refuses decisions reviewed against another catalogue, since
their codes and row citations would no longer be the catalogue's, and reports —
rather than hides — decisions made for another snapshot or identity-rule version,
whose identities may no longer exist.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from icor.eurocodes.allocation import Allocation, EurocodeDecisions
from icor.eurocodes.catalog import from_document
from icor.evidence.normalization import source_vehicle_display_label

RAW_IDENTITY_PREFIX = "raw"


class _IdentityIndex(Protocol):
    def identity_for(self, vehicle_id: str) -> tuple[str, str] | None: ...


def identity_key(index: _IdentityIndex, vehicle_id: str, make: str, model: str) -> str:
    """The key a vehicle's registrations are reviewed and allocated under.

    A folded identity is keyed by its make and model; a vehicle the identity
    rules cannot fold keeps its raw labels, case- and space-normalised.
    """

    key = index.identity_for(vehicle_id)
    if key is not None:
        return f"{key[0]}|{key[1]}"
    raw_make = " ".join(make.split()).casefold()
    raw_model = " ".join(model.split()).casefold()
    return f"{RAW_IDENTITY_PREFIX}|{raw_make}|{raw_model}"


@dataclass(frozen=True, slots=True)
class CodeSummary:
    """One Eurocode as the app shows it: who carries it and in which years."""

    code: str
    label: str
    families: tuple[str, ...]
    first_year: int
    last_year: int | None
    bodies: tuple[str, ...]
    drive: str


@dataclass(frozen=True, slots=True)
class EurocodeState:
    """What the running app can claim about its Eurocodes."""

    loaded: bool
    catalogue_sha256: str | None = None
    decisions_snapshot_id: str | None = None
    reviewed_identities: int = 0
    orphaned_identities: int = 0
    warnings: tuple[str, ...] = ()


NOT_LOADED = EurocodeState(loaded=False)


def _code_label(make: str, label: str) -> str:
    display_make = source_vehicle_display_label(make) if make else ""
    if not display_make or label.casefold().startswith(display_make.casefold()):
        return label
    return f"{display_make} {label}"


class EurocodeMap:
    """Reviewed decisions joined to the catalogue they were reviewed against."""

    def __init__(
        self,
        decisions: EurocodeDecisions,
        catalogue_codes: Iterable[str],
        catalogue_sha256: str,
    ) -> None:
        self.decisions = decisions
        self.catalogue_sha256 = catalogue_sha256
        self._catalogue_codes = frozenset(catalogue_codes)
        self._summaries = self._summarise()

    @classmethod
    def from_paths(cls, catalogue_path: Path, decisions_path: Path) -> EurocodeMap:
        catalogue = from_document(json.loads(catalogue_path.read_text(encoding="utf-8")))
        document = json.loads(decisions_path.read_text(encoding="utf-8"))
        return cls.from_documents(catalogue, document)

    @classmethod
    def from_documents(cls, catalogue: Any, document: Mapping[str, Any]) -> EurocodeMap:
        reviewed_against = document.get("metadata", {}).get("catalogue_sha256")
        if reviewed_against != catalogue.source_sha256:
            raise ValueError("Eurocode decisions were reviewed against another catalogue")
        rows_by_code = {code: [row.row for row in rows] for code, rows in catalogue.codes().items()}
        decisions = EurocodeDecisions.from_document(document, catalogue_codes=rows_by_code)
        return cls(decisions, rows_by_code, catalogue.source_sha256)

    def _summarise(self) -> dict[str, CodeSummary]:
        labels: dict[str, list[str]] = {}
        families: dict[str, list[str]] = {}
        spans: dict[str, tuple[int, int | None]] = {}
        bodies: dict[str, set[str]] = {}
        drives: dict[str, set[str]] = {}
        for key, family in sorted(self.decisions.families.items()):
            family_spans = self.decisions.timeline(key).spans()
            for fact in family.codes:
                label = _code_label(family.make, fact.label)
                if label not in labels.setdefault(fact.code, []):
                    labels[fact.code].append(label)
                families.setdefault(fact.code, []).append(key)
                first, last = family_spans[fact.code]
                if fact.code in spans:
                    known_first, known_last = spans[fact.code]
                    first = min(first, known_first)
                    last = None if last is None or known_last is None else max(last, known_last)
                spans[fact.code] = (first, last)
                bodies.setdefault(fact.code, set()).update(fact.bodies)
                drives.setdefault(fact.code, set()).add(fact.drive)
        return {
            code: CodeSummary(
                code=code,
                label=" / ".join(labels[code]),
                families=tuple(families[code]),
                first_year=spans[code][0],
                last_year=spans[code][1],
                bodies=tuple(sorted(bodies[code])),
                drive=next(iter(drives[code])) if len(drives[code]) == 1 else "both",
            )
            for code in labels
        }

    def is_catalogue_code(self, code: str) -> bool:
        return code in self._catalogue_codes

    def summary(self, code: str) -> CodeSummary | None:
        return self._summaries.get(code)

    def summaries(self) -> Mapping[str, CodeSummary]:
        return self._summaries

    def allocate(
        self,
        identity: str,
        year: int,
        geography: str,
        raw_body: str | None = None,
        *,
        raw_model: str | None = None,
    ) -> Allocation:
        """The codes a registration may carry.

        ``raw_model`` is the publisher's model label (``canonical_vehicle.model``).
        Without it no reviewed variant applies: spelling routes, model-range pools
        and year-limited pools are all keyed on that label.
        """

        return self.decisions.allocate(identity, year, geography, raw_body, raw_model=raw_model)

    def state(
        self,
        *,
        snapshot_id: str,
        identity_rules_sha256: str,
        present_identities: Iterable[str],
    ) -> EurocodeState:
        metadata = self.decisions.metadata
        warnings: list[str] = []
        if metadata.get("snapshot_id") != snapshot_id:
            warnings.append("Eurocode decisions were reviewed for another snapshot.")
        if metadata.get("identity_rules_sha256") != identity_rules_sha256:
            warnings.append("Eurocode decisions were reviewed under other identity rules.")
        present = set(present_identities)
        orphaned = sum(1 for identity in self.decisions.identities if identity not in present)
        if orphaned:
            warnings.append(f"{orphaned} reviewed identities no longer exist in this snapshot.")
        return EurocodeState(
            loaded=True,
            catalogue_sha256=self.catalogue_sha256,
            decisions_snapshot_id=metadata.get("snapshot_id"),
            reviewed_identities=len(self.decisions.identities),
            orphaned_identities=orphaned,
            warnings=tuple(warnings),
        )

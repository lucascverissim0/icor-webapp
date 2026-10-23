"""The decisions a client image may carry.

``decisions.json`` is ICOR-private review work, and a reviewer can add any key to
any level of it (a note, a rationale, a source quotation). The client map is
therefore re-serialised from the fields the loader reads (``EurocodeDecisions``
``from_document`` and its ``_family`` and ``_identity`` parsers), not copied: a
key outside these lists never ships, and the preflight refuses a map that has one.

``why`` is read by the loader but only stored: no allocation, ranking or display
uses it. It is where a reviewer explains a decision, so the client map leaves it
out; nothing served changes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from typing import Any

TOP_KEYS = ("format", "version", "metadata", "families", "identities")
#: What `EurocodeMap` and the snapshot checks read; the review's own `review` and
#: `rounds` entries describe how the decisions were made and are not served.
METADATA_KEYS = ("catalogue_sha256", "snapshot_id", "identity_rules_sha256")
FAMILY_KEYS = ("label", "make", "codes")
CODE_KEYS = (
    "code",
    "label",
    "first_year",
    "first_year_latest",
    "last_year",
    "bodies",
    "role",
    "drive",
    "replaces",
    "rows",
    "uncoded_successor",
    "uncoded_successor_latest",
    "uncoded_sibling",
)
IDENTITY_KEYS = ("status", "family", "body", "variants")
VARIANT_KEYS = ("match", "status", "family", "body", "geographies", "years")


def _keep(item: Any, allowed: tuple[str, ...], level: str, dropped: Counter[str]) -> Any:
    if not isinstance(item, Mapping):
        return item
    dropped[level] += sum(1 for key in item if key not in allowed)
    return {key: value for key, value in item.items() if key in allowed}


def _project(
    document: Mapping[str, Any], identities: Collection[str] | None, dropped: Counter[str]
) -> dict[str, Any]:
    result = _keep(document, TOP_KEYS, "document", dropped)
    if "metadata" in result:
        result["metadata"] = _keep(result["metadata"], METADATA_KEYS, "metadata", dropped)
    if isinstance(result.get("families"), Mapping):
        families: dict[str, Any] = {}
        for key, item in result["families"].items():
            family = _keep(item, FAMILY_KEYS, "family", dropped)
            if isinstance(family, dict) and isinstance(family.get("codes"), list):
                family["codes"] = [
                    _keep(raw, CODE_KEYS, "code", dropped) for raw in family["codes"]
                ]
            families[key] = family
        result["families"] = families
    if isinstance(result.get("identities"), Mapping):
        kept: dict[str, Any] = {}
        for key, item in result["identities"].items():
            if identities is not None and key not in identities:
                continue
            decision = _keep(item, IDENTITY_KEYS, "identity", dropped)
            if isinstance(decision, dict) and isinstance(decision.get("variants"), list):
                decision["variants"] = [
                    _keep(raw, VARIANT_KEYS, "variant", dropped) for raw in decision["variants"]
                ]
            kept[key] = decision
        result["identities"] = kept
    return result


def client_decisions(
    document: Mapping[str, Any], *, identities: Collection[str] | None = None
) -> dict[str, Any]:
    """``document`` reduced to the fields the loader reads, for only ``identities`` if given."""

    return _project(document, identities, Counter())


def foreign_keys(document: Mapping[str, Any]) -> dict[str, int]:
    """How many keys at each level of ``document`` fall outside the client allowlist.

    Counts only, by level: the keys are the reviewer's, and so may be what they wrote.
    """

    dropped: Counter[str] = Counter()
    _project(document, None, dropped)
    return {level: count for level, count in dropped.items() if count}

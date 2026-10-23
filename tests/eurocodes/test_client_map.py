"""The allowlist behind the client map must follow what the loader reads."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import textwrap
from typing import Any

import pytest

from icor.eurocodes import allocation, client_map
from icor.eurocodes.demo import _decisions

#: Read by the loader but only stored; the client map leaves them out on purpose.
RATIONALE = {"why"}


def _read_keys(*functions: Any) -> set[str]:
    """Every literal key a loader function looks up with ``.get("x")`` or ``["x"]``."""

    keys: set[str] = set()
    for function in functions:
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(function)))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                keys.add(node.slice.value)
    return keys


def test_the_allowlist_is_exactly_what_the_loader_reads() -> None:
    """A key the loader starts reading must be allowed (or its omission decided) here."""

    read = _read_keys(
        allocation.EurocodeDecisions.from_document, allocation._family, allocation._identity
    )
    allowed = {
        *client_map.TOP_KEYS,
        *client_map.FAMILY_KEYS,
        *client_map.CODE_KEYS,
        *client_map.IDENTITY_KEYS,
        *client_map.VARIANT_KEYS,
    }

    assert read - RATIONALE == allowed


def _full() -> dict[str, Any]:
    """A document that uses every field the loader reads, over the demo map."""

    document = _decisions()
    code = document["families"]["aurora/a1-horizon"]["codes"][1]
    code.update(
        {
            "bodies": ["hatchback"],
            "role": "main",
            "drive": "both",
            "uncoded_successor": 2028,
            "uncoded_successor_latest": 2029,
            "uncoded_sibling": 2027,
        }
    )
    document["identities"]["aurora mobility|a1 horizon"].update(
        {
            "body": "hatchback",
            "variants": [
                {
                    "match": "meridian",
                    "status": "mapped",
                    "family": "meridian/m5-vector",
                    "body": "suv",
                    "geographies": ["DE"],
                    "years": [2024, 2030],
                }
            ],
        }
    )
    document["identities"]["velora works|v3 current"] = {"status": "not_a_vehicle"}
    return document


def _strewn(value: Any) -> Any:
    """``value`` with a free-text key added to every mapping in it."""

    if isinstance(value, dict):
        return {**{key: _strewn(item) for key, item in value.items()}, "note": "free text"}
    if isinstance(value, list):
        return [_strewn(item) for item in value]
    return value


def test_a_document_of_known_fields_is_reserialised_unchanged() -> None:
    document = _full()

    assert client_map.client_decisions(document) == document
    assert client_map.foreign_keys(document) == {}


def test_unknown_keys_are_dropped_at_every_level_and_counted_by_level() -> None:
    document = _full()
    strewn = copy.deepcopy(document)
    strewn["note"] = "free text"
    strewn["metadata"]["note"] = "free text"
    for family in strewn["families"].values():
        family["note"] = "free text"
        for code in family["codes"]:
            code["note"] = "free text"
    for decision in strewn["identities"].values():
        decision["note"] = "free text"
        decision["why"] = "free text"
        for variant in decision.get("variants", ()):
            variant["note"] = "free text"
            variant["why"] = "free text"

    assert client_map.client_decisions(strewn) == document
    assert client_map.foreign_keys(strewn) == {
        "document": 1,
        "metadata": 1,
        "family": 3,
        "code": 4,
        "identity": 6,
        "variant": 2,
    }


def test_only_the_named_identities_are_kept() -> None:
    kept = client_map.client_decisions(_full(), identities={"meridian motors|m5 vector"})

    assert list(kept["identities"]) == ["meridian motors|m5 vector"]
    assert len(kept["families"]) == 3


def test_the_counts_name_no_key() -> None:
    document = _full()
    document["identities"]["velora works|v3 current"]["a reviewer's quotation"] = 1

    assert "quotation" not in json.dumps(client_map.foreign_keys(document))


@pytest.mark.parametrize("shape", [None, [], "text", 7])
def test_a_malformed_section_is_not_mistaken_for_a_map(shape: object) -> None:
    document = {"format": "x", "families": shape, "identities": shape, "metadata": shape}

    assert client_map.foreign_keys(document) == {}
    assert client_map.client_decisions(document)["families"] == shape

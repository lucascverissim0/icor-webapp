from __future__ import annotations

import json
from pathlib import Path

import pytest

from icor.eurocodes.demo import write_demo_eurocodes
from icor.eurocodes.runtime import EurocodeMap
from scripts import build_client_eurocodes as builder


def test_the_client_map_serves_the_private_one_with_only_the_files_it_loads(
    tmp_path: Path,
) -> None:
    source = write_demo_eurocodes(tmp_path / "private")
    (source / "reports").mkdir()
    (source / "reports" / "data-coverage-x.json").write_text("{}", encoding="utf-8")

    result = builder.build(source, tmp_path / "client")

    client = tmp_path / "client"
    assert sorted(path.name for path in client.iterdir()) == ["catalog.json", "decisions.json"]
    assert EurocodeMap.from_directory(client).summaries() == (
        EurocodeMap.from_directory(source).summaries()
    )
    rows = json.loads((client / "catalog.json").read_text(encoding="utf-8"))["rows"]
    assert rows and all(row["description"] == "" for row in rows)
    assert result["codes"] == len(EurocodeMap.from_directory(source).summaries())
    assert not list(tmp_path.glob(".client-eurocode-*"))


def test_a_failed_build_leaves_the_previous_map_and_no_staging(tmp_path: Path) -> None:
    source = write_demo_eurocodes(tmp_path / "private")
    builder.build(source, tmp_path / "client")
    before = (tmp_path / "client" / "decisions.json").read_bytes()
    document = json.loads((source / "decisions.json").read_text(encoding="utf-8"))
    document["metadata"]["catalogue_sha256"] = "f" * 64
    (source / "decisions.json").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="another catalogue"):
        builder.build(source, tmp_path / "client")

    assert (tmp_path / "client" / "decisions.json").read_bytes() == before
    assert not list(tmp_path.glob(".client-eurocode-*"))


def test_identities_the_client_snapshot_does_not_forecast_are_left_out(tmp_path: Path) -> None:
    """They carry no units; kept, they only show clients a review warning."""

    source = write_demo_eurocodes(tmp_path / "private")
    private = EurocodeMap.from_directory(source)
    identities = sorted(private.decisions.identities)
    ranked = frozenset(identities[1:])

    result = builder.build(source, tmp_path / "client", ranked=ranked)

    client = EurocodeMap.from_directory(tmp_path / "client")
    assert set(client.decisions.identities) == set(ranked)
    assert result["unranked_identities_left_out"] == 1
    assert client.summaries() == private.summaries()
    for identity in ranked:
        assert client.decisions.identities[identity] == private.decisions.identities[identity]
    state = client.state(
        snapshot_id=str(private.decisions.metadata.get("snapshot_id")),
        identity_rules_sha256=str(private.decisions.metadata.get("identity_rules_sha256")),
        present_identities=ranked,
    )
    assert state.orphaned_identities == 0

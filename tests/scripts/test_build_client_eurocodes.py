from __future__ import annotations

import json
from pathlib import Path

import pytest

from icor.eurocodes.demo import write_demo_eurocodes
from icor.eurocodes.runtime import EurocodeMap
from scripts import build_client_eurocodes as builder

FREE_TEXT = "free text"


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


def _walk(value: object):
    """Every mapping key and string leaf in a JSON document."""

    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    elif isinstance(value, str):
        yield value


def _annotate(source: Path) -> None:
    """Free text a reviewer might add at every level of the private decisions."""

    path = source / "decisions.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["note"] = FREE_TEXT
    document["metadata"]["rounds"] = [FREE_TEXT]
    for family in document["families"].values():
        family["note"] = FREE_TEXT
        for code in family["codes"]:
            code["rationale"] = FREE_TEXT
    for item in document["identities"].values():
        item["note"] = FREE_TEXT
        item["why"] = FREE_TEXT
    document["identities"]["aurora mobility|a1 horizon"]["variants"] = [
        {
            "match": "meridian",
            "family": "meridian/m5-vector",
            "geographies": ["DE"],
            "years": [2024, 2030],
            "why": FREE_TEXT,
            "note": FREE_TEXT,
        }
    ]
    path.write_text(json.dumps(document), encoding="utf-8")


def test_free_text_a_reviewer_adds_never_reaches_the_client_map(tmp_path: Path) -> None:
    source = write_demo_eurocodes(tmp_path / "private")
    _annotate(source)

    builder.build(source, tmp_path / "client")

    written = json.loads((tmp_path / "client" / "decisions.json").read_text(encoding="utf-8"))
    assert FREE_TEXT not in set(_walk(written))
    assert not {"note", "rationale", "rounds", "why"} & set(_walk(written))
    private, client = (EurocodeMap.from_directory(path) for path in (source, tmp_path / "client"))
    assert client.summaries() == private.summaries()
    cells = [
        (identity, year, geography, model)
        for identity in private.decisions.identities
        for year in range(2018, 2028)
        for geography in ("DE", "GB")
        for model in (None, "A1 Horizon", "Meridian M5")
    ]
    assert len({private.allocate(i, y, g, raw_model=m).status for i, y, g, m in cells}) > 1
    for identity, year, geography, model in cells:
        assert client.allocate(identity, year, geography, raw_model=model) == private.allocate(
            identity, year, geography, raw_model=model
        )


def _refused(source: Path, destination: Path, reason: str) -> None:
    with pytest.raises(SystemExit, match=reason):
        builder.build(source, destination)
    assert sorted(path.name for path in source.iterdir()) == ["catalog.json", "decisions.json"]
    assert not list(source.parent.glob(".client-eurocode-*"))


@pytest.fixture
def private_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(builder, "PRIVATE", tmp_path / ".local" / "private")
    return tmp_path / ".local" / "private"


def test_a_destination_that_is_the_source_is_refused(tmp_path: Path) -> None:
    source = write_demo_eurocodes(tmp_path / "private")

    _refused(source, source, "is the source")
    _refused(source, tmp_path / "other" / ".." / "private", "is the source")


def test_a_destination_inside_the_source_is_refused(tmp_path: Path) -> None:
    source = write_demo_eurocodes(tmp_path / "private")

    _refused(source, source / "client", "inside the source")


def test_a_destination_containing_the_source_is_refused(tmp_path: Path) -> None:
    source = write_demo_eurocodes(tmp_path / "work" / "private")

    _refused(source, tmp_path / "work", "contains the source")


def test_a_destination_reached_through_a_link_to_the_source_is_refused(tmp_path: Path) -> None:
    source = write_demo_eurocodes(tmp_path / "private")
    link = tmp_path / "client"
    try:
        link.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are not available")

    _refused(source, link, "is the source")


def test_a_destination_in_the_private_folder_is_refused(
    tmp_path: Path, private_folder: Path
) -> None:
    source = write_demo_eurocodes(tmp_path / "elsewhere")
    private_folder.mkdir(parents=True)

    _refused(source, private_folder, "private folder")
    _refused(source, private_folder / "client", "private folder")


def test_a_destination_containing_the_private_folder_is_refused(
    tmp_path: Path, private_folder: Path
) -> None:
    source = write_demo_eurocodes(tmp_path / "elsewhere")
    private_folder.mkdir(parents=True)

    _refused(source, private_folder.parent, "private folder")


def test_a_swap_that_fails_puts_the_previous_map_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_demo_eurocodes(tmp_path / "private")
    destination = tmp_path / "client"
    builder.build(source, destination)
    before = (destination / "decisions.json").read_bytes()
    rename = Path.rename

    def fail_to_install(self: Path, target: Path) -> Path:
        if Path(target) == destination and not self.name.endswith("-retired"):
            raise OSError("the volume went away")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_to_install)

    with pytest.raises(OSError, match="went away"):
        builder.build(source, destination)

    assert (destination / "decisions.json").read_bytes() == before
    assert not list(tmp_path.glob(".client-eurocode-*"))

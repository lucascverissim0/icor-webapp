from __future__ import annotations

import io
from pathlib import Path

import pytest

from icor.evidence.register_aggregate import (
    HashingReader,
    cohort_of,
    collapse,
    sha256_file,
    write_aggregate,
)


def test_the_aggregate_is_sorted_counts_only(tmp_path: Path) -> None:
    destination = tmp_path / "fleet.csv"

    result = write_aggregate(
        {("ZETA", "Z1", "2012"): 3, ("ALPHA", "A2", ""): 1, ("ALPHA", "A2", "2009"): 5},
        destination,
    )

    assert destination.read_text(encoding="utf-8") == (
        "make,model,cohort,vehicles\nALPHA,A2,2009,5\nALPHA,A2,,1\nZETA,Z1,2012,3\n"
    )
    assert (result.group_count, result.vehicles) == (3, 9)
    assert result.sha256 == sha256_file(destination)
    assert result.artifact_bytes == destination.stat().st_size


def test_an_empty_or_non_positive_aggregate_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no passenger cars"):
        write_aggregate({}, tmp_path / "empty.csv")
    with pytest.raises(ValueError, match="positive"):
        write_aggregate({("A", "B", "2010"): 0}, tmp_path / "zero.csv")
    assert list(tmp_path.iterdir()) == []


def test_labels_and_cohorts_are_normalised() -> None:
    assert collapse("  Škoda   Octavia ") == "Škoda Octavia"
    assert collapse(None) == ""
    assert cohort_of("2014") == "2014"
    assert [cohort_of(value) for value in ("", None, "14", "２０１４", "201a")] == [""] * 5


def test_the_reader_hashes_what_passes_and_enforces_its_limit() -> None:
    reader = HashingReader(io.BytesIO(b"abcdef"))
    assert reader.read(4) + reader.read() == b"abcdef"
    assert reader.bytes_read == 6
    assert reader.sha256 == "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"

    limited = HashingReader(io.BytesIO(b"abcdef"), limit=5)
    with pytest.raises(ValueError, match="byte limit"):
        limited.read()

"""Report the method that produced the rows a channel serves.

Four separate defects in this repository have had the same shape: a channel
reported a method it reconstructed from application code rather than one read
from the data it was serving, and two pages then disagreed about the same
vehicle. The numbers were never wrong; the claim about how they were produced
was.

The rule this module exists to enforce is that any provenance a channel reports
is read from the rows it serves. When those rows disagree among themselves the
answer is to say so, not to pick a winner: a snapshot mixing methods is a build
problem, and hiding it behind a plausible single value is how the previous
instances survived two rounds of fixing.
"""

from __future__ import annotations

from collections.abc import Iterable

MIXED_PREFIX = "mixed:"


def resolve_reported_method(values: Iterable[object], *, label: str) -> str:
    """Collapse the method recorded on a set of served rows into one report.

    Raises when there is nothing to report, because a channel that cannot say
    how a number was produced must not quietly claim a default.
    """

    methods = {str(value) for value in values if value is not None and str(value).strip()}
    if not methods:
        raise ValueError(f"served rows carry no {label}")
    if len(methods) > 1:
        return MIXED_PREFIX + ",".join(sorted(methods))
    return next(iter(methods))


def is_mixed(method: str) -> bool:
    """Whether a reported method describes rows that disagreed."""

    return method.startswith(MIXED_PREFIX)

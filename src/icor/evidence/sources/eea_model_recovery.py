"""Name EEA registrations whose rows carry no commercial name, from their type approval.

Some member states report a year's registrations without a commercial name (Cn):
Denmark left 57% of 2022 unnamed. Most such rows still carry the make and the
type approval (TAN, type, variant, version), which the same year's named rows
share. A name is taken over only when one label holds at least 95% of at least
20 named registrations under the most specific key that exists, from
(MS, Mk, TAN, T, Va, Ve) down to (Mk, TAN, T); a make is never crossed and a
type approval alone is never enough. Pooling neighbouring years was measured and
names fewer cars (it dilutes the share), so a release recovers from itself only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal

from icor.evidence.normalization import normalize_vehicle_label

MIN_SHARE = Decimal("0.95")
MIN_SUPPORT = 20
# Positions in the loaders' key (MS, Mk, Cn, TAN, T, Va, Ve, Ft).
_LEVELS = (
    ("MS+Mk+TAN+T+Va+Ve", (0, 1, 3, 4, 5, 6)),
    ("Mk+TAN+T+Va+Ve", (1, 3, 4, 5, 6)),
    ("Mk+TAN+T+Va", (1, 3, 4, 5)),
    ("Mk+TAN+T", (1, 3, 4)),
)


@dataclass(frozen=True, slots=True)
class Recovery:
    model: str
    level: str
    support: int
    share: Decimal


class ModelRecovery:
    """Named registrations by technical key, to name the unnamed rows of one release."""

    def __init__(self) -> None:
        self._labels: list[dict[tuple[str, ...], Counter[str]]] = [
            defaultdict(Counter) for _ in _LEVELS
        ]
        self._spellings: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    @staticmethod
    def is_candidate(key: tuple[str, ...]) -> bool:
        """A row with a country and a make but no commercial name."""

        return (
            normalize_vehicle_label(key[0]) is not None
            and normalize_vehicle_label(key[1]) is not None
            and normalize_vehicle_label(key[2]) is None
        )

    def add_named(self, key: tuple[str, ...], registrations: int) -> None:
        model = normalize_vehicle_label(key[2])
        if model is None:
            raise ValueError("only named rows teach recovery")
        self._spellings[key[1], model][key[2]] += registrations
        for index, (_name, positions) in enumerate(_LEVELS):
            parts = tuple(key[position] for position in positions)
            if all(parts):
                self._labels[index][parts][model] += registrations

    def resolve(self, key: tuple[str, ...]) -> Recovery | None:
        """The name the most specific known key gives, or None when it is not clear."""

        for index, (name, positions) in enumerate(_LEVELS):
            parts = tuple(key[position] for position in positions)
            if not all(parts):
                continue
            labels = self._labels[index].get(parts)
            if not labels:
                continue
            support = sum(labels.values())
            model, top = min(labels.items(), key=lambda item: (-item[1], item[0]))
            share = Decimal(top) / Decimal(support)
            if support < MIN_SUPPORT or share < MIN_SHARE:
                return None
            spellings = self._spellings[key[1], model]
            spelling = min(spellings.items(), key=lambda item: (-item[1], item[0]))[0]
            return Recovery(spelling, name, support, share.quantize(Decimal("0.0001")))
        return None

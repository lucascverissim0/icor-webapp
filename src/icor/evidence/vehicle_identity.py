"""Resolve raw publisher vehicle labels to one canonical identity at query time.

`canonical_vehicle` stores whatever each publisher wrote. In the active snapshot
that is 557 distinct `make` strings over 22,892 make/model pairs: Volkswagen
appears 14 ways (`VOLKSWAGEN`, `vw`, `volkswagen, vw`, `volkswagen v w`, a
`volksawgen` typo), and Audi carries 1,194 "models" that are really trim strings
(`a4 / 2.0 / tdi q aut.`, `golf se navigation tdi s-a`).

Every one of those spellings is its own silo: a selection resolved only the exact
pair, so the dropdowns repeated brands, a trim silo offered a truncated year
range, and markets held by a different spelling reported no data. This module
folds the silos back together without rewriting the 9 GB snapshot.

The rules are deliberately asymmetric. Under-merging leaves a duplicate in a
dropdown; over-merging sums two different windshields into one forecast. Where a
label cannot be resolved safely it is kept as itself, and where it names two
vehicles it is dropped.

The reviewed rules live in `data/vehicle_identity.json` so they can be audited.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from icor.evidence.normalization import (
    normalize_vehicle_label,
    source_vehicle_display_label,
)

_RULES_PATH = Path(__file__).resolve().parents[3] / "data" / "vehicle_identity.json"
_SEPARATORS = re.compile(r"[./\\|]+")
_PUNCTUATION = re.compile(r"[^0-9a-z!+\- ]+")
# A model keeps dots inside a token, so VW's ID.3 and ID.4 stay different cars.
_MODEL_SEPARATORS = re.compile(r"[/\\|]+")
_MODEL_PUNCTUATION = re.compile(r"[^0-9a-z!+.\- ]+")
_TRAILING_DOT = re.compile(r"\.+(?=\s|$)")
_ENGINE_SIZE = re.compile(r"\A\d+(?:[.,]\d+)?\Z")
_YEAR = re.compile(r"\b(19[8-9]\d|20[0-9]\d|2100)\b")
_MULTI_MODEL = re.compile(r"[0-9a-z]\s*[,&]\s*[0-9a-z]")
_EARLIEST_YEAR = 1980
_LATEST_YEAR = 2100


@dataclass(frozen=True, slots=True)
class _Rules:
    corporate_suffixes: frozenset[str]
    make_aliases: dict[str, str]
    trim_vocabulary_max_tokens: int
    passenger_car_volume_share: float
    variant_volume_share: float


@lru_cache(maxsize=1)
def _rules() -> _Rules:
    document = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    return _Rules(
        corporate_suffixes=frozenset(document["corporate_suffixes"]),
        make_aliases=dict(document["make_aliases"]),
        trim_vocabulary_max_tokens=int(document.get("trim_vocabulary_max_tokens", 3)),
        passenger_car_volume_share=float(document.get("passenger_car_volume_share", 2e-4)),
        variant_volume_share=float(document.get("variant_volume_share", 5e-3)),
    )


def _without_diacritics(value: str) -> str:
    """Fold accents, so `Škoda` and `Skoda` are one make rather than two silos.

    Removing the accented letter instead of folding it turned `Škoda` into
    `koda`, a make of its own holding a real share of the Czech registrations.
    """

    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _compact(value: str) -> str:
    """Lower-case and remove presentation punctuation, keeping token identity."""

    normalized = normalize_vehicle_label(value)
    if normalized is None:
        return ""
    spaced = _SEPARATORS.sub(" ", _without_diacritics(normalized))
    cleaned = _PUNCTUATION.sub(" ", spaced.replace(",", " "))
    return " ".join(cleaned.replace("-", " - ").split())


def _merge_initialisms(tokens: list[str]) -> list[str]:
    """Join runs of single letters, so `volkswagen v w` reads as `volkswagen vw`."""

    merged: list[str] = []
    run: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if run:
            merged.append("".join(run))
            run = []
        merged.append(token)
    if run:
        merged.append("".join(run))
    return merged


def _base_fold(value: str) -> str:
    tokens = _merge_initialisms([token for token in _compact(value).split() if token != "-"])
    suffixes = _rules().corporate_suffixes
    while len(tokens) > 1 and tokens[-1] in suffixes:
        tokens.pop()
    folded = " ".join(tokens)
    aliases = _rules().make_aliases
    return aliases.get(folded, aliases.get(folded.replace("-", " "), folded))


def fold_make(value: str) -> str | None:
    """Fold one raw make label to its canonical make, or `None` when it is empty.

    Two-token forms are collapsed only when both halves name the same make, so
    `volkswagen vw` and the `volksawgen vw` typo become `volkswagen` while
    `volkswagen knaus` and `mercedes-benz hymer` stay separate: those are
    coachbuilders, and their vehicles are not the parent make's vehicles.
    """

    if type(value) is not str:
        raise ValueError("vehicle make must be text")
    folded = _base_fold(value)
    if not folded:
        return None
    tokens = folded.split()
    if len(tokens) == 2:
        head, tail = _base_fold(tokens[0]), _base_fold(tokens[1])
        if head and head == tail:
            return head
    return folded


def identity_display_label(canonical: str) -> str:
    """Present a folded identity, restoring acronyms (`bmw` reads as `BMW`)."""

    return source_vehicle_display_label(canonical)


@dataclass(frozen=True, slots=True)
class ParsedVehicleQuery:
    """One free-text search, split into the parts the catalogue can match."""

    text: str
    make: str | None
    tokens: tuple[str, ...]
    year: int | None


def parse_vehicle_query(text: str) -> ParsedVehicleQuery:
    """Split `VW Golf 2020` into make, model tokens and a registration year.

    The previous search sent the whole string to one `LIKE '%vw golf 2020%'`, so
    a year made every query return nothing and `VW` never reached `Volkswagen`.
    """

    if type(text) is not str:
        raise ValueError("vehicle query must be text")
    compact = _compact(text)
    year: int | None = None
    match = _YEAR.search(compact)
    if match is not None:
        candidate = int(match.group(0))
        if _EARLIEST_YEAR <= candidate <= _LATEST_YEAR:
            year = candidate
            compact = f"{compact[: match.start()]} {compact[match.end() :]}"
    tokens = [token for token in compact.split() if token not in {"-", ""}]
    make: str | None = None
    aliases = _rules().make_aliases
    known = set(aliases) | set(aliases.values())
    for size in (2, 1):
        if len(tokens) > size and " ".join(tokens[:size]) in known:
            make = fold_make(" ".join(tokens[:size]))
            tokens = tokens[size:]
            break
    return ParsedVehicleQuery(text.strip(), make, tuple(tokens), year)


@dataclass(frozen=True, slots=True)
class VehicleIdentity:
    """One canonical vehicle, as every publisher's spellings agree it exists."""

    make: str
    model: str
    display_make: str
    display_model: str
    volume: float


class VehicleIdentityIndex:
    """Canonical make/model identities derived from one snapshot's own labels.

    The base-model vocabulary is learned from the data rather than curated: a
    label that a publisher wrote on its own is a real model name, so a longer
    label starting with it is that model plus trim. `golf sportsvan` survives as
    its own identity because it too was written on its own and carries real
    volume, while `golf se navigation tdi s-a` folds onto `golf`.
    """

    def __init__(
        self,
        identities: dict[tuple[str, str], VehicleIdentity],
        vehicle_ids: dict[tuple[str, str], tuple[str, ...]],
        make_volume: dict[str, float],
        total_volume: float,
    ) -> None:
        self._identities = identities
        self._vehicle_ids = vehicle_ids
        self._make_volume = make_volume
        self._total_volume = total_volume
        self._identity_by_vehicle = {
            vehicle_id: key
            for key, vehicle_ids_for_key in vehicle_ids.items()
            for vehicle_id in vehicle_ids_for_key
        }

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[str, str, str, float]]) -> VehicleIdentityIndex:
        """Build the index from `(vehicle_id, make, model, registrations)` rows."""

        cleaned: list[tuple[str, str, str, float]] = []
        make_volume: dict[str, float] = {}
        for vehicle_id, raw_make, raw_model, volume in rows:
            make = fold_make(raw_make)
            if make is None:
                continue
            label = _clean_model(make, raw_model)
            if label is None:
                continue
            amount = float(volume or 0.0)
            cleaned.append((vehicle_id, make, label, amount))
            make_volume[make] = make_volume.get(make, 0.0) + amount

        label_volume: dict[tuple[str, str], float] = {}
        for _vehicle_id, make, label, amount in cleaned:
            label_volume[(make, label)] = label_volume.get((make, label), 0.0) + amount

        vocabulary = _vocabulary(label_volume, make_volume)
        identities: dict[tuple[str, str], VehicleIdentity] = {}
        vehicle_ids: dict[tuple[str, str], list[str]] = {}
        for vehicle_id, make, label, amount in cleaned:
            model = _resolve(label, vocabulary.get(make, frozenset()))
            key = (make, model)
            vehicle_ids.setdefault(key, []).append(vehicle_id)
            existing = identities.get(key)
            identities[key] = VehicleIdentity(
                make,
                model,
                identity_display_label(make),
                identity_display_label(model),
                (existing.volume if existing else 0.0) + amount,
            )
        return cls(
            identities,
            {key: tuple(values) for key, values in vehicle_ids.items()},
            make_volume,
            sum(make_volume.values()),
        )

    def identities(self) -> tuple[VehicleIdentity, ...]:
        return tuple(self._identities.values())

    def makes(self, *, include_all: bool = False) -> tuple[str, ...]:
        """Display labels for every canonical make, alphabetically.

        By default only makes carrying real passenger-car volume are returned.
        The snapshot's registration sources also cover farm machinery, quads and
        motorcycles (`AGRIFAC`, `AEON`, `AJS`), which belong behind `include_all`
        rather than at the top of a windshield planner's brand list. Nothing is
        deleted: `include_all` still returns them.
        """

        floor = self._total_volume * _rules().passenger_car_volume_share
        makes = {
            identity.display_make
            for identity in self._identities.values()
            if include_all or self._make_volume.get(identity.make, 0.0) >= floor
        }
        return tuple(sorted(makes, key=str.casefold))

    def models(self, make: str) -> tuple[str, ...]:
        """Display labels for every canonical model of one make, alphabetically."""

        folded = fold_make(make)
        models = {
            identity.display_model
            for identity in self._identities.values()
            if identity.make == folded
        }
        return tuple(sorted(models, key=str.casefold))

    def vehicle_ids(self, make: str, model: str) -> tuple[str, ...]:
        """Every raw vehicle id stored under any spelling of this identity."""

        return self._vehicle_ids.get(self._key(make, model), ())

    def volume(self, make: str, model: str | None = None) -> float:
        if model is None:
            return self._make_volume.get(fold_make(make) or "", 0.0)
        identity = self._identities.get(self._key(make, model))
        return identity.volume if identity else 0.0

    def identity(self, make: str, model: str) -> VehicleIdentity | None:
        return self._identities.get(self._key(make, model))

    def identity_for(self, vehicle_id: str) -> tuple[str, str] | None:
        """The canonical identity one raw vehicle id belongs to.

        Lets a population keyed by raw vehicle id be grouped by real vehicle, so
        a car whose rows are split across publisher spellings is counted once at
        its full size rather than several times at a fraction of it.
        """

        return self._identity_by_vehicle.get(vehicle_id)

    def match(
        self, query: ParsedVehicleQuery, *, limit: int | None = None
    ) -> tuple[VehicleIdentity, ...]:
        """Identities matching every token of the query, best first.

        Matching is an AND over tokens rather than one contiguous substring, so
        word order and a stray trim word no longer decide whether a real vehicle
        is findable.
        """

        make, tokens = query.make, list(query.tokens)
        if make is None and tokens:
            for size in (2, 1):
                candidate = fold_make(" ".join(tokens[:size]))
                if len(tokens) > size and candidate in self._make_volume:
                    make, tokens = candidate, tokens[size:]
                    break
        if make is None and not tokens:
            return ()
        wanted = tuple(tokens)
        joined = " ".join(wanted)
        matches = [
            identity
            for identity in self._identities.values()
            if (make is None or identity.make == make)
            and all(
                token in f"{identity.make} {identity.model}".split()
                or token in identity.model
                for token in wanted
            )
        ]
        matches.sort(
            key=lambda identity: (
                identity.model != joined,
                not identity.model.startswith(joined),
                -identity.volume,
                identity.make,
                identity.model,
            )
        )
        return tuple(matches if limit is None else matches[:limit])

    def _key(self, make: str, model: str) -> tuple[str, str]:
        folded = fold_make(make) or ""
        label = normalize_vehicle_label(model) or ""
        return (folded, label)


def _clean_model(make: str, raw: str) -> str | None:
    """Strip a repeated make, a trim tail after a separator, and reject pairs."""

    if type(raw) is not str:
        return None
    normalized = normalize_vehicle_label(raw)
    if normalized is None:
        return None
    if _MULTI_MODEL.search(normalized):
        # `a4 , s4` names two vehicles. Guessing which one owns the cohort would
        # attribute registrations to a car that never carried them.
        return None
    head = _MODEL_SEPARATORS.split(_without_diacritics(normalized))[0]
    stripped = _TRAILING_DOT.sub("", _MODEL_PUNCTUATION.sub(" ", head))
    tokens = [token for token in stripped.split() if token != "-"]
    make_tokens = make.split()
    if len(tokens) > len(make_tokens) and tokens[: len(make_tokens)] == make_tokens:
        tokens = tokens[len(make_tokens) :]
    # A bare number after the first token is an engine size, not part of the name:
    # `a4 2.0 tdi aut.` is an A4. The dot is kept inside a token so that VW's
    # ID.3 and ID.4 stay the different cars they are.
    for position, token in enumerate(tokens):
        if position and _ENGINE_SIZE.fullmatch(token):
            tokens = tokens[:position]
            break
    return " ".join(tokens) or None


def _vocabulary(
    label_volume: dict[tuple[str, str], float], make_volume: dict[str, float]
) -> dict[str, frozenset[str]]:
    """Learn each make's real model names from the labels publishers wrote alone.

    A one-token label cannot itself be a trim of something shorter, so it always
    counts. A two or three token label counts only when it carries a real share
    of the make, which keeps `golf sportsvan` and drops `polo s ac`.
    """

    rules = _rules()
    vocabulary: dict[str, set[str]] = {}
    for (make, label), volume in label_volume.items():
        tokens = label.split()
        if len(tokens) > rules.trim_vocabulary_max_tokens:
            continue
        floor = make_volume.get(make, 0.0) * rules.variant_volume_share
        if len(tokens) == 1 or volume >= floor:
            vocabulary.setdefault(make, set()).add(label)
    return {make: frozenset(labels) for make, labels in vocabulary.items()}


def _resolve(label: str, vocabulary: frozenset[str]) -> str:
    """Fold a label onto the longest real model name it starts with."""

    tokens = label.split()
    for size in range(min(len(tokens), _rules().trim_vocabulary_max_tokens), 0, -1):
        candidate = " ".join(tokens[:size])
        if candidate in vocabulary:
            return candidate
    return label

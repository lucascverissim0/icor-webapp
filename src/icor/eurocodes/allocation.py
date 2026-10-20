"""Attribute registrations to Eurocodes from reviewed decisions, without guessing.

Reviewers state facts about each code of a vehicle family: the years it was
launched and built, the bodies and drive side it covers, which earlier code it
replaced, and the catalogue rows that say so. Everything else — which codes a
registration year can belong to — is computed here, so two reviewers' facts can
be compared mechanically and the arithmetic cannot drift between them.

A registration year that more than one code can explain is never split: it
returns every code it may belong to. Launch years are such years (registration
year is not production year), and so is a body the source does not name when
two bodies of one model carry different codes. The demand is then shown as a
range — certain for a code only where one code alone explains it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

DECISIONS_FORMAT = "icor-eurocode-decisions"
DECISIONS_VERSION = 1

RHD_MARKETS = frozenset({"GB", "IE", "CY", "MT"})
BODIES = frozenset(
    {"hatchback", "saloon", "estate", "coupe", "convertible", "suv", "mpv", "van", "pickup"}
)
ALL_BODIES = "all"
ROLES = frozenset({"main", "minor"})
DRIVES = frozenset({"both", "LHD", "RHD"})
IDENTITY_STATUSES = frozenset({"mapped", "not_in_catalogue", "ambiguous_identity", "not_a_vehicle"})

_BODY_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "estate",
        re.compile(
            r"\b(?:variant|estate|break|sw|sportswagon|sports\s?wagon|touring|avant|"
            r"kombi|combi|grandtour|grand\s?tour|sports\s?tourer|shooting\s?brake|wagon|"
            r"allroad|cw)\b"
        ),
    ),
    ("convertible", re.compile(r"\b(?:cabrio|cabriolet|convertible|roadster|spider|spyder)\b")),
    ("coupe", re.compile(r"\b(?:coupe|coupé)\b")),
    ("van", re.compile(r"\b(?:van|kasten|bestel|fourgon|panel)\b")),
    ("pickup", re.compile(r"\b(?:pick-?up|double\s?cab|crew\s?cab)\b")),
    ("saloon", re.compile(r"\b(?:saloon|sedan|limousine|berline|notchback)\b")),
)


def drive_side(geography: str) -> str:
    """Right-hand drive markets register right-hand drive cars."""

    return "RHD" if geography.upper() in RHD_MARKETS else "LHD"


def body_hint(raw_model: str | None) -> str | None:
    """The body a raw publisher label names, if it names exactly one."""

    if not raw_model:
        return None
    # `s.w.` is `sw`; hyphens and underscores separate words.
    text = raw_model.casefold().replace(".", "").replace("-", " ").replace("_", " ")
    found = {body for body, pattern in _BODY_HINTS if pattern.search(text)}
    return next(iter(found)) if len(found) == 1 else None


@dataclass(frozen=True, slots=True)
class CodeFact:
    """What the catalogue (and reviewed knowledge) says about one code of a family.

    ``first_year`` is the earliest registration year the code's vehicle can
    appear in; ``first_year_latest`` the year from which it certainly has, when
    the catalogue's start years disagree. ``last_year`` is the last production
    year when known; otherwise the code runs until a code that ``replaces`` it
    is certainly in volume.
    """

    code: str
    label: str
    first_year: int
    first_year_latest: int
    last_year: int | None
    bodies: frozenset[str]
    role: str
    drive: str
    replaces: tuple[str, ...]
    rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Family:
    key: str
    label: str
    make: str
    codes: tuple[CodeFact, ...]


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    identity: str
    status: str
    family: str | None
    body: str | None
    why: str


@dataclass(frozen=True, slots=True)
class Allocation:
    """The codes a registration may belong to, and why.

    ``status`` is ``single`` (one code), ``shared`` (several, never split),
    ``outside`` (a reviewed vehicle in a year no code of its family covers),
    ``unassignable`` (reviewed: the identity cannot name one vehicle or is not
    in the catalogue) or ``unreviewed``.
    """

    codes: tuple[str, ...]
    status: str
    family: str | None = None


UNREVIEWED = Allocation((), "unreviewed")


def _covers(fact: CodeFact, body: str) -> bool:
    return body == ALL_BODIES or ALL_BODIES in fact.bodies or body in fact.bodies


def _until(fact: CodeFact, successors: Iterable[CodeFact]) -> int | None:
    """The last registration year a code can explain, for one of its bodies.

    Stock built in a code's last production year is registered the next year,
    so a stated last year extends one year. A successor's launch span is shared
    with the code it replaces; when the stated last year falls inside that span
    the run-out is already covered there, so the result does not depend on
    whether the catalogue happens to state the end.
    """

    takeover = min((successor.first_year_latest for successor in successors), default=None)
    if fact.last_year is None:
        return takeover
    if takeover is not None and fact.last_year == takeover:
        return takeover
    return fact.last_year + 1


def _drive_matches(fact: CodeFact, drive: str) -> bool:
    return fact.drive == "both" or fact.drive == drive


@dataclass(frozen=True, slots=True)
class _Window:
    fact: CodeFact
    until: Mapping[str, int | None]

    def active(self, year: int, body: str | None) -> bool:
        """Whether the code can explain a registration of ``body`` (any body: None)."""

        if year < self.fact.first_year:
            return False
        ends = [self.until[body]] if body in self.until else list(self.until.values())
        return any(end is None or year <= end for end in ends)


@dataclass(frozen=True)
class FamilyTimeline:
    """Per drive side, every code's active registration years, per body it covers."""

    family: Family
    _windows: Mapping[str, tuple[_Window, ...]] = field(repr=False)

    @classmethod
    def build(cls, family: Family) -> FamilyTimeline:
        windows: dict[str, tuple[_Window, ...]] = {}
        for drive in ("LHD", "RHD"):
            facts = [fact for fact in family.codes if _drive_matches(fact, drive)]
            computed = []
            for fact in facts:
                replacing = [other for other in facts if fact.code in other.replaces]
                until = {
                    body: _until(fact, [other for other in replacing if _covers(other, body)])
                    for body in fact.bodies
                }
                computed.append(_Window(fact, until))
            windows[drive] = tuple(computed)
        return cls(family, windows)

    def codes(self, year: int, drive: str, body: str | None = None) -> tuple[str, ...]:
        """Every code a registration of this family can belong to.

        Without a body, only the family's main codes apply: a niche body (a
        cabriolet with its own code) gets demand only from registrations that
        name it. With a body, codes stating that body win and codes covering
        every body come next. A body that has its own codes in this family but
        none in that year has no known code: lending it another body's code
        would be a false certainty. A body the family never distinguishes falls
        back to the main codes.
        """

        windows = self._windows[drive]
        if body is not None:
            explicit = [w.fact for w in windows if body in w.fact.bodies and w.active(year, body)]
            if explicit:
                return tuple(sorted(fact.code for fact in explicit))
            generic = [
                w.fact
                for w in windows
                if ALL_BODIES in w.fact.bodies and w.active(year, ALL_BODIES)
            ]
            if generic:
                return tuple(sorted(fact.code for fact in generic))
            if any(body in w.fact.bodies for w in windows):
                return ()
        return tuple(
            sorted(w.fact.code for w in windows if w.fact.role == "main" and w.active(year, None))
        )


class EurocodeDecisions:
    """Reviewed families and identity assignments, validated against the catalogue."""

    def __init__(
        self,
        families: Mapping[str, Family],
        identities: Mapping[str, IdentityDecision],
        metadata: Mapping[str, Any],
    ) -> None:
        self.families = dict(families)
        self.identities = dict(identities)
        self.metadata = dict(metadata)
        self._timelines = {key: FamilyTimeline.build(family) for key, family in families.items()}

    def allocate(
        self,
        identity: str,
        year: int,
        geography: str,
        raw_body: str | None = None,
    ) -> Allocation:
        decision = self.identities.get(identity)
        if decision is None:
            return UNREVIEWED
        if decision.status != "mapped" or decision.family is None:
            return Allocation((), "unassignable")
        body = decision.body or raw_body
        codes = self._timelines[decision.family].codes(year, drive_side(geography), body)
        if not codes:
            return Allocation((), "outside", decision.family)
        return Allocation(codes, "single" if len(codes) == 1 else "shared", decision.family)

    def timeline(self, family: str) -> FamilyTimeline:
        return self._timelines[family]

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        *,
        catalogue_codes: Mapping[str, Iterable[int]] | None = None,
    ) -> EurocodeDecisions:
        """Load and validate a decisions document.

        With ``catalogue_codes`` (code → its catalogue row numbers), every code
        must exist and every cited row must belong to that code, so a mistyped
        code or an invented citation is refused rather than displayed.
        """

        if (
            document.get("format") != DECISIONS_FORMAT
            or document.get("version") != DECISIONS_VERSION
        ):
            raise ValueError("not an ICOR Eurocode decisions document")
        rows_by_code = (
            {code: frozenset(rows) for code, rows in catalogue_codes.items()}
            if catalogue_codes is not None
            else None
        )
        families: dict[str, Family] = {}
        for key, item in document.get("families", {}).items():
            families[key] = _family(key, item, rows_by_code)
        identities: dict[str, IdentityDecision] = {}
        for identity, item in document.get("identities", {}).items():
            decision = _identity(identity, item)
            if decision.family is not None and decision.family not in families:
                raise ValueError(f"identity {identity} names an unknown family")
            identities[identity] = decision
        return cls(families, identities, document.get("metadata", {}))


def _family(
    key: str, item: Mapping[str, Any], rows_by_code: Mapping[str, frozenset[int]] | None
) -> Family:
    facts: list[CodeFact] = []
    for raw in item["codes"]:
        code = str(raw["code"])
        first = int(raw["first_year"])
        latest = int(raw.get("first_year_latest") or first)
        last = raw.get("last_year")
        bodies = frozenset(raw.get("bodies") or [ALL_BODIES])
        fact = CodeFact(
            code=code,
            label=str(raw.get("label") or item.get("label") or key),
            first_year=first,
            first_year_latest=latest,
            last_year=None if last is None else int(last),
            bodies=bodies,
            role=str(raw.get("role") or "main"),
            drive=str(raw.get("drive") or "both"),
            replaces=tuple(str(value) for value in raw.get("replaces") or ()),
            rows=tuple(int(value) for value in raw.get("rows") or ()),
        )
        _check_fact(key, fact, rows_by_code)
        facts.append(fact)
    codes = [fact.code for fact in facts]
    if len(set(codes)) != len(codes):
        raise ValueError(f"family {key} lists a code twice")
    for fact in facts:
        unknown = set(fact.replaces) - set(codes)
        if unknown:
            raise ValueError(f"family {key}: {fact.code} replaces a code outside the family")
    for fact in facts:
        replacing = [other for other in facts if fact.code in other.replaces]
        if not replacing or fact.last_year is not None:
            continue
        orphaned = sorted(
            body for body in fact.bodies if not any(_covers(other, body) for other in replacing)
        )
        if orphaned:
            raise ValueError(
                f"family {key}: {fact.code} is replaced but not for {orphaned}; "
                "state its last_year or a successor for those bodies"
            )
    return Family(key, str(item.get("label") or key), str(item.get("make") or ""), tuple(facts))


def _check_fact(
    key: str, fact: CodeFact, rows_by_code: Mapping[str, frozenset[int]] | None
) -> None:
    where = f"family {key}, code {fact.code}"
    if fact.first_year_latest < fact.first_year:
        raise ValueError(f"{where}: first_year_latest precedes first_year")
    if fact.last_year is not None and fact.last_year < fact.first_year:
        raise ValueError(f"{where}: last_year precedes first_year")
    if not fact.bodies <= BODIES | {ALL_BODIES}:
        raise ValueError(f"{where}: unknown body {sorted(fact.bodies - BODIES)}")
    if fact.role not in ROLES:
        raise ValueError(f"{where}: unknown role {fact.role}")
    if fact.drive not in DRIVES:
        raise ValueError(f"{where}: unknown drive {fact.drive}")
    if fact.code in fact.replaces:
        raise ValueError(f"{where}: a code cannot replace itself")
    if rows_by_code is not None:
        if fact.code not in rows_by_code:
            raise ValueError(f"{where}: not in the catalogue")
        if not fact.rows:
            raise ValueError(f"{where}: cites no catalogue row")
        foreign = set(fact.rows) - rows_by_code[fact.code]
        if foreign:
            raise ValueError(f"{where}: cites rows of another code {sorted(foreign)}")


def _identity(identity: str, item: Mapping[str, Any]) -> IdentityDecision:
    status = str(item.get("status") or "mapped")
    if status not in IDENTITY_STATUSES:
        raise ValueError(f"identity {identity}: unknown status {status}")
    family = item.get("family")
    if status == "mapped" and not family:
        raise ValueError(f"identity {identity}: mapped without a family")
    if status != "mapped" and family:
        raise ValueError(f"identity {identity}: only a mapped identity names a family")
    body = item.get("body")
    if body is not None and body not in BODIES:
        raise ValueError(f"identity {identity}: unknown body {body}")
    return IdentityDecision(identity, status, family, body, str(item.get("why") or ""))

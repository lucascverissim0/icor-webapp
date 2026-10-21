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
    is certainly in volume. ``uncoded_successor`` is the first year a successor
    the catalogue has no code for can appear, and ``uncoded_successor_latest``
    (default: that year) the year it certainly took over. As with a coded
    successor, the launch span is shared: a registration may be the code's last
    stock or the successor, so the code stays possible but is flagged as shared
    with a vehicle the catalogue lacks. After the span the two cannot be told
    apart and the code explains neither. ``uncoded_sibling`` is the year a
    vehicle without a code went on sale alongside under the same name (an
    electric version on another platform): the code stays, flagged from then on.
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
    uncoded_successor: int | None = None
    uncoded_successor_latest: int | None = None
    uncoded_sibling: int | None = None

    @property
    def uncoded_takeover(self) -> int | None:
        """The year an uncoded successor certainly took over (None: there is none)."""

        if self.uncoded_successor is None:
            return None
        return self.uncoded_successor_latest or self.uncoded_successor

    def shared_with_uncoded(self, year: int) -> bool:
        """Whether a registration of this year may be a vehicle without a code instead."""

        return any(start is not None and start <= year for start in self._uncoded_starts())

    def _uncoded_starts(self) -> tuple[int | None, int | None]:
        return (self.uncoded_successor, self.uncoded_sibling)


@dataclass(frozen=True, slots=True)
class Family:
    key: str
    label: str
    make: str
    codes: tuple[CodeFact, ...]


@dataclass(frozen=True, slots=True)
class Variant:
    """Raw labels of an identity that name another family, a body, or no vehicle.

    The app's identity layer folds some distinct models together (an electric
    derivative spelled under its base model's name). A variant claims the raw
    labels its pattern matches, so those registrations follow their own family
    instead of making the whole identity shared or ambiguous. A variant with a
    non-mapped status (and no family) leaves its labels unassigned. A variant
    limited to some geographies or registration years applies only to those
    registrations: some publishers label a whole model range with the bare model
    name, in every year or only in some.
    """

    pattern: re.Pattern[str]
    families: tuple[str, ...]
    body: str | None
    why: str
    status: str = "mapped"
    geographies: frozenset[str] = frozenset()
    years: tuple[int, int] | None = None

    @property
    def family(self) -> str | None:
        return self.families[0] if self.families else None

    def matches(
        self, raw_model: str, geography: str | None = None, year: int | None = None
    ) -> bool:
        if self.geographies and geography not in self.geographies:
            return False
        if self.years and (year is None or not self.years[0] <= year <= self.years[1]):
            return False
        return self.pattern.search(raw_model.casefold()) is not None


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    """How one identity's registrations map to families.

    Several families mean the label may be any of those models (a market that
    labels a whole model range with the bare name): their codes are pooled.
    """

    identity: str
    status: str
    families: tuple[str, ...]
    body: str | None
    why: str
    variants: tuple[Variant, ...] = ()

    @property
    def family(self) -> str | None:
        return self.families[0] if self.families else None


@dataclass(frozen=True, slots=True)
class Allocation:
    """The codes a registration may belong to, and why.

    ``status`` is ``single`` (one code), ``shared`` (several, never split),
    ``outside`` (a reviewed vehicle in a year no code of its family covers),
    ``unassignable`` (reviewed: the identity cannot name one vehicle or is not
    in the catalogue) or ``unreviewed``. ``uncoded`` marks a registration that
    may also be a vehicle the catalogue has no code for (a successor in its
    launch span, or a sibling sold under the same name): it is ``shared`` even
    with a single code.
    """

    codes: tuple[str, ...]
    status: str
    family: str | None = None
    uncoded: bool = False


UNREVIEWED = Allocation((), "unreviewed")


def _covers(fact: CodeFact, body: str) -> bool:
    return body == ALL_BODIES or ALL_BODIES in fact.bodies or body in fact.bodies


def _until(
    fact: CodeFact, successors: Iterable[CodeFact], replacing: Iterable[CodeFact] = ()
) -> int | None:
    """The last registration year a code can explain, for one of its bodies.

    ``successors`` replace the code for this body; ``replacing`` replace it for
    any body. Stock built in a code's last production year is registered the
    next year, so a stated last year extends one year. A successor's launch span
    is shared with the code it replaces; when the stated last year is the year a
    successor certainly took over, the run-out is already covered there, so the
    result does not depend on whether the catalogue happens to state the end —
    nor on whether this body got a successor of its own or was dropped when the
    rest of the line was replaced.
    """

    takeover = min((successor.first_year_latest for successor in successors), default=None)
    if fact.last_year is None:
        return takeover
    if takeover is None:
        takeover = min((other.first_year_latest for other in replacing), default=None)
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
                    body: _until(
                        fact, [other for other in replacing if _covers(other, body)], replacing
                    )
                    for body in fact.bodies
                }
                end = fact.uncoded_takeover
                if end is not None:
                    until = {
                        body: end if last is None else min(last, end)
                        for body, last in until.items()
                    }
                computed.append(_Window(fact, until))
            windows[drive] = tuple(computed)
        return cls(family, windows)

    def spans(self) -> dict[str, tuple[int, int | None]]:
        """Each code's registration years in this family: first and last (None: still current)."""

        spans: dict[str, tuple[int, int | None]] = {}
        for windows in self._windows.values():
            for window in windows:
                ends = list(window.until.values())
                end = None if None in ends else max(end for end in ends if end is not None)
                code = window.fact.code
                if code in spans:
                    first, last = spans[code]
                    end = None if last is None or end is None else max(last, end)
                    spans[code] = (min(first, window.fact.first_year), end)
                else:
                    spans[code] = (window.fact.first_year, end)
        return spans

    def shared_with_uncoded(self, year: int, drive: str, codes: Iterable[str]) -> bool:
        """Whether any of these codes may, in this year, be a vehicle without a code instead."""

        wanted = set(codes)
        return any(
            w.fact.code in wanted and w.fact.shared_with_uncoded(year) for w in self._windows[drive]
        )

    def codes(self, year: int, drive: str, body: str | None = None) -> tuple[str, ...]:
        """Every code a registration of this family can belong to.

        Without a body, only the family's main codes apply: a niche body (a
        cabriolet with its own code) gets demand only from registrations that
        name it. With a body, codes stating that body win and codes covering
        every body come next. A body that has its own codes in this family but
        none in that year has no known code: lending it another body's code
        would be a false certainty. A body the family never distinguishes falls
        back to the main codes.

        A code covering every body that one of the body's own codes replaces (or
        that replaces one of them) is the same body's other generation, not a
        parallel line, so it still applies while active.
        """

        windows = self._windows[drive]
        if body is not None:
            explicit = [w.fact for w in windows if body in w.fact.bodies and w.active(year, body)]
            if explicit:
                own = [fact for fact in self.family.codes if body in fact.bodies]
                own_codes = {fact.code for fact in own}
                replaced = {code for fact in own for code in fact.replaces}
                lineage = [
                    w.fact
                    for w in windows
                    if ALL_BODIES in w.fact.bodies
                    and w.active(year, ALL_BODIES)
                    and (w.fact.code in replaced or own_codes & set(w.fact.replaces))
                ]
                return tuple(sorted({fact.code for fact in (*explicit, *lineage)}))
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
        *,
        raw_model: str | None = None,
    ) -> Allocation:
        """The codes a registration may carry.

        ``raw_model`` is the publisher's model label: it selects a variant and,
        when ``raw_body`` is not given, supplies the body the label names.
        """

        decision = self.identities.get(identity)
        if decision is None:
            return UNREVIEWED
        families = decision.families if decision.status == "mapped" else ()
        body = decision.body
        if raw_model is not None:
            variant = next(
                (item for item in decision.variants if item.matches(raw_model, geography, year)),
                None,
            )
            if variant is not None:
                families, body = variant.families, variant.body
            if raw_body is None:
                raw_body = body_hint(raw_model)
        if not families:
            return Allocation((), "unassignable")
        drive = drive_side(geography)
        per_family = {
            family: self._timelines[family].codes(year, drive, body or raw_body)
            for family in families
        }
        codes = tuple(sorted({code for found in per_family.values() for code in found}))
        if not codes:
            return Allocation((), "outside", families[0])
        uncoded = any(
            self._timelines[family].shared_with_uncoded(year, drive, found)
            for family, found in per_family.items()
        )
        status = "single" if len(codes) == 1 and not uncoded else "shared"
        return Allocation(codes, status, families[0], uncoded)

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
            named = [*decision.families, *(f for v in decision.variants for f in v.families)]
            if any(family not in families for family in named):
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
        uncoded = raw.get("uncoded_successor")
        uncoded_latest = raw.get("uncoded_successor_latest")
        sibling = raw.get("uncoded_sibling")
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
            uncoded_successor=None if uncoded is None else int(uncoded),
            uncoded_successor_latest=None if uncoded_latest is None else int(uncoded_latest),
            uncoded_sibling=None if sibling is None else int(sibling),
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
    if fact.uncoded_successor is not None:
        if fact.uncoded_successor <= fact.first_year:
            raise ValueError(f"{where}: uncoded_successor must follow first_year")
        if fact.last_year is not None and fact.uncoded_successor > fact.last_year + 1:
            raise ValueError(f"{where}: uncoded_successor after the run-out year")
    if fact.uncoded_successor_latest is not None:
        if fact.uncoded_successor is None:
            raise ValueError(f"{where}: uncoded_successor_latest without uncoded_successor")
        if fact.uncoded_successor_latest < fact.uncoded_successor:
            raise ValueError(f"{where}: uncoded_successor_latest before uncoded_successor")
        if fact.last_year is not None and fact.uncoded_successor_latest > fact.last_year + 1:
            raise ValueError(f"{where}: uncoded_successor_latest after the run-out year")
    if fact.uncoded_sibling is not None:
        if fact.uncoded_sibling < fact.first_year:
            raise ValueError(f"{where}: uncoded_sibling before first_year")
        if fact.last_year is not None and fact.uncoded_sibling > fact.last_year + 1:
            raise ValueError(f"{where}: uncoded_sibling after the run-out year")
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


def _family_keys(identity: str, value: Any) -> tuple[str, ...]:
    """A family key, or a list of them for a label that may be any of several models."""

    if value is None or value == "":
        return ()
    keys = [value] if isinstance(value, str) else value
    if not isinstance(keys, list) or not all(isinstance(key, str) and key for key in keys):
        raise ValueError(f"identity {identity}: a family is a key or a list of keys")
    if len(set(keys)) != len(keys):
        raise ValueError(f"identity {identity}: a family is listed twice")
    return tuple(keys)


def _identity(identity: str, item: Mapping[str, Any]) -> IdentityDecision:
    status = str(item.get("status") or "mapped")
    if status not in IDENTITY_STATUSES:
        raise ValueError(f"identity {identity}: unknown status {status}")
    family = _family_keys(identity, item.get("family"))
    if status == "mapped" and not family:
        raise ValueError(f"identity {identity}: mapped without a family")
    if status != "mapped" and family:
        raise ValueError(f"identity {identity}: only a mapped identity names a family")
    body = item.get("body")
    if body is not None and body not in BODIES:
        raise ValueError(f"identity {identity}: unknown body {body}")
    variants: list[Variant] = []
    for raw in item.get("variants") or ():
        try:
            pattern = re.compile(str(raw["match"]))
        except (KeyError, re.error) as error:
            raise ValueError(f"identity {identity}: unreadable variant pattern") from error
        variant_status = str(raw.get("status") or "mapped")
        if variant_status not in IDENTITY_STATUSES:
            raise ValueError(f"identity {identity}: unknown variant status {variant_status}")
        variant_family = _family_keys(identity, raw.get("family"))
        if variant_status == "mapped" and not variant_family:
            raise ValueError(f"identity {identity}: a mapped variant names no family")
        if variant_status != "mapped" and variant_family:
            raise ValueError(f"identity {identity}: an unassigned variant names a family")
        variant_body = raw.get("body")
        if variant_body is not None and variant_body not in BODIES:
            raise ValueError(f"identity {identity}: unknown body {variant_body}")
        geographies = raw.get("geographies") or ()
        if isinstance(geographies, str) or not all(
            isinstance(item, str) and len(item) == 2 and item.isupper() for item in geographies
        ):
            raise ValueError(f"identity {identity}: variant geographies are country codes")
        years = raw.get("years")
        if years is not None and not (
            isinstance(years, list)
            and len(years) == 2
            and all(isinstance(item, int) for item in years)
            and years[0] <= years[1]
        ):
            raise ValueError(f"identity {identity}: variant years are a first and a last year")
        variants.append(
            Variant(
                pattern,
                variant_family,
                variant_body,
                str(raw.get("why") or ""),
                variant_status,
                frozenset(geographies),
                (years[0], years[1]) if years is not None else None,
            )
        )
    return IdentityDecision(
        identity, status, family, body, str(item.get("why") or ""), tuple(variants)
    )

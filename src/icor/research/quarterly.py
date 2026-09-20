"""Bounded quarterly discovery of new official vehicle-registration releases."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

MODEL = "o4-mini"
MAX_TOOL_CALLS_PER_BATCH = 24
MAX_OUTPUT_TOKENS_PER_BATCH = 12_000
MAX_CANDIDATES_PER_BATCH = 24
BUDGET_USD = Decimal("20.00")

# OpenAI prices checked on 2026-09-12. They are recorded in every report so a
# reviewer can reconcile the estimate against the client's API usage dashboard.
INPUT_USD_PER_MILLION = Decimal("1.10")
OUTPUT_USD_PER_MILLION = Decimal("4.40")
WEB_SEARCH_USD_PER_CALL = Decimal("0.01")
MODEL_CONTEXT_TOKENS = 200_000
PRICING_SOURCE = "https://developers.openai.com/api/docs/models/o4-mini"


@dataclass(frozen=True, slots=True)
class ResearchTarget:
    key: str
    batch: str
    publisher: str
    query: str
    official_domains: tuple[str, ...]


TARGETS = (
    ResearchTarget(
        key="eea-passenger-cars",
        batch="european-and-western",
        publisher="European Environment Agency",
        query=(
            "latest final or provisional EU passenger-car CO2 monitoring dataset "
            "with make and commercial-name registration records"
        ),
        official_domains=("europa.eu",),
    ),
    ResearchTarget(
        key="uk-dft-vehicle-model",
        batch="european-and-western",
        publisher="UK Department for Transport / DVLA",
        query=(
            "latest VEH0160 model-level first registrations and VEH0120 model-level "
            "licensed vehicle tables"
        ),
        official_domains=("gov.uk",),
    ),
    ResearchTarget(
        key="kba-fz10",
        batch="european-and-western",
        publisher="Kraftfahrt-Bundesamt",
        query="latest annual FZ 10 passenger-car registrations by brand and model series",
        official_domains=("kba.de",),
    ),
    ResearchTarget(
        key="france-sdes-vehicle-model",
        batch="european-and-western",
        publisher="French SDES",
        query="latest passenger-car fleet or registrations by make and model",
        official_domains=("statistiques.developpement-durable.gouv.fr", "data.gouv.fr"),
    ),
    ResearchTarget(
        "austria-statistik",
        "european-and-western",
        "Statistics Austria",
        "latest passenger-car registrations or stock by make and model",
        ("statistik.at",),
    ),
    ResearchTarget(
        "belgium-statbel",
        "european-and-western",
        "Statbel",
        "latest passenger-car registrations or stock by make and model",
        ("statbel.fgov.be",),
    ),
    ResearchTarget(
        "ireland-cso",
        "european-and-western",
        "Central Statistics Office Ireland",
        "latest licensed vehicles or new registrations by make and model",
        ("cso.ie",),
    ),
    ResearchTarget(
        "luxembourg-statec",
        "european-and-western",
        "STATEC Luxembourg",
        "latest passenger-car registrations or fleet by make and model",
        ("statistiques.public.lu",),
    ),
    ResearchTarget(
        "netherlands-rdw",
        "european-and-western",
        "Netherlands Vehicle Authority (RDW)",
        "latest open vehicle register or registrations with make and model",
        ("rdw.nl", "opendata.rdw.nl"),
    ),
    ResearchTarget(
        "switzerland-fso",
        "european-and-western",
        "Swiss Federal Statistical Office",
        "latest passenger-car registrations or stock by make and model",
        ("bfs.admin.ch",),
    ),
    ResearchTarget(
        "liechtenstein-as",
        "european-and-western",
        "Liechtenstein Office of Statistics",
        "latest passenger-car registrations or stock by make and model",
        ("llv.li",),
    ),
    ResearchTarget(
        "denmark-statbank",
        "nordic-and-baltic",
        "Statistics Denmark",
        "latest passenger-car registrations or fleet by make and model in StatBank",
        ("dst.dk", "statbank.dk"),
    ),
    ResearchTarget(
        "finland-traficom",
        "nordic-and-baltic",
        "Finnish Transport and Communications Agency Traficom",
        "latest open vehicle registrations or stock by make and model",
        ("traficom.fi",),
    ),
    ResearchTarget(
        "sweden-trafa",
        "nordic-and-baltic",
        "Transport Analysis Sweden",
        "latest vehicle registrations or stock by make and model",
        ("trafa.se",),
    ),
    ResearchTarget(
        "norway-ssb",
        "nordic-and-baltic",
        "Statistics Norway",
        "latest passenger-car registrations or stock by make and model",
        ("ssb.no",),
    ),
    ResearchTarget(
        "iceland-statice",
        "nordic-and-baltic",
        "Statistics Iceland",
        "latest passenger-car registrations or stock by make and model",
        ("statice.is",),
    ),
    ResearchTarget(
        "estonia-statistics",
        "nordic-and-baltic",
        "Statistics Estonia",
        "latest registered vehicles or first registrations by make and model",
        ("stat.ee",),
    ),
    ResearchTarget(
        "latvia-statistics",
        "nordic-and-baltic",
        "Official Statistics of Latvia",
        "latest vehicle stock or first registrations by make and model",
        ("stat.gov.lv",),
    ),
    ResearchTarget(
        "lithuania-statistics",
        "nordic-and-baltic",
        "State Data Agency Lithuania",
        "latest passenger-car registrations or stock by make and model",
        ("stat.gov.lt",),
    ),
    ResearchTarget(
        "cyprus-cystat",
        "southern",
        "Statistical Service of Cyprus",
        "latest monthly motor-vehicle registrations with make detail",
        ("cystat.gov.cy", "gov.cy"),
    ),
    ResearchTarget(
        "greece-elstat",
        "southern",
        "Hellenic Statistical Authority",
        "latest road motor vehicle registrations or stock by make and model",
        ("statistics.gr",),
    ),
    ResearchTarget(
        "italy-aci",
        "southern",
        "Automobile Club d Italia",
        "latest passenger-car registrations or circulating fleet by make and model",
        ("aci.it",),
    ),
    ResearchTarget(
        "malta-nso",
        "southern",
        "National Statistics Office Malta",
        "latest licensed motor vehicles or registrations by make and model",
        ("nso.gov.mt",),
    ),
    ResearchTarget(
        "portugal-imt",
        "southern",
        "Institute for Mobility and Transport Portugal",
        "latest passenger-car registrations or stock by make and model",
        ("imt-ip.pt",),
    ),
    ResearchTarget(
        "spain-dgt",
        "southern",
        "Directorate-General for Traffic Spain",
        "latest vehicle registrations or fleet microdata by make and model",
        ("dgt.es",),
    ),
    ResearchTarget(
        "bulgaria-nsi",
        "central-and-eastern",
        "National Statistical Institute Bulgaria",
        "latest passenger-car registrations or stock by make and model",
        ("nsi.bg",),
    ),
    ResearchTarget(
        "croatia-dzs",
        "central-and-eastern",
        "Croatian Bureau of Statistics",
        "latest registered road vehicles or first registrations by make and model",
        ("dzs.hr",),
    ),
    ResearchTarget(
        "czechia-transport",
        "central-and-eastern",
        "Ministry of Transport Czech Republic",
        "latest central vehicle register statistics by make and model",
        ("md.gov.cz",),
    ),
    ResearchTarget(
        "hungary-ksh",
        "central-and-eastern",
        "Hungarian Central Statistical Office",
        "latest first passenger-car registrations by make and model",
        ("ksh.hu",),
    ),
    ResearchTarget(
        "poland-cepik",
        "central-and-eastern",
        "Central Register of Vehicles and Drivers Poland",
        "latest CEPiK vehicle registration open data by make and model",
        ("gov.pl", "dane.gov.pl"),
    ),
    ResearchTarget(
        "romania-dgpci",
        "central-and-eastern",
        "Directorate General for Driving Licences and Registrations Romania",
        "latest vehicle registrations or fleet by make and model",
        ("mai.gov.ro", "gov.ro"),
    ),
    ResearchTarget(
        "slovakia-statistics",
        "central-and-eastern",
        "Statistical Office of the Slovak Republic",
        "latest passenger-car registrations or stock by make and model",
        ("statistics.sk",),
    ),
    ResearchTarget(
        "slovenia-surs",
        "central-and-eastern",
        "Statistical Office of the Republic of Slovenia",
        "latest passenger-car registrations or stock by make and model",
        ("stat.si",),
    ),
)

BATCHES = tuple(dict.fromkeys(target.batch for target in TARGETS))


class ResponsesClient(Protocol):
    class _Responses(Protocol):
        def create(self, **kwargs: Any) -> Any: ...

    responses: _Responses


class ResearchError(RuntimeError):
    """A safe, actionable quarterly-research failure."""


def conservative_run_ceiling_usd() -> Decimal:
    """Return a deliberately conservative ceiling for every regional batch."""
    per_batch = (
        Decimal(MODEL_CONTEXT_TOKENS) * INPUT_USD_PER_MILLION / Decimal(1_000_000)
        + Decimal(MAX_OUTPUT_TOKENS_PER_BATCH) * OUTPUT_USD_PER_MILLION / Decimal(1_000_000)
        + Decimal(MAX_TOOL_CALLS_PER_BATCH) * WEB_SEARCH_USD_PER_CALL
    ).quantize(Decimal("0.0001"))
    return (per_batch * len(BATCHES)).quantize(Decimal("0.0001"))


def build_prompt(now: datetime, targets: tuple[ResearchTarget, ...]) -> str:
    target_lines = "\n".join(
        f"- {target.key}: {target.query}; publisher={target.publisher}; "
        f"allowed domains={','.join(target.official_domains)}"
        for target in targets
    )
    return f"""Research date: {now.date().isoformat()}.
Find newly published or revised official vehicle registration/fleet releases for the
targets below. Use web search for discovery only. Return a candidate only when its URL
is on the target's allowed official domain and the page identifies the publisher,
period, measure, geography, and whether make/model detail exists. Do not copy numbers,
infer model years, convert sales years to registration years, or treat a search result
as accepted evidence. Omit uncertain candidates. Prefer releases newer than 2025-01-01.

{target_lines}
"""


_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates"],
    "properties": {
        "candidates": {
            "type": "array",
            "maxItems": MAX_CANDIDATES_PER_BATCH,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "target_key",
                    "publisher",
                    "source_url",
                    "release_title",
                    "period_start",
                    "period_end",
                    "measure",
                    "geography",
                    "model_detail",
                    "publication_status",
                    "evidence_note",
                ],
                "properties": {
                    "target_key": {"type": "string"},
                    "publisher": {"type": "string"},
                    "source_url": {"type": "string"},
                    "release_title": {"type": "string"},
                    "period_start": {"type": "string"},
                    "period_end": {"type": "string"},
                    "measure": {"type": "string"},
                    "geography": {"type": "string"},
                    "model_detail": {"type": "boolean"},
                    "publication_status": {
                        "type": "string",
                        "enum": ["final", "provisional", "unknown"],
                    },
                    "evidence_note": {"type": "string"},
                },
            },
        }
    },
}


def _run_batch(
    client: ResponsesClient,
    model: str,
    now: datetime,
    targets: tuple[ResearchTarget, ...],
) -> tuple[Any, list[dict[str, Any]], int]:
    response = client.responses.create(
        model=model,
        instructions=(
            "You are an evidence-discovery assistant. Search every supplied target. "
            "Follow domain allowlists exactly. Return strict JSON only. Discovery "
            "never approves evidence."
        ),
        input=build_prompt(now, targets),
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
        max_tool_calls=MAX_TOOL_CALLS_PER_BATCH,
        max_output_tokens=MAX_OUTPUT_TOKENS_PER_BATCH,
        text={
            "format": {
                "type": "json_schema",
                "name": "quarterly_source_candidates",
                "strict": True,
                "schema": _OUTPUT_SCHEMA,
            }
        },
        store=False,
        timeout=600,
    )
    try:
        payload = json.loads(response.output_text)
    except (AttributeError, TypeError, json.JSONDecodeError) as error:
        raise ResearchError("OpenAI returned an invalid structured research response") from error
    calls = sum(
        1
        for item in getattr(response, "output", ())
        if getattr(item, "type", None) == "web_search_call"
    )
    if calls > MAX_TOOL_CALLS_PER_BATCH:
        raise ResearchError("response exceeded the configured web-search limit")
    return response, _validate_candidates(payload, allowed_targets=targets), calls


def run_quarterly_research(
    client: ResponsesClient,
    output_dir: Path,
    *,
    now: datetime | None = None,
    model: str = MODEL,
) -> Path:
    now = (now or datetime.now(UTC)).astimezone(UTC)
    if conservative_run_ceiling_usd() >= BUDGET_USD:
        raise ResearchError("configured run ceiling is not below the run budget")

    response_ids: list[str] = []
    candidates: list[dict[str, Any]] = []
    input_tokens = output_tokens = web_search_calls = 0
    for batch in BATCHES:
        targets = tuple(target for target in TARGETS if target.batch == batch)
        response, batch_candidates, batch_calls = _run_batch(client, model, now, targets)
        response_ids.append(str(getattr(response, "id", "unknown")))
        candidates.extend(batch_candidates)
        web_search_calls += batch_calls
        input_tokens += _required_usage_value(response, "input_tokens")
        output_tokens += _required_usage_value(response, "output_tokens")
        if _estimated_cost(input_tokens, output_tokens, web_search_calls) >= BUDGET_USD:
            raise ResearchError("metered run cost reached the configured budget")

    identities = {(item["target_key"], item["source_url"]) for item in candidates}
    if len(identities) != len(candidates):
        raise ResearchError("research responses contain a duplicate candidate URL")
    estimated_cost = _estimated_cost(input_tokens, output_tokens, web_search_calls)
    if estimated_cost >= BUDGET_USD:
        raise ResearchError("metered run cost reached the configured budget")

    report = {
        "schema_version": 1,
        "run_id": f"quarterly-source-research-{now.strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "model": model,
        "response_ids": response_ids,
        "review_status": "pending-human-review",
        "automatic_promotion": False,
        "budget": {
            "limit_usd": str(BUDGET_USD),
            "conservative_run_ceiling_usd": str(conservative_run_ceiling_usd()),
            "estimated_metered_cost_usd": str(estimated_cost),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "web_search_calls": web_search_calls,
            "pricing_checked_on": "2026-09-12",
            "pricing_source": PRICING_SOURCE,
        },
        "targets": [asdict(target) for target in TARGETS],
        "candidates": candidates,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{report['run_id']}.json"
    if destination.exists():
        raise ResearchError("quarterly report already exists")
    _atomic_write(destination, report)
    return destination


def require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ResearchError("OPENAI_API_KEY is required for a paid research run")
    return key


def _validate_candidates(
    payload: Any,
    *,
    allowed_targets: tuple[ResearchTarget, ...] = TARGETS,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or set(payload) != {"candidates"}:
        raise ResearchError("research response has an unexpected top-level shape")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) > MAX_CANDIDATES_PER_BATCH:
        raise ResearchError("research response has an invalid candidate list")
    target_by_key = {target.key: target for target in allowed_targets}
    required_fields = set(_OUTPUT_SCHEMA["properties"]["candidates"]["items"]["required"])
    seen: set[tuple[str, str]] = set()
    validated: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ResearchError("research candidate is not an object")
        if set(candidate) != required_fields:
            raise ResearchError("research candidate has an unexpected shape")
        target = target_by_key.get(candidate.get("target_key"))
        if target is None:
            raise ResearchError("research candidate has an unknown target")
        url = candidate.get("source_url")
        if not isinstance(url, str) or not _official_url(url, target.official_domains):
            raise ResearchError("research candidate URL is outside its official allowlist")
        identity = (target.key, url)
        if identity in seen:
            raise ResearchError("research response contains a duplicate candidate URL")
        seen.add(identity)
        if candidate.get("publisher") != target.publisher:
            raise ResearchError("research candidate publisher does not match its target")
        if not all(
            isinstance(candidate.get(field), str) and candidate[field].strip()
            for field in (
                "release_title",
                "period_start",
                "period_end",
                "measure",
                "geography",
                "evidence_note",
            )
        ):
            raise ResearchError("research candidate contains an empty required field")
        if type(candidate.get("model_detail")) is not bool:
            raise ResearchError("research candidate model_detail must be boolean")
        if candidate.get("publication_status") not in {"final", "provisional", "unknown"}:
            raise ResearchError("research candidate has an invalid publication status")
        validated.append(candidate)
    return validated


def _official_url(url: str, domains: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False
    hostname = parsed.hostname.lower().rstrip(".")
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains)


def _required_usage_value(response: Any, name: str) -> int:
    usage = getattr(response, "usage", None)
    value = getattr(usage, name, None) if usage is not None else None
    if type(value) is not int or value < 0:
        raise ResearchError("OpenAI response did not include valid token usage")
    return value


def _estimated_cost(input_tokens: int, output_tokens: int, web_calls: int) -> Decimal:
    return (
        Decimal(input_tokens) * INPUT_USD_PER_MILLION / Decimal(1_000_000)
        + Decimal(output_tokens) * OUTPUT_USD_PER_MILLION / Decimal(1_000_000)
        + Decimal(web_calls) * WEB_SEARCH_USD_PER_CALL
    ).quantize(Decimal("0.0001"))


def _atomic_write(destination: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

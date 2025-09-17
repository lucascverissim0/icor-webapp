#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation-specific sales estimator (EU + World)

This revision makes the generation window come from the new wikipedia_gen
module FIRST (your most accurate detector). If that fails, it falls back to the
existing Wikipedia-API parse, then SERP, then local/default.

Everything else (seeding, constraints, smoothing, outputs) is unchanged.
"""

import os, re, csv, json, glob, time, math, sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import difflib
import requests
import pandas as pd
from tabulate import tabulate
from collections import Counter

# --- path safety so we can "import scripts.wikipedia_gen" even when run directly
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

# NEW: high-accuracy Wikipedia module (first-choice detector)
try:
    from scripts.wikipedia_gen import detect_via_wikipedia
except Exception:
    from wikipedia_gen import detect_via_wikipedia  # type: ignore

# NEW: for Wikipedia HTML parsing (kept for existing fallback code)
from bs4 import BeautifulSoup

# ------------------ Secrets (env) ------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")
if not OPENAI_API_KEY: print("⚠️  OPENAI_API_KEY not set", file=sys.stderr)
if not SERPAPI_KEY:    print("⚠️  SERPAPI_KEY not set (web seeding disabled)", file=sys.stderr)

# ------------------ Config -------------------------
MODEL_NAME          = "gpt-5"
DEADLINE_YEAR       = 2035
APPLY_SMOOTHING     = True
SEARCH_TIMEOUT      = 20
MAX_RESULTS_TO_SCAN = 10
WORLD_MAX_CAP       = 3_000_000
DECAY_RATE          = 0.0556
REPAIR_RATE         = 0.021
MAX_FUTURE_GAP      = 1

# Keep for internal fallback (Wikipedia API parsing)
GEN_WINDOW_SOURCE   = "web_first"  # options: "web_first" | "local_first"

SYSTEM_INSTRUCTIONS = (
    "You are an automotive market analyst. Use provided seed data and constraints as anchors. "
    "This forecast is generation-specific; do not mix other generations. "
    "If world is provided as a range, ensure your start-year world is inside it. "
    "For years outside the generation window, output 0 for both Europe and World. "
    "Explain key assumptions briefly."
)

# ------------------ Normalizers --------------------
def normalize_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[/\-]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_generation(g: Optional[str]) -> str:
    if not g: return ""
    g = str(g).lower()
    g = re.sub(r"[\(\)\[\]{}]", " ", g)
    g = re.sub(r"\bmk\s*", "mk", g)
    g = re.sub(r"\s+", " ", g).strip()
    return g

# ------------------ Local DB loaders ---------------
def _load_json_array(path: str) -> Optional[List[dict]]:
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return None

def load_local_database_eu(folder: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    db: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(folder, "Top100_*.txt"))):
        if "_World_" in os.path.basename(path): continue
        m = re.search(r"(\d{4})", os.path.basename(path)); 
        if not m: continue
        year = int(m.group(1))
        arr = _load_json_array(path)
        if not arr: continue
        for rec in arr:
            model = rec.get("model") or rec.get("name") or ""
            if not model: continue
            gen   = rec.get("generation")
            units = rec.get("units_sold", rec.get("projected_units_2025"))
            if not isinstance(units, int): continue
            key = normalize_name(model)
            db.setdefault(key, {})
            db[key][year] = {
                "model": model, "generation": gen,
                "units_europe": int(units),
                "estimated": bool(rec.get("estimated", False)),
            }
    return db

def load_local_database_world(folder: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    db: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(folder, "Top100_World_*.txt"))):
        m = re.search(r"(\d{4})", os.path.basename(path))
        if not m: continue
        year = int(m.group(1))
        arr = _load_json_array(path)
        if not arr: continue
        for rec in arr:
            model = rec.get("model") or rec.get("name") or ""
            if not model: continue
            gen   = rec.get("generation")
            units = rec.get("units_sold", rec.get("projected_units_2025"))
            if not isinstance(units, int): continue
            key = normalize_name(model)
            db.setdefault(key, {})
            db[key][year] = {
                "model": model, "generation": gen,
                "units_world": int(units),
                "estimated": bool(rec.get("estimated", False)),
            }
    return db

# ------------------ Safer model matching -----------
def find_best_model_key(*dbs: Dict[str, Dict[int, Dict[str, Any]]], user_model: str) -> Optional[str]:
    key = normalize_name(user_model)
    for db in dbs:
        if key in db:
            return key
    all_keys = sorted(set(k for db in dbs for k in db.keys()))
    if not all_keys:
        return None
    def digit_tokens(s: str) -> set[str]:
        toks = s.split()
        return {t for t in toks if t.isdigit() or re.fullmatch(r"[a-z]+?\d+", t)}
    target_digits = digit_tokens(key)
    filtered = all_keys
    if target_digits:
        same = [k for k in all_keys if digit_tokens(k) == target_digits]
        if same:
            filtered = same
    best, best_r = None, -1.0
    for cand in filtered:
        r = difflib.SequenceMatcher(None, key, cand).ratio()
        if r > best_r:
            best, best_r = cand, r
    if best and best_r >= 0.88:
        return best
    return None

def window_from_local_history(hist: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    if not hist: return None
    years = [h["year"] for h in hist]
    return (min(years), max(years))

# ------------------ Serp helpers -------------------
SERP_ENDPOINT = "https://serpapi.com/search.json"
SALES_WORDS = ["sales","sold","units","registrations","deliveries","volume","shipments"]
CURRENCY_TOKENS = ["€","$","£"]
def looks_like_price(text: str) -> bool:
    t = text.lower()
    return any(sym in text for sym in CURRENCY_TOKENS) or "msrp" in t or "price" in t
def has_sales_context(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in SALES_WORDS)
def extract_candidate_numbers(text: str) -> List[int]:
    NUM_PATTERN = re.compile(r"(?<![A-Z0-9])(\d{1,3}(?:[,\.\s]\d{3})+|\d{4,})(?![A-Z0-9])")
    def _to_int(s: str):
        s = re.sub(r"[,\.\s]", "", s)
        return int(s) if s.isdigit() else None
    out = []
    for m in NUM_PATTERN.finditer(text or ""):
        v = _to_int(m.group(1))
        if v is not None: out.append(v)
    return [n for n in out if 10 <= n <= 5_000_000]
def serp_search(query: str) -> Dict[str, Any]:
    params = {"engine":"google","q":query,"api_key":SERPAPI_KEY,"hl":"en","num":"10","safe":"active"}
    try:
        r = requests.get(SERP_ENDPOINT, params=params, timeout=SEARCH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "query": query}
def pull_text_blobs(serp_json: Dict[str, Any]) -> List[Dict[str, str]]:
    blobs = []
    def add(source, text, url):
        if not text: return
        if looks_like_price(text): return
        blobs.append({"source": source, "text": text, "url": url})
    for key in ("answer_box","knowledge_graph"):
        box = serp_json.get(key)
        if isinstance(box, dict):
            for _, v in box.items():
                if isinstance(v, str): add(key, v, serp_json.get("search_metadata",{}).get("google_url",""))
    for item in serp_json.get("organic_results", [])[:MAX_RESULTS_TO_SCAN]:
        add("organic", f"{item.get('title','')}\n{item.get('snippet','')}".strip(), item.get("link",""))
    return blobs
def build_generation_aliases(gen: str) -> List[str]:
    g = (gen or "").strip()
    if not g: return []
    gnorm = normalize_generation(g)
    out = {g}
    if gnorm.startswith("mk"):
        n = gnorm[2:]
        out.update({f"mk{n}", f"mk {n}", f"mark {n}", f"{n}th generation"})
    return sorted(a for a in out if a)
def best_number_for_region(blobs: List[Dict[str,str]], region: str, model: str, gen: str, year: int) -> Optional[Dict[str,Any]]:
    region_aliases = {
        "europe": ["europe","eu","efta","european","eu27","eu28","eu+efta","eu/efta","eu+uk"],
        "world":  ["world","global","worldwide"],
    }
    aliases = region_aliases["europe"] if region=="europe" else region_aliases["world"]
    gen_aliases = [a.lower() for a in build_generation_aliases(gen)]
    best, best_score = None, -1e9
    for b in blobs:
        text = (b.get("text") or "") + " " + (b.get("url") or "")
        t = text.lower()
        if not has_sales_context(t): continue
        if region=="world" and not any(a in t for a in aliases): continue
        if gen and not any(a in t for a in gen_aliases): continue
        nums = extract_candidate_numbers(text)
        if not nums: continue
        is_model = normalize_name(model) in normalize_name(text)
        is_country_only = bool(re.search(r"\b(uk|germany|france|italy|spain|poland|netherlands|sweden|norway)\b", t))
        is_monthly = bool(re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|month|monthly)\b", t))
        has_year = str(year) in t
        region_bonus = 3 if any(a in t for a in aliases) else 0
        model_bonus  = 4 if is_model else -2
        gen_bonus    = 2 if (not gen or any(a in t for a in gen_aliases)) else -3
        year_bonus   = 2 if has_year else -2
        country_pen  = -5 if (region=="europe" and is_country_only) else 0
        monthly_pen  = -5 if is_monthly else 0
        candidate    = max(nums)
        size_pen     = -6 if (region=="europe" and candidate < 20000 and not is_country_only) else 0
        score = (region_bonus+model_bonus+gen_bonus+year_bonus+country_pen+monthly_pen+size_pen) * 10 + math.log10(candidate+1)
        if score > best_score:
            best_score = score
            best = {"value": candidate, "url": b.get("url",""), "snippet": b.get("text",""), "is_model_level": is_model and not is_country_only and not is_monthly}
    return best
def build_search_queries(model: str, gen: str, year: int) -> List[str]:
    gen_alias = " OR ".join(f'"{a}"' for a in build_generation_aliases(gen)) if gen else ""
    base = [
        f'"{model}" {year} sales europe -price -msrp -€ -$',
        f'"{model}" {year} registrations europe -price -msrp -€ -$',
        f'"{model}" {year} global sales -price -msrp -€ -$',
        f'"{model}" {year} worldwide sales -price -msrp -€ -$',
    ]
    if gen_alias:
        base = [q.replace(f'"{model}"', f'"{model}" ({gen_alias})') for q in base]
    preferred = [
        f'site:carsalesbase.com "{model}" {year} sales',
        f'site:acea.auto "{model}" {year} registrations',
        f'site:marklines.com "{model}" {year} sales',
    ]
    return base + preferred
def serp_seed(model: str, gen: str, year: int) -> Dict[str, Any]:
    blobs_all: List[Dict[str,str]] = []
    for q in build_search_queries(model, gen, year):
        res = serp_search(q)
        if "error" not in res: blobs_all.extend(pull_text_blobs(res))
        time.sleep(0.35)
    eu = best_number_for_region(blobs_all, "europe", model, gen, year)
    w  = best_number_for_region(blobs_all, "world",  model, gen, year)
    return {"model": model, "gen": gen, "year": year, "europe": eu, "world": w}

# ------------------ Wikipedia API helpers (fallback) ---------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
YEAR_RANGE_RE = re.compile(
    r"(?P<start>\b\d{4}\b)\s*[–—-]\s*(?P<end>\b\d{4}\b|present|current|ongoing|to\s+present)", re.IGNORECASE
)
PARENS_RANGE_RE = re.compile(
    r"\((?:[^()]*?;?\s*)?(?P<start>\d{4})\s*[–—-]\s*(?P<end>\d{4}|present|current|ongoing)\)", re.IGNORECASE
)
GEN_TOKENS_RE = re.compile(r'\b(mk\s*\d+|mark\s*\d+|gen(?:eration)?\s*\w+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b', re.I)
YEAR_TOKEN_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')
def sanitize_model_for_wiki(q: str) -> str:
    s = q.strip(); s = GEN_TOKENS_RE.sub('', s); s = YEAR_TOKEN_RE.sub('', s)
    s = re.sub(r'\s+', ' ', s).strip(' -–—'); return s
def looks_like_generation_page(title: str) -> bool:
    t = title.lower()
    if re.search(r'\bmk\s*\d+\b', t): return True
    if 'generation' in t: return True
    if re.search(r'\([^)0-9]*\d{4}', t): return True
    return False
def wiki_search_page(query: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        base_q = sanitize_model_for_wiki(query)
        params = {"action":"query","list":"search","format":"json","srlimit":5,"srsearch": f'intitle:"{base_q}" {base_q}',"srnamespace":0}
        r = requests.get(WIKI_API, params=params, timeout=SEARCH_TIMEOUT)
        r.raise_for_status()
        results = r.json().get("query", {}).get("search", [])
        if not results: return None, None
        for res in results:
            title = res.get("title","")
            if title and not looks_like_generation_page(title):
                return res["pageid"], title
        top = results[0]
        return top["pageid"], top["title"]
    except Exception:
        return None, None
def wiki_fetch_page_html(pageid: int) -> str:
    try:
        params = {"action":"parse","pageid":pageid,"prop":"text","format":"json","disableeditsection":1}
        r = requests.get(WIKI_API, params=params, timeout=SEARCH_TIMEOUT)
        r.raise_for_status()
        return r.json().get("parse", {}).get("text", {}).get("*", "") or ""
    except Exception:
        return ""
def _normalize_dash(s: str) -> str: return s.replace("—", "–").replace("-", "–")
def _clean_text(s: str) -> str: return " ".join((s or "").split())
def _extract_generation_headings(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    gens = []
    for tag in soup.find_all(["h2", "h3", "h4"]):
        heading = tag.get_text(" ", strip=True)
        lower = heading.lower()
        if ("generation" in lower) or re.search(r"\bmk\d+\b", lower):
            gens.append((tag.name, heading))
    return gens
def _extract_years_from_heading(heading: str) -> Tuple[Optional[str], Optional[str]]:
    m = PARENS_RANGE_RE.search(heading) or YEAR_RANGE_RE.search(heading)
    if not m: return None, None
    start = m.group("start")
    end = m.group("end").lower()
    if end.startswith("to"): end = "present"
    return start, end
def parse_generations_from_wikipedia_html(html_content: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_content, "html.parser")
    gens = _extract_generation_headings(soup)
    if not gens:
        for tag in soup.find_all(["h2", "h3"]):
            txt = tag.get_text(" ", strip=True)
            if YEAR_RANGE_RE.search(txt):
                gens.append((tag.name, txt))
    out = []; seen = set()
    for _, heading in gens:
        heading_norm = _normalize_dash(heading)
        start, end = _extract_years_from_heading(heading_norm)
        if start or end:
            key = (start or "", end or "", heading_norm)
            if key in seen: 
                continue
            seen.add(key)
            out.append({
                "generation_heading": _clean_text(heading_norm),
                "launch_year": start,
                "end_year": (None if not end else ("present" if end.lower() in {"present", "current", "ongoing"} else end)),
            })
    def sort_key(item):
        try: return int(item["launch_year"]) if item["launch_year"] else 99999
        except Exception: return 99999
    out.sort(key=sort_key)
    return out
def detect_generation_via_wikipedia_api(model: str, view_year: int) -> Tuple[Optional[str], Optional[Tuple[int,int]], Optional[Dict[str,Any]]]:
    pageid, title = wiki_search_page(model)
    if not pageid: 
        return None, None, {"basis": "wikipedia_api", "status": "no_page"}
    html = wiki_fetch_page_html(pageid)
    if not html:
        return None, None, {"basis": "wikipedia_api", "status": "no_html", "pageid": pageid, "title": title}
    gens_primary = parse_generations_from_wikipedia_html(html)
    gens_extra = []
    diag_merge = {"merged": False, "extra_title": None}
    if looks_like_generation_page(title):
        base_q = sanitize_model_for_wiki(model)
        pid2, title2 = wiki_search_page(base_q)
        if pid2 and title2 and title2 != title and not looks_like_generation_page(title2):
            html2 = wiki_fetch_page_html(pid2)
            if html2:
                gens_extra = parse_generations_from_wikipedia_html(html2)
                diag_merge = {"merged": True, "extra_title": title2}
    all_gens = {(g.get("launch_year"), g.get("end_year"), g.get("generation_heading")) for g in (gens_primary + gens_extra)}
    gens = [{"launch_year": a, "end_year": b, "generation_heading": c} for (a,b,c) in all_gens if a or b]
    if not gens:
        return None, None, {"basis": "wikipedia_api", "status": "no_generations", "pageid": pageid, "title": title, **diag_merge}
    windows = []
    for g in gens:
        try:
            start = int(g["launch_year"]) if g["launch_year"] else None
        except Exception:
            start = None
        end_raw = g.get("end_year")
        end = DEADLINE_YEAR
        if isinstance(end_raw, str) and end_raw and end_raw.lower() not in {"present","current","ongoing"}:
            try: end = int(end_raw)
            except Exception: end = DEADLINE_YEAR
        if start:
            windows.append((start, end, g["generation_heading"]))
    if not windows: 
        return None, None, {"basis": "wikipedia_api", "status": "no_windows_parsed", "pageid": pageid, "title": title, **diag_merge}
    covering = [(s,e,h) for (s,e,h) in windows if (s-1) <= view_year <= (e+1)]
    if covering:
        covering.sort(key=lambda t: ((t[1]-t[0]), -t[0]))
        pick = covering[0]
    else:
        def dist(s,e):
            if view_year < s: return s - view_year
            if view_year > e: return view_year - e
            return 0
        windows.sort(key=lambda t: (dist(t[0], t[1]), -t[0]))
        pick = windows[0]
    start, end, heading = pick
    m = re.search(r'\b(?:mk|mark|gen(?:eration)?)\s*([ivx\d]{1,3})\b', heading, flags=re.I)
    gen_label = f"Mk{m.group(1).upper()}" if m else "GEN"
    diag = {"basis": "wikipedia_api_fixed", "pageid": pageid, "title": title, "heading": heading, **diag_merge, "note": "Window from Wikipedia headings."}
    return gen_label, (start, min(end, DEADLINE_YEAR)), diag

# ------------------ Continuation decisions ------------
def decide_continuation_if_present(model: str, window: Tuple[int,int], ref_year: int = 2025) -> Tuple[int, Dict[str, Any]]:
    start, end = window
    if end != DEADLINE_YEAR:
        return end, {"basis": "fixed_range", "signals": []}
    terms_stop = [
        "production ended", "end of production", "discontinued", "final model year",
        "final edition", "successor", "all-new generation", "redesign", "replaced by"
    ]
    terms_continue = [
        "facelift", "minor change", "carryover", "2026 model year", "MY2026", "MY2027", "continues for"
    ]
    q = f'"{model}" {" OR ".join([f"{t}" for t in (terms_stop+terms_continue)])}'
    res = serp_search(q)
    blobs = pull_text_blobs(res) if "error" not in res else []
    stop_hit_years = []
    cont_signal = False
    YEAR = re.compile(r'20\d{2}')
    for b in blobs:
        t = (b.get("text") or "").lower()
        if any(kw in t for kw in terms_stop):
            ys = [int(m.group(0)) for m in YEAR.finditer(t)]
            ymax = max([y for y in ys if 2015 <= y <= 2100], default=None)
            if ymax: stop_hit_years.append(ymax)
        if any(kw in t for kw in terms_continue):
            cont_signal = True
    diag = {"basis": "present_resolution", "signals": {"stop_years": stop_hit_years, "continuation": cont_signal}}
    if stop_hit_years:
        return max(min(ref_year, max(stop_hit_years)), start), diag
    assumed = max(ref_year + 2, start + 4)
    return min(assumed, DEADLINE_YEAR), diag

def decide_continuation_if_exact_2025(model: str, window: Tuple[int,int], ref_year: int = 2025) -> Tuple[int, Dict[str, Any]]:
    start, end = window
    if end != ref_year:
        return end, {"basis":"not_2025_end","signals":[]}
    terms_stop = [
        "production ended in 2025", "discontinued in 2025", "final model year 2025",
        "end of production 2025"
    ]
    terms_continue = [
        "2026 model year", "MY2026", "MY 2026", "continues into 2026", "facelift 2026", "carryover 2026"
    ]
    q = f'"{model}" {" OR ".join([f"{t}" for t in (terms_stop+terms_continue)])}'
    res = serp_search(q)
    blobs = pull_text_blobs(res) if "error" not in res else []
    stop_explicit = False
    cont_signal = False
    for b in blobs:
        t = (b.get("text") or "").lower()
        if any(kw in t for kw in terms_stop):
            stop_explicit = True
        if any(kw in t for kw in terms_continue):
            cont_signal = True
    diag = {"basis":"end_2025_resolution", "signals":{"explicit_stop_2025": stop_explicit, "continuation": cont_signal}}
    if stop_explicit:
        return end, diag
    if cont_signal:
        return min(ref_year + 2, DEADLINE_YEAR), diag
    return min(ref_year + 1, DEADLINE_YEAR), diag

# ------------------ Local seeds/history -------------
def infer_local_gen_alias(db_eu, db_world, user_model: str, detected_gen: str, gen_window: Tuple[int,int]) -> Optional[str]:
    mk = find_best_model_key(db_eu, db_world, user_model=user_model)
    if not mk: return None
    lo, hi = gen_window
    det_norm = normalize_generation(detected_gen)
    labels = []
    for db in (db_eu, db_world):
        for y, rec in db.get(mk, {}).items():
            if lo <= y <= hi:
                g = rec.get("generation")
                if g: labels.append(normalize_generation(g))
    labels = [g for g in labels if g]
    if not labels: return None
    c = Counter(labels)
    if det_norm in c: return None
    top, cnt = c.most_common(1)[0]
    if len(c)==1 or cnt/sum(c.values())>=0.7: return top
    return None

def local_seed_for_generation(db_eu, db_world, user_model, target_gen, start_year, accepted_alias=None, window=None) -> Dict[str,Any]:
    out = {"found_model": False, "model_key": None, "display_model": None,
           "eu": None, "world": None, "history_europe": [], "history_world": [],
           "rank_eu": None, "rank_world": None, "presence_years_eu": 0, "presence_years_world": 0}
    mk = find_best_model_key(db_eu, db_world, user_model=user_model)
    if not mk: return out
    out["found_model"] = True; out["model_key"] = mk
    accepted = {normalize_generation(target_gen)}
    if accepted_alias: accepted.add(normalize_generation(accepted_alias))
    def _hist(db, units_field):
        hist = []
        for y, rec in db.get(mk, {}).items():
            if window and not (window[0] <= y <= window[1]): continue
            if normalize_generation(rec.get("generation")) in accepted:
                hist.append({"year": y, "units": int(rec.get(units_field,0)), "estimated": bool(rec.get("estimated",False))})
        return sorted(hist, key=lambda r: r["year"])
    eu_hist = _hist(db_eu, "units_europe"); w_hist = _hist(db_world, "units_world")
    out["history_europe"] = eu_hist; out["history_world"] = w_hist
    disp = None
    for db in (db_eu, db_world):
        if mk in db and start_year in db[mk]:
            disp = db[mk][start_year]["model"]; break
    if disp is None:
        for db in (db_eu, db_world):
            if mk in db and db[mk]:
                disp = db[mk][sorted(db[mk].keys())[-1]]["model"]; break
    out["display_model"] = disp
    if mk in db_eu and start_year in db_eu[mk]:
        if normalize_generation(db_eu[mk][start_year].get("generation")) in accepted:
            out["eu"] = {"value": int(db_eu[mk][start_year]["units_europe"]), "source": "local-db", "is_model_level": True}
    if mk in db_world and start_year in db_world[mk]:
        if normalize_generation(db_world[mk][start_year].get("generation")) in accepted:
            out["world"] = {"value": int(db_world[mk][start_year]["units_world"]), "source": "local-db", "is_model_level": True}
    return out

# ------------------ Gen detection orchestrator (UPDATED) --
def autodetect_generation(db_eu, db_world, _icor_map_unused, model, user_year, user_gen):
    """
    Generation detection order:
      1) wikipedia_gen.detect_via_wikipedia  (NEW – best)
      2) Existing Wikipedia API parser
      3) SERP fallback
      4) User/local/default fallbacks
    Returns: (gen_label, (start,end), basis, note)
    """

    def _window_from_local_for_label(gen_label: str):
        det_norm = normalize_generation(gen_label)
        mk = find_best_model_key(db_eu, db_world, user_model=model)
        years = []
        if mk:
            for db in (db_eu, db_world):
                for y, rec in db.get(mk, {}).items():
                    if normalize_generation(rec.get("generation")) == det_norm:
                        years.append(y)
        return (min(years), max(years)) if years else None

    def _clip(window):
        gs, ge = window
        return (max(1990, gs), min(ge, DEADLINE_YEAR))

    # --- 1) WIKIPEDIA MODULE FIRST (high-accuracy) ---
    try:
        wm_label, wm_window, wm_diag = detect_via_wikipedia(model, user_year)
    except Exception:
        wm_label, wm_window, wm_diag = None, None, {"basis":"wikipedia_module","status":"exception"}

    if wm_window:
        start, end = wm_window
        # Resolve if open-ended or exactly 2025
        end_resolved = end
        if end >= DEADLINE_YEAR:
            end_resolved, _ = decide_continuation_if_present(model, (start, end), ref_year=2025)
            basis = "wikipedia_module_present_resolved"; note = "Window from wikipedia_gen; 'present' resolved."
        elif end == 2025:
            end_resolved, _ = decide_continuation_if_exact_2025(model, (start, end), ref_year=2025)
            basis = "wikipedia_module_end_2025_resolved"; note = "Window from wikipedia_gen; 2025 end resolved."
        else:
            basis = "wikipedia_module_fixed"; note = "Window from wikipedia_gen."
        return wm_label or (user_gen or "GEN"), _clip((start, end_resolved)), basis, note

    # --- 2) Existing WEB FIRST (Wikipedia API) ---
    if GEN_WINDOW_SOURCE == "web_first":
        gen_label_wiki, win_wiki, wiki_diag = detect_generation_via_wikipedia_api(model, user_year)
        if win_wiki:
            start, end = win_wiki
            end_resolved, _ = decide_continuation_if_present(model, (start, end), ref_year=2025)
            if end != DEADLINE_YEAR and end == 2025:
                end_resolved, _ = decide_continuation_if_exact_2025(model, (start, end), ref_year=2025)
                basis = "wikipedia_api_end_2025_resolved"
                note = "Wikipedia window; end=2025 resolved via web signals."
                return gen_label_wiki or (user_gen or "GEN"), _clip((start, end_resolved)), basis, note
            basis = "wikipedia_api_present_resolved" if end == DEADLINE_YEAR else "wikipedia_api_fixed"
            note = "Window from Wikipedia headings; 'present' resolved vs 2025." if end == DEADLINE_YEAR else "Window from Wikipedia headings."
            return gen_label_wiki or (user_gen or "GEN"), _clip((start, end_resolved)), basis, note

    # --- 3) SERP fallback for window ---
    def detect_generation_via_serp(model: str, year: int) -> Tuple[Optional[str], Optional[Tuple[int,int]], Optional[Dict[str,Any]]]:
        queries = [
            f'site:wikipedia.org "{model}" generation production',
            f'site:wikipedia.org "{model}" (Mk OR Mark OR generation) production',
            f'"{model}" generation production years',
            f'"{model}" model years generation',
        ]
        blobs = []
        for q in queries:
            res = serp_search(q)
            if "error" in res: continue
            blobs.extend(pull_text_blobs(res))
            time.sleep(0.25)
        RANGE = re.compile(r'(?:(?:Production|Model years)\s*[:\-]?\s*)?(20\d{2})\s*(?:–|-|to)\s*(20\d{2}|present)', re.I)
        GEN = re.compile(r'\b(?:mk|mark|gen(?:eration)?)\s?([ivx\d]{1,3})\b', re.I)
        best = None; best_score = -1e9; best_window = None; best_label = None
        def score_blob(t: str) -> int:
            s = 0
            if "wikipedia.org" in t.lower(): s += 20
            if str(year) in t: s += 5
            if "production" in t.lower() or "model years" in t.lower(): s += 5
            return s
        for b in blobs:
            t = (b.get("text") or "") + " " + (b.get("url") or "")
            if normalize_name(model) not in normalize_name(t): continue
            for m in RANGE.finditer(t):
                y1 = int(m.group(1))
                y2 = m.group(2)
                y2i = DEADLINE_YEAR if isinstance(y2, str) and y2.lower()=="present" else int(y2)
                if not (1990 <= y1 <= 2100 and y1 <= y2i <= 2100): continue
                in_view_bonus = 8 if (y1-1) <= year <= (y2i+1) else 0
                sc = score_blob(t) + in_view_bonus
                if sc > best_score:
                    best_score = sc
                    best_window = (y1, y2i)
                    g = GEN.search(t)
                    best_label = f"Mk{g.group(1).upper()}" if g else "GEN"
                    best = b
        return best_label, best_window, best

    web_gen, win_serp, _ = detect_generation_via_serp(model, user_year)
    if win_serp:
        start, end = win_serp
        if end == DEADLINE_YEAR:
            end_resolved, _ = decide_continuation_if_present(model, (start, end), ref_year=2025)
            return web_gen or (user_gen or "GEN"), _clip((start, end_resolved)), "web_serp_present_resolved", "SERP window; 'present' resolved vs 2025."
        if end == 2025:
            end_resolved, _ = decide_continuation_if_exact_2025(model, (start, end), ref_year=2025)
            return web_gen or (user_gen or "GEN"), _clip((start, end_resolved)), "web_serp_end_2025_resolved", "SERP window; 2025 end resolved."
        return web_gen or (user_gen or "GEN"), _clip((start, end)), "web_serp_fixed", "Window from web SERP."

    # --- 4) USER INPUT / LOCAL / DEFAULT ---
    if user_gen:
        gen_label = user_gen
        win_local = _window_from_local_for_label(gen_label)
        if win_local:
            return gen_label, _clip(win_local), "user_input", "Generation provided by user (window from local DB)."
        return gen_label, (max(1990, user_year-1), min(user_year+7, DEADLINE_YEAR)), "user_input_default", "User gen; default 8y window."

    mk = find_best_model_key(db_eu, db_world, user_model=model)
    if mk:
        for db in (db_eu, db_world):
            if mk in db and user_year in db[mk] and db[mk][user_year].get("generation"):
                gen_label = db[mk][user_year]["generation"]
                win_local = _window_from_local_for_label(gen_label)
                if win_local:
                    return gen_label, _clip(win_local), "local_top100", "Detected from Top100 at the user year."
        years = sorted(set(list(db_eu.get(mk, {}).keys()) + list(db_world.get(mk, {}).keys())))
        prev_years = [y for y in years if y <= user_year and (
            (mk in db_eu and y in db_eu[mk] and db_eu[mk][y].get("generation")) or
            (mk in db_world and y in db_world[mk] and db_world[mk][y].get("generation"))
        )]
        if prev_years:
            y0 = prev_years[-1]
            gen_label = db_eu.get(mk, {}).get(y0, db_world.get(mk, {}).get(y0))["generation"]
            win_local = _window_from_local_for_label(gen_label)
            if win_local:
                return gen_label, _clip(win_local), "local_top100", f"Detected from Top100 nearest ≤ year ({y0})."
        next_years = [y for y in years if y >= user_year and (
            (mk in db_eu and y in db_eu[mk] and db_eu[mk][y].get("generation")) or
            (mk in db_world and y in db_world[mk] and db_world[mk][y].get("generation"))
        )]
        if next_years and (next_years[0] - user_year) <= MAX_FUTURE_GAP:
            y1 = next_years[0]
            gen_label = db_eu.get(mk, {}).get(y1, db_world.get(mk, {}).get(y1))["generation"]
            win_local = _window_from_local_for_label(gen_label)
            if win_local:
                return gen_label, _clip(win_local), "future_top100", f"Detected from Top100 nearest ≥ year ({y1})."

    gen_label = "GEN"
    return gen_label, (user_year, min(user_year+8, DEADLINE_YEAR)), "default_8yr", "Fallback default 8-year window."

# ------------------ Priors & constraints ------------
def model_total_eu_for_year(db_eu, user_model, year) -> Optional[int]:
    mk = find_best_model_key(db_eu, {}, user_model=user_model)
    if not mk: return None
    rec = db_eu.get(mk, {}).get(year)
    return int(rec["units_europe"]) if rec and "units_europe" in rec else None

def prior_generation_avg_eu(db_eu, user_model, detected_gen: str, start_year: int, lookback_years: int = 3) -> Optional[int]:
    mk = find_best_model_key(db_eu, {}, user_model=user_model)
    if not mk: return None
    det_norm = normalize_generation(detected_gen)
    vals = []
    for y in range(start_year-1, max(1990, start_year - lookback_years) - 1, -1):
        rec = db_eu.get(mk, {}).get(y)
        if not rec: continue
        if normalize_generation(rec.get("generation")) == det_norm:  # same-gen
            continue
        units = rec.get("units_europe")
        if isinstance(units, int): vals.append(units)
    if not vals: return None
    return int(sum(vals) / len(vals))

def infer_eu_share_bounds(db_eu, model_key: Optional[str], start_year: int) -> Tuple[Tuple[float,float], Dict[str,Any]]:
    def presence_years(db, mk): return len(db.get(mk, {}))
    def rank_for(db, mk, y, units_field):
        rows = []
        for _mk, by in db.items():
            if y in by:
                rows.append((_mk, by[y][units_field]))
        rows.sort(key=lambda t: t[1], reverse=True)
        m = {mk_: i+1 for i, (mk_, _) in enumerate(rows)}
        return m.get(mk)
    diag = {"basis":"presence+rank","rank":None,"presence_years":0,"bands":None}
    if not model_key:
        diag["bands"] = "unknown_model"; return (0.01,0.25), diag
    presence = presence_years(db_eu, model_key)
    rank = rank_for(db_eu, model_key, start_year, "units_europe")
    diag["rank"]=rank; diag["presence_years"]=presence
    if rank is None:
        if presence>=5: lo,hi=0.08,0.60; diag["bands"]="freq_present_no_rank"
        elif presence>=1: lo,hi=0.04,0.40; diag["bands"]="sporadic_present_no_rank"
        else: lo,hi=0.01,0.25; diag["bands"]="never_present"
        return (lo,hi), diag
    if rank<=10:   lo,hi=0.25,0.85; diag["bands"]="rank<=10"
    elif rank<=30: lo,hi=0.15,0.65; diag["bands"]="rank<=30"
    elif rank<=60: lo,hi=0.08,0.50; diag["bands"]="rank<=60"
    else:          lo,hi=0.03,0.35; diag["bands"]="rank<=100"
    return (lo,hi), diag

def build_constraints(start_year: int, display_model: str, target_gen: str,
                      gen_window: Tuple[int,int], db_eu, local, web) -> Tuple[dict, dict, dict]:
    seed = {
        "model": display_model,
        "generation": target_gen,
        "generation_window": {"start": gen_window[0], "end": gen_window[1]},
        "generation_window_basis": None,
        "year": start_year,
        "europe": None,
        "world": None,
        "history_europe": local.get("history_europe", []),
        "history_world": local.get("history_world", []),
        "notes": "Wikipedia-first window; Local Top100 EU/World authoritative for seeds if present. "
                 "Web seeding is used only if local has no coverage."
    }
    constraints = {"world": {}, "europe": {}, "zero_years": []}
    plaus = {"flag": False, "reason": "", "source_note": ""}

    zero_years = list(range(1990, gen_window[0])) + list(range(gen_window[1]+1, DEADLINE_YEAR+1))
    constraints["zero_years"] = [y for y in zero_years if y >= start_year]

    eu_val = None; world_val = None
    if local.get("eu"):
        eu_val = int(local["eu"]["value"])
        seed["europe"] = {"value": eu_val, "source": local["eu"]["source"], "is_model_level": True}
        constraints["europe"]["exact"] = {start_year: eu_val}
    if local.get("world"):
        world_val = int(local["world"]["value"])
        seed["world"] = {"value": world_val, "source": local["world"]["source"], "is_model_level": True}
        constraints["world"]["exact"] = {start_year: min(world_val, WORLD_MAX_CAP)}

    if (eu_val is None and world_val is None
        and not seed["history_europe"] and not seed["history_world"]):
        if web and web.get("europe"):
            eu_val = int(web["europe"]["value"])
            seed["europe"] = {"value": eu_val, "source": "web-serp", "is_model_level": bool(web["europe"].get("is_model_level"))}
            constraints.setdefault("europe",{}).setdefault("exact",{})[start_year] = eu_val
        if web and web.get("world"):
            world_val = int(web["world"]["value"])
            seed["world"] = {"value": world_val, "source": "web-serp", "is_model_level": bool(web["world"].get("is_model_level"))}
            constraints.setdefault("world",{}).setdefault("exact",{})[start_year] = min(world_val, WORLD_MAX_CAP)

    if seed.get("europe") and str(seed["europe"].get("source","")).startswith("web"):
        prior_avg  = prior_generation_avg_eu(db_eu, display_model, target_gen, start_year)
        model_tot  = model_total_eu_for_year(db_eu, display_model, start_year)
        floors, notes = [], []
        if prior_avg:
            floors.append(int(0.10 * prior_avg)); notes.append(f"10% prior-gen avg {prior_avg:,}")
        if model_tot:
            floors.append(int(0.05 * model_tot)); notes.append(f"5% model EU {model_tot:,}")
        if floors:
            floor_val = max(floors + [20_000])
            if eu_val < floor_val:
                plaus["flag"] = True
                plaus["reason"] = f"EU seed {eu_val:,} is very small vs " + " & ".join(notes)
                eu_val = floor_val
                seed["europe"]["value"] = eu_val
                constraints["europe"]["exact"][start_year] = eu_val
        plaus["source_note"] = "EU seed derived from web; checked vs local history"
    elif seed.get("europe") and str(seed["europe"].get("source","")).startswith("local"):
        plaus["source_note"] = "EU seed from local Top100"

    if world_val is None and eu_val is not None:
        (lo_share, hi_share), diag = infer_eu_share_bounds(db_eu, local.get("model_key"), start_year)
        world_min = max(eu_val, int(math.ceil(eu_val / max(hi_share, 1e-6))))
        world_max = int(min(WORLD_MAX_CAP, math.floor(eu_val / max(lo_share, 1e-6))))
        constraints["world"]["range"] = {start_year: (world_min, world_max)}
        seed["eu_share_prior"] = {"low": lo_share, "high": hi_share,
                                  "rank": diag["rank"], "presence_years": diag["presence_years"]}
    return seed, constraints, plaus

# ------------------ OpenAI call --------------------
def build_messages(car_model: str, target_gen: str, start_year: int, seed: dict, constraints: dict) -> List[Dict[str,str]]:
    seed_text = json.dumps(seed, ensure_ascii=False, indent=2)
    cons_text = json.dumps(constraints, ensure_ascii=False, indent=2)
    user_prompt = (
        f"Model: {car_model}\n"
        f"Generation: {target_gen}\n"
        f"Generation window (inclusive): {seed['generation_window']}\n"
        f"Starting year: {start_year}\n\n"
        f"Seed & history (THIS GENERATION ONLY):\n{seed_text}\n\n"
        f"HARD RULES:\n{cons_text}\n"
        f"- Generation-only forecast; zero outside window.\n"
        f"- Apply exact/range constraints at start year.\n"
        f"- Ensure Europe ≤ World each year.\n\n"
        f"Task: Estimate annual unit sales from {start_year} through {DEADLINE_YEAR} for this generation only. "
        f"Return ONE JSON object with fields: model, generation_or_trim_context, start_year, end_year, assumptions[], "
        f"methodology_summary, confidence, yearly_estimates:[{{year:int, world_sales_units:int, europe_sales_units:int, rationale:string}}], notes."
    )
    return [{"role":"system","content":SYSTEM_INSTRUCTIONS},{"role":"user","content":user_prompt}]

def call_openai(messages: List[Dict[str,str]]) -> Dict[str,Any]:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, response_format={"type":"json_object"}
    )
    content = completion.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if "```" in content:
            chunk = content.split("```json")[-1].split("```")[0]
            return json.loads(chunk)
        raise RuntimeError("Model did not return valid JSON.")

# ------------------ Enforcement & smoothing --------
def enforce_constraints_and_zero(data: dict, constraints: dict, start_year: int) -> dict:
    rows = data.setdefault("yearly_estimates", [])
    row = next((r for r in rows if r.get("year")==start_year), None)
    if row is None:
        row = {"year": start_year, "world_sales_units": 0, "europe_sales_units": 0, "rationale": ""}; rows.append(row)
    def clamp(v, lo, hi): return max(lo, min(hi, v))
    ce = constraints.get("europe", {})
    if "exact" in ce: 
        row["europe_sales_units"] = int(ce["exact"][start_year])
        row["rationale"] = (row.get("rationale","") + " Europe start fixed.").strip()
    cw = constraints.get("world", {})
    if "exact" in cw:
        row["world_sales_units"] = int(cw["exact"][start_year])
        row["rationale"] = (row.get("rationale","") + " World start fixed.").strip()
    elif "range" in cw:
        lo, hi = cw["range"][start_year]
        lo = max(lo, row["europe_sales_units"]); hi = max(hi, lo)
        cur = int(row.get("world_sales_units", 0))
        row["world_sales_units"] = clamp(cur, lo, hi)
    if row["europe_sales_units"] > row["world_sales_units"]:
        row["world_sales_units"] = row["europe_sales_units"]
    needed = set(range(start_year, DEADLINE_YEAR+1))
    have = {r.get("year") for r in rows}
    for y in sorted(needed - have):
        rows.append({"year": y, "world_sales_units": 0, "europe_sales_units": 0, "rationale":"Filled by client."})
    rows.sort(key=lambda r: r["year"])
    zero_years = set(constraints.get("zero_years", []))
    protected = set()
    if "exact" in ce: protected.update(ce["exact"].keys())
    if "exact" in cw: protected.update(cw["exact"].keys())
    for r in rows:
        if r["year"] in zero_years and r["year"] not in protected:
            r["world_sales_units"] = 0
            r["europe_sales_units"] = 0
            r["rationale"] = (r.get("rationale","") + " Outside generation window -> zero.").strip()
    return data

def smooth_lifecycle(data: dict, start_year: int, zero_years: set):
    rows = sorted(data.get("yearly_estimates", []), key=lambda r: r["year"])
    if not rows: return data
    def smooth(series):
        y0, v0 = series[0]; out=[(y0, max(0,v0))]
        for i in range(1,len(series)):
            y, s = series[i]
            if y in zero_years: out.append((y,0)); continue
            _, vp = out[-1]
            d = y - y0
            if d==1: g=0.35
            elif 2<=d<=3: g=0.12
            elif 4<=d<=5: g=0.05
            else: g=-0.08
            prop = int(round(vp*(1+g)))
            out.append((y, int(max(0, 0.6*prop + 0.4*s))))
        return out
    yrs = [r["year"] for r in rows]
    wsm = smooth([(r["year"], int(r.get("world_sales_units",0))) for r in rows])
    esm = smooth([(r["year"], int(r.get("europe_sales_units",0))) for r in rows])
    for i,_ in enumerate(yrs):
        w = min(max(0, wsm[i][1]), WORLD_MAX_CAP)
        e = min(max(0, esm[i][1]), w)
        rows[i]["world_sales_units"] = w
        rows[i]["europe_sales_units"] = e
    data["yearly_estimates"] = rows
    return data

# ------------------ Fleet & repairs ----------------
def compute_fleet_and_repairs(rows: List[Dict[str,Any]], decay_rate=DECAY_RATE, repair_rate=REPAIR_RATE) -> List[Dict[str,Any]]:
    years = [r["year"] for r in rows]
    ws = [int(r.get("world_sales_units",0)) for r in rows]
    es = [int(r.get("europe_sales_units",0)) for r in rows]
    n = len(years)
    wf, ef = [0.0]*n, [0.0]*n
    for i in range(n):
        y0, w0, e0 = years[i], ws[i], es[i]
        for j in range(i, n):
            age = years[j] - y0
            surv = 1.0 if age<=1 else (1.0 - decay_rate)**(age-1)
            wf[j] += w0*surv; ef[j] += e0*surv
    res = []
    for i in range(n):
        res.append({
            "year": years[i],
            "world_fleet": int(round(wf[i])),
            "europe_fleet": int(round(ef[i])),
            "world_repairs": int(round(wf[i]*repair_rate)),
            "europe_repairs": int(round(ef[i]*repair_rate)),
        })
    return res

# ------------------ Output: CSV & Excel ------------
def save_csv(data: Dict[str,Any], base: str) -> str:
    csv_name = f"{base}.csv"
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["year","world_sales_units","europe_sales_units"])
        for r in data.get("yearly_estimates", []):
            w.writerow([r.get("year"), r.get("world_sales_units"), r.get("europe_sales_units")])
    return csv_name

def save_excel(data: Dict[str,Any], fleet_repair: List[Dict[str,Any]],
               seed: Dict[str,Any], constraints: Dict[str,Any],
               diag: Dict[str,Any], base: str) -> str:
    estimates = pd.DataFrame([
        {"Year": r["year"], "World_Sales": int(r.get("world_sales_units",0)),
         "Europe_Sales": int(r.get("europe_sales_units",0)), "Rationale": r.get("rationale","")}
        for r in data.get("yearly_estimates", [])
    ])
    fr = pd.DataFrame(fleet_repair)[["year","world_fleet","europe_fleet","world_repairs","europe_repairs"]] \
           .rename(columns={"year":"Year","world_fleet":"World_Fleet","europe_fleet":"Europe_Fleet",
                            "world_repairs":"World_Windshield_Repairs","europe_repairs":"Europe_Windshield_Repairs"})
    estimates_merged = estimates.merge(fr, on="Year", how="left")
    gen_win = seed.get("generation_window", {})
    gstart, gend = gen_win.get("start"), gen_win.get("end")
    estimates_merged["Gen_Active"] = estimates_merged["Year"].apply(lambda y: bool(gstart is not None and gend is not None and gstart <= y <= gend))
    estimates_merged["ICOR_Supported"] = diag.get("supported_flag")
    estimates_merged["ICOR_Match_Type"] = diag.get("match_type")

    summary = pd.DataFrame({
        "Model": [data.get("model")],
        "Generation_Input_or_Detected": [seed.get("generation")],
        "Generation_Window_Start": [gstart],
        "Generation_Window_End": [gend],
        "Generation_Window_Basis": [seed.get("generation_window_basis")],
        "Generation_Alias_Used": [seed.get("generation_alias_used")],
        "Generation_Context_From_Model": [data.get("generation_or_trim_context")],
        "Start_Year": [data.get("start_year")],
        "End_Year": [data.get("end_year")],
        "Confidence": [data.get("confidence")],
        "Plausibility_Flag": [bool(diag.get("plausibility", {}).get("flag"))],
        "Plausibility_Reason": [diag.get("plausibility", {}).get("reason","")],
        "Seed_Source_Note": [diag.get("plausibility", {}).get("source_note","")],
        "ICOR_Supported": [diag.get("supported_flag")],
        "ICOR_Match_Type": [diag.get("match_type")],
        "ICOR_Matched_Row": [json.dumps(diag.get("matched_row"), ensure_ascii=False)],
    })

    seeds_constraints = pd.DataFrame({
        "Seed_or_Constraint": ["seed","constraints"],
        "JSON": [json.dumps(seed, indent=2), json.dumps(constraints, indent=2)],
    })

    xlsx = f"{base}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        estimates_merged.to_excel(writer, sheet_name="Estimates", index=False)
        fr.to_excel(writer, sheet_name="Fleet_Repairs", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        seeds_constraints.to_excel(writer, sheet_name="Seeds_Constraints", index=False)
    return xlsx

# ------------------ ICOR support (unchanged) -------
def load_icor_catalog_csv_json(path: str) -> Optional[pd.DataFrame]:
    try:
        if path.lower().endswith(".csv"): return pd.read_csv(path)
        if path.lower().endswith(".json"): return pd.read_json(path)
    except Exception as e:
        print(f"[WARN] Failed to read ICOR CSV/JSON at {path}: {e}")
    return None

def parse_icor_supported_txt(path: str) -> Optional[Dict[str, Dict[int, str]]]:
    if not os.path.exists(path): return None
    try:
        text = open(path, "r", encoding="utf-8").read()
    except Exception as e:
        print(f"[WARN] Cannot read ICOR TXT at {path}: {e}"); return None
    i, n = 0, len(text); mapping={}
    while i < n:
        m = re.search(r'"([^"]+)"\s*:', text[i:])
        if not m: break
        model = m.group(1); start = i + m.end()
        b = re.search(r'\{', text[start:])
        if not b: break
        brace_start = start + b.start()
        depth, j = 0, brace_start
        while j < n:
            if text[j]=='{': depth += 1
            elif text[j]=='}':
                depth -= 1
                if depth == 0: break
            j += 1
        if j >= n: break
        inner = text[brace_start+1:j]
        for y, gen in re.findall(r'(\d{4})\s*:\s*"([^"]+)"', inner):
            mapping.setdefault(normalize_name(model), {})[int(y)] = gen
        i = j + 1
    return mapping

def check_icor_support(icor_map, icor_df, model_name, generation_input, start_year):
    if icor_map:
        norm = normalize_name(model_name)
        if norm in icor_map:
            years = sorted(icor_map[norm].keys())
            active = [y for y in years if y <= start_year]
            if active:
                yact = max(active); icor_gen = icor_map[norm][yact]
                match = "by_year_exact_gen" if normalize_generation(icor_gen)==normalize_generation(generation_input) else "by_year_diff_gen_label"
                return {"supported_flag": True, "match_type": match, "matched_row": {"model": model_name, "icor_gen_code": icor_gen, "icor_gen_from_year": yact}}
            return {"supported_flag": False, "match_type": "model_present_no_year_coverage", "matched_row": {"model": model_name}}
        return {"supported_flag": False, "match_type": "no_model_match", "matched_row": None}
    if icor_df is None or icor_df.empty:
        return {"supported_flag":"unknown","match_type":"no_catalog","matched_row":None}
    cols = {c.lower(): c for c in icor_df.columns}
    if "model" not in cols: return {"supported_flag":"unknown","match_type":"no_model_column","matched_row":None}
    norm_target_model = normalize_name(model_name)
    rows = icor_df[icor_df[cols["model"]].apply(lambda v: normalize_name(str(v)) == norm_target_model)]
    if rows.empty:
        return {"supported_flag": False, "match_type": "no_model_match", "matched_row": None}
    return {"supported_flag": True, "match_type": "model_present_diff_gen_no_year_map", "matched_row": {"model": rows.iloc[0][cols["model"]] }}

# ------------------ Summary print ------------------
def print_summary(data, seed, constraints, gen_basis, icor_status, autodetect_note=""):
    print("\n=== Generation-Specific Car Sales Estimates ===")
    print(f"Model: {data.get('model','N/A')} | Generation: {seed.get('generation')}  [basis: {gen_basis}]")
    if autodetect_note: print(f"Auto-detect note: {autodetect_note}")
    gw = seed.get("generation_window",{})
    print(f"Generation window: {gw.get('start')}–{gw.get('end')}")
    print(f"Coverage: {data.get('start_year')}–{data.get('end_year')}")
    print(f"Confidence: {data.get('confidence','N/A')}")
    if seed.get("europe"):
        print(f"  Europe {seed['year']}: ~{seed['europe']['value']:,}  [{seed['europe']['source']}]")
    if seed.get("world"):
        print(f"  World  {seed['year']}: ~{seed['world']['value']:,}  [{seed['world']['source']}]")
    rows = data.get("yearly_estimates",[])
    if rows:
        tbl = [[r["year"], r["world_sales_units"], r["europe_sales_units"]] for r in rows[:10]]
        print(tabulate(tbl, headers=["Year","World","Europe"], tablefmt="github", numalign="right"))

# ------------------ CLI inputs & main --------------
def ask_user_inputs() -> Dict[str,Any]:
    model = input("Enter the car model: ").strip()
    generation = input("Enter the generation tag (blank to auto): ").strip()
    year = int(input("Enter the starting year you'd like to view from (we will model from launch): ").strip())
    if year < 1990 or year > DEADLINE_YEAR:
        print(f"Year must be between 1990 and {DEADLINE_YEAR}.", file=sys.stderr); sys.exit(1)
    return {"car_model": model, "generation": generation, "start_year": year}

def detect_generation_via_web_legacy(model: str, year: int) -> Tuple[Optional[str], Optional[Tuple[int,int]], Optional[Dict[str,Any]]]:
    """Kept for API compatibility if needed elsewhere (not used)."""
    return detect_generation_via_wikipedia_api(model, year)

def main():
    user = ask_user_inputs()

    # Local DBs from CWD (Streamlit page sets cwd=data/)
    db_eu = load_local_database_eu(os.getcwd())
    db_world = load_local_database_world(os.getcwd())

    # ICOR (kept only for "support" flag in Summary)
    here = os.path.dirname(os.path.abspath(__file__))
    icor_txt_path = os.path.join(os.getcwd(), "icor_supported_models.txt")
    if not os.path.exists(icor_txt_path):
        icor_txt_path = os.path.join(here, "icor_supported_models.txt")
    icor_map = parse_icor_supported_txt(icor_txt_path)
    icor_df = None

    # Gen detect (wikipedia module first) — returns full window
    gen_label, gen_window, gen_basis, autodetect_note = autodetect_generation(
        db_eu, db_world, icor_map, user["car_model"], user["start_year"], user["generation"]
    )

    # Launch year = window start (we model from launch, not from user's year)
    launch_year = max(1990, gen_window[0])

    # Alias (optional)
    alias = infer_local_gen_alias(db_eu, db_world, user["car_model"], gen_label, gen_window)

    # Local seeds/history for that generation AT LAUNCH YEAR
    local = local_seed_for_generation(
        db_eu, db_world, user["car_model"], gen_label, launch_year,
        accepted_alias=alias, window=gen_window
    )

    # Web only if no local coverage at all
    use_web = not (local.get("history_europe") or local.get("history_world") or local.get("eu") or local.get("world"))
    web = serp_seed(local.get("display_model") or user["car_model"], gen_label, launch_year) if use_web else {}

    # Build prompt seeds & constraints (+plausibility) at LAUNCH YEAR
    seed, constraints, plaus = build_constraints(
        launch_year, local.get("display_model") or user["car_model"],
        gen_label, gen_window, db_eu, local, web
    )
    seed["generation_window_basis"] = gen_basis
    if alias: seed["generation_alias_used"] = alias

    # Call OpenAI
    messages = build_messages(local.get("display_model") or user["car_model"], gen_label, launch_year, seed, constraints)
    data = call_openai(messages)

    # Enforce + smooth
    data = enforce_constraints_and_zero(data, constraints, launch_year)
    if APPLY_SMOOTHING:
        data = smooth_lifecycle(data, launch_year, set(constraints.get("zero_years", [])))

    # Horizon coverage (from LAUNCH YEAR)
    need = set(range(launch_year, DEADLINE_YEAR+1))
    have = {r.get("year") for r in data.get("yearly_estimates", [])}
    for y in sorted(need - have):
        data.setdefault("yearly_estimates", []).append({"year": y, "world_sales_units": 0, "europe_sales_units": 0, "rationale": "Filled by client."})
    data["yearly_estimates"] = sorted(data["yearly_estimates"], key=lambda r: r["year"])
    data["start_year"] = launch_year
    data["end_year"] = DEADLINE_YEAR
    data["model"] = local.get("display_model") or user["car_model"]

    # Fleet + ICOR support
    fleet = compute_fleet_and_repairs(data["yearly_estimates"], DECAY_RATE, REPAIR_RATE)
    icor_status = check_icor_support(icor_map, icor_df, local.get("display_model") or user["car_model"], gen_label, launch_year)
    icor_status["plausibility"] = plaus

    # -------------------- SAVE --------------------
    # Filenames now use the *user's* typed model to avoid fuzzy-mismatch naming.
    safe_model_for_name = user["car_model"].replace(" ", "_")
    safe_gen = normalize_generation(gen_label or "GEN").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"sales_estimates_{safe_model_for_name}_{safe_gen}_{launch_year}_{MODEL_NAME}_{timestamp}"

    csv_path = save_csv(data, base)
    xlsx_path = save_excel(data, fleet, seed, constraints, icor_status, base)

    # Console summary
    print_summary(data, seed, constraints, gen_basis, icor_status, autodetect_note)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Excel: {xlsx_path}")
    print(f"\nNotes: {plaus.get('source_note','')}")
    print("- Wikipedia-first window (module) with fallbacks; local Top100 used for seeds/history when available.")
    print(f"- User requested view-from year: {user['start_year']}; modeled from LAUNCH year: {launch_year}.")

if __name__ == "__main__":
    main()

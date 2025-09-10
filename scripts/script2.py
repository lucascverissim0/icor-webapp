#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window-first generation sales estimator (EU + World)

Key behaviors:
- Detect the generation WINDOW that is active at the user's chosen year:
  * Prefer local Top100 "presence windows" that cover the year
  * Else detect a window from the web (year-aware: 2012–2018, 2023–present, since 2023, launched in 2023)
  * Else default to an 8-year window starting at the user's year
- Use local Top100 data ONLY within that window; web seeds fill gaps (EU/World) at the launch year.
- Guardrails:
  * EU ≤ World
  * Zero outside generation window
  * EU→World range prior when World seed is missing
- Outputs: CSV + Excel with Estimates, Fleet_Repairs, Summary, Seeds_Constraints
"""

import os, re, csv, json, glob, time, math, sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import difflib
import requests
import pandas as pd
from tabulate import tabulate
from collections import Counter

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

SYSTEM_INSTRUCTIONS = (
    "You are an automotive market analyst. Use provided seed data and constraints as anchors. "
    "This forecast is generation-window-specific; do not mix other generations. "
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

# ------------------ Shared lookups -----------------
def find_best_model_key(*dbs: Dict[str, Dict[int, Dict[str, Any]]], user_model: str) -> Optional[str]:
    """
    Token-aware resolver:
    - Lowercases & tokenizes
    - Requires query tokens ⊆ candidate tokens (prevents matching 'Transit' when user typed 'Transit Custom')
    - Light sibling guards for Ford family (extend as needed)
    """
    qnorm = normalize_name(user_model)
    qtokens = set(qnorm.split())
    all_keys = sorted(set(k for db in dbs for k in db.keys()))
    if qnorm in all_keys:
        return qnorm

    forbids = []
    if "custom" in qtokens:  forbids += ["connect", "tourneo", "courier"]
    if "connect" in qtokens: forbids += ["custom", "tourneo", "courier"]
    if "tourneo" in qtokens: forbids += ["custom", "connect", "courier"]

    def ok(cand: str) -> bool:
        ct = set(cand.split())
        if not qtokens.issubset(ct): return False
        return not any(b in ct for b in forbids)

    filtered = [k for k in all_keys if ok(k)]
    if filtered:
        return sorted(filtered, key=lambda s: (len(s), s))[0]

    cand = difflib.get_close_matches(qnorm, all_keys, n=5, cutoff=0.8)
    cand = [k for k in cand if ok(k)]
    return cand[0] if cand else None

def _presence_years_for_model(db_eu, db_world, mk: str) -> List[int]:
    years = set()
    for db in (db_eu, db_world):
        for y in db.get(mk, {}).keys():
            years.add(y)
    return sorted(y for y in years if 1990 <= y <= DEADLINE_YEAR)

def _windows_from_presence(years: List[int]) -> List[Tuple[int,int]]:
    if not years: return []
    windows = []
    s = years[0]; p = years[0]
    for y in years[1:]:
        if y == p + 1:
            p = y
        else:
            windows.append((s, p))
            s = p = y
    windows.append((s, p))
    return windows

def _majority_gen_label_for_window(db_eu, db_world, mk: str, window: Tuple[int,int]) -> Optional[str]:
    lo, hi = window
    labels = []
    for db in (db_eu, db_world):
        for y, rec in db.get(mk, {}).items():
            if lo <= y <= hi:
                g = rec.get("generation")
                if g: labels.append(normalize_generation(g))
    if not labels: return None
    c = Counter(labels); lab, _ = c.most_common(1)[0]
    return lab

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

def build_search_queries(model: str, gen: str, year: int) -> List[str]:
    # Force exact phrase for the model (reduces hits for Transit / Connect / Tourneo)
    exact = f'"{model}"'
    nm = normalize_name(model)
    minus_parts = []
    sib_exclusions = ["transit connect", "tourneo", "transit courier"]
    for sib in sib_exclusions:
        if sib not in nm:
            minus_parts.append(f'-"{sib.title()}"')
    if "transit custom" in nm:
        minus_parts.append('-"Transit"')
    minus = " ".join(minus_parts) + " -price -msrp -€ -$"

    gen_alias = " OR ".join(f'"{a}"' for a in build_generation_aliases(gen)) if gen else ""
    with_gen = f'({gen_alias}) ' if gen_alias else ""

    base = [
        f'{exact} {with_gen}{year} generation europe {minus}',
        f'{exact} {with_gen}{year} launch europe {minus}',
        f'{exact} {with_gen}{year} introduced europe {minus}',
        f'{exact} {with_gen}{year} registrations europe {minus}',
        f'{exact} {with_gen}{year} global sales {minus}',
        f'{exact} {with_gen}{year} worldwide sales {minus}',
    ]
    preferred = [
        f'site:wikipedia.org {exact} generation {year}',
        f'site:carsalesbase.com {exact} sales {year}',
        f'site:acea.auto {exact} registrations {year}',
        f'site:marklines.com {exact} sales {year}',
    ]
    return base + preferred

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
        if normalize_name(model) not in normalize_name(text): continue
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

def detect_generation_window_via_web(model: str, year: int) -> Optional[Tuple[int,int]]:
    """
    Parse web snippets for windows like '2012–2018', '2023–present', 'launched in 2023', 'since 2023'.
    Score to prefer windows overlapping 'year'; penalize those ending before 'year'.
    Return (start,end) or None.
    """
    queries = [
        f'"{model}" {year} generation europe -price -msrp -€ -$',
        f'"{model}" {year} launch europe -price -msrp -€ -$',
        f'"{model}" {year} introduced europe -price -msrp -€ -$',
        f'site:wikipedia.org "{model}" generation {year}',
    ]
    YEAR_RANGE = re.compile(r'(20\d{2})\s*(?:–|-|to|—)\s*(20\d{2}|present)', re.I)
    SINCE = re.compile(r'(?:since|from|introduced in|launched in|debuted in)\s+(20\d{2})', re.I)

    def extract_window(text: str) -> Optional[Tuple[int,int]]:
        m = YEAR_RANGE.search(text)
        if m:
            y1 = int(m.group(1))
            y2 = m.group(2).lower()
            if y2 == "present": return (y1, DEADLINE_YEAR)
            try:
                y2i = int(y2)
                if 1990 <= y1 <= 2100 and y1 <= y2i <= 2100: return (y1, y2i)
            except: pass
        m2 = SINCE.search(text)
        if m2:
            y1 = int(m2.group(1))
            if 1990 <= y1 <= 2100: return (y1, DEADLINE_YEAR)
        return None

    def exact_model_present(t: str) -> bool:
        return normalize_name(model) in normalize_name(t)

    best, best_score = None, -1e9
    blobs = []
    for q in queries:
        res = serp_search(q)
        if "error" not in res: blobs.extend(pull_text_blobs(res))
        time.sleep(0.25)

    for b in blobs:
        text = (b.get("text") or "") + " " + (b.get("url") or "")
        if not exact_model_present(text): continue
        win = extract_window(text)
        if not win: continue
        s, e = win
        score = 0.0
        if s <= year <= e: score += 200
        elif e < year:     score -= 200
        else:              score -= min(60, (s - year) * 10)
        score -= abs(s - year) * 2
        if "wikipedia.org" in text.lower(): score += 40
        if score > best_score:
            best_score, best = score, win
    return best

# ------------------ Local-first, window-first  -----
def detect_window_then_collect_local(db_eu, db_world, model: str, user_year: int):
    """
    1) Pick generation window that is active at user_year.
       - Prefer local presence windows that contain user_year.
       - Else try web to get a window; if web returns a window, use that.
       - Else default to (user_year, min(user_year+8, DEADLINE_YEAR)).
    2) Within that window, collect local EU/World rows for the model (any label).
    Return: window:(start,end), basis, gen_label_guess, mk, eu_hist, w_hist
    """
    mk = find_best_model_key(db_eu, db_world, user_model=model)
    window = None; basis = None
    gen_label_guess = None

    if mk:
        pres_years = _presence_years_for_model(db_eu, db_world, mk)
        windows = _windows_from_presence(pres_years)
        for w in windows:
            if w[0] <= user_year <= w[1]:
                window = w; basis = "local_presence"
                gen_label_guess = _majority_gen_label_for_window(db_eu, db_world, mk, w)
                break

    if not window:
        wweb = detect_generation_window_via_web(model, user_year)
        if wweb:
            window = (max(1990, wweb[0]), min(wweb[1], DEADLINE_YEAR)); basis = "web_window"
        else:
            window = (user_year, min(user_year+8, DEADLINE_YEAR)); basis = "default_8yr"

    lo, hi = window
    eu_hist, w_hist = [], []
    if mk:
        for y, rec in db_eu.get(mk, {}).items():
            if lo <= y <= hi and isinstance(rec.get("units_europe"), int):
                eu_hist.append({"year": y, "units": int(rec["units_europe"]), "estimated": bool(rec.get("estimated", False))})
        for y, rec in db_world.get(mk, {}).items():
            if lo <= y <= hi and isinstance(rec.get("units_world"), int):
                w_hist.append({"year": y, "units": int(rec["units_world"]), "estimated": bool(rec.get("estimated", False))})
        eu_hist.sort(key=lambda r: r["year"]); w_hist.sort(key=lambda r: r["year"])

    return window, basis, gen_label_guess, mk, eu_hist, w_hist

# ------------------ Seeds & constraints (window) ---
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

def build_constraints_from_window(window: Tuple[int,int], display_model: str,
                                  db_eu, db_world, mk: Optional[str],
                                  eu_hist: List[Dict[str,Any]], w_hist: List[Dict[str,Any]]):
    start_year, end_year = window
    seed = {
        "model": display_model,
        "generation": "",  # optional label; window-first approach
        "generation_window": {"start": start_year, "end": end_year},
        "generation_window_basis": None,
        "year": start_year,
        "europe": None,   # may become a dict below
        "world": None,    # may become a dict below
        "history_europe": eu_hist,
        "history_world": w_hist,
        "notes": "Window-first seeding: use local Top100 within window; web only for gaps.",
    }
    constraints = {"world": {}, "europe": {}, "zero_years": [y for y in range(end_year+1, DEADLINE_YEAR+1)]}
    plaus = {"flag": False, "reason": "", "source_note": ""}

    def _ensure_branch_dict(branch: str):
        """Make sure seed[branch] is a dict before assigning keys into it."""
        if not isinstance(seed.get(branch), dict):
            seed[branch] = {}

    eu_val = None
    w_val = None

    # ---- Prefer local exact seeds at launch if present
    if mk:
        if mk in db_eu and start_year in db_eu[mk] and isinstance(db_eu[mk][start_year].get("units_europe"), int):
            eu_val = int(db_eu[mk][start_year]["units_europe"])
            _ensure_branch_dict("europe")
            seed["europe"].update({"value": eu_val, "source": "local-db", "is_model_level": True})
            constraints.setdefault("europe", {}).setdefault("exact", {})[start_year] = eu_val

        if mk in db_world and start_year in db_world[mk] and isinstance(db_world[mk][start_year].get("units_world"), int):
            w_val = int(db_world[mk][start_year]["units_world"])
            _ensure_branch_dict("world")
            seed["world"].update({"value": w_val, "source": "local-db", "is_model_level": True})
            constraints.setdefault("world", {}).setdefault("exact", {})[start_year] = min(w_val, WORLD_MAX_CAP)

    # ---- If missing, try web seeding strictly for start year
    if eu_val is None or w_val is None:
        web = serp_seed(display_model, "", start_year)  # gen label not required

        if eu_val is None and web.get("europe"):
            eu_val = int(web["europe"]["value"])
            _ensure_branch_dict("europe")
            seed["europe"].update({
                "value": eu_val,
                "source": "web-serp",
                "is_model_level": bool(web["europe"].get("is_model_level"))
            })
            constraints.setdefault("europe", {}).setdefault("exact", {})[start_year] = eu_val
            plaus["source_note"] = "EU seed derived from web (window-first)."

        if w_val is None and web.get("world"):
            w_val = int(web["world"]["value"])
            _ensure_branch_dict("world")
            seed["world"].update({
                "value": w_val,
                "source": "web-serp",
                "is_model_level": bool(web["world"].get("is_model_level"))
            })
            constraints.setdefault("world", {}).setdefault("exact", {})[start_year] = min(w_val, WORLD_MAX_CAP)

    # ---- If World still missing but EU exists → EU→World range prior
    if w_val is None and eu_val is not None:
        (lo_share, hi_share), diag = infer_eu_share_bounds(db_eu, mk, start_year)
        world_min = max(eu_val, int(math.ceil(eu_val / max(hi_share, 1e-6))))
        world_max = int(min(WORLD_MAX_CAP, math.floor(eu_val / max(lo_share, 1e-6))))
        constraints["world"]["range"] = {start_year: (world_min, world_max)}
        seed["eu_share_prior"] = {
            "low": lo_share, "high": hi_share,
            "rank": diag["rank"], "presence_years": diag["presence_years"]
        }

    return seed, constraints, plaus


# ------------------ OpenAI call --------------------
def build_messages(car_model: str, target_gen_label: str, launch_year: int, seed: dict, constraints: dict) -> List[Dict[str,str]]:
    seed_text = json.dumps(seed, ensure_ascii=False, indent=2)
    cons_text = json.dumps(constraints, ensure_ascii=False, indent=2)
    user_prompt = (
        f"Model: {car_model}\n"
        f"Generation label (optional): {target_gen_label}\n"
        f"Generation window (inclusive): {seed['generation_window']}\n"
        f"Starting year (launch): {launch_year}\n\n"
        f"Seed & history (THIS GENERATION ONLY):\n{seed_text}\n\n"
        f"HARD RULES:\n{cons_text}\n"
        f"- Generation-only forecast; zero outside window.\n"
        f"- Apply exact/range constraints at start year.\n"
        f"- Ensure Europe ≤ World each year.\n\n"
        f"Task: Estimate annual unit sales from {launch_year} through {DEADLINE_YEAR} for this generation only. "
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
    # ICOR fields retained for compatibility; plausibility attached in diag
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
        # Plausibility summary
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
    if rows.empty: return {"supported_flag": False, "match_type": "no_model_match", "matched_row": None}
    return {"supported_flag": True, "match_type": "model_present_diff_gen_no_year_map", "matched_row": {"model": rows.iloc[0][cols["model"]] }}

# ------------------ Summary print ------------------
def print_summary(data, seed, constraints, gen_basis, icor_status, autodetect_note=""):
    print("\n=== Generation-Specific Car Sales Estimates ===")
    print(f"Model: {data.get('model','N/A')} | Generation: {seed.get('generation','')}  [basis: {gen_basis}]")
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
    generation = input("Enter the generation tag (blank to auto): ").strip()  # optional cosmetic
    year = int(input("Enter the starting year you'd like to view from (we will model from launch): ").strip())
    if year < 1990 or year > DEADLINE_YEAR:
        print(f"Year must be between 1990 and {DEADLINE_YEAR}.", file=sys.stderr); sys.exit(1)
    return {"car_model": model, "generation": generation, "start_year": year}

def serp_seed(model: str, gen: str, year: int) -> Dict[str, Any]:
    blobs_all: List[Dict[str,str]] = []
    for q in build_search_queries(model, gen, year):
        res = serp_search(q)
        if "error" not in res: blobs_all.extend(pull_text_blobs(res))
        time.sleep(0.35)
    eu = best_number_for_region(blobs_all, "europe", model, gen, year)
    w  = best_number_for_region(blobs_all, "world",  model, gen, year)
    return {"model": model, "gen": gen, "year": year, "europe": eu, "world": w}

def main():
    user = ask_user_inputs()

    # Local DBs from CWD (Streamlit page sets cwd=data/)
    db_eu = load_local_database_eu(os.getcwd())
    db_world = load_local_database_world(os.getcwd())

    # === WINDOW-FIRST DETECTION ===
    window, basis, gen_guess, mk, eu_hist, w_hist = detect_window_then_collect_local(
        db_eu, db_world, user["car_model"], user["start_year"]
    )
    launch_year, window_end = window
    display_model = user["car_model"]  # what user typed (also used in filenames/UI)

    # Build constraints based on the chosen window and local history
    seed, constraints, plaus = build_constraints_from_window(
        window, display_model, db_eu, db_world, mk, eu_hist, w_hist
    )
    seed["generation_window_basis"] = basis
    if gen_guess:
        seed["generation"] = gen_guess  # optional label for UI

    # === CALL MODEL ON LAUNCH YEAR ===
    messages = build_messages(display_model, seed.get("generation",""), launch_year, seed, constraints)
    data = call_openai(messages)

    # Enforce & smooth
    data = enforce_constraints_and_zero(data, constraints, launch_year)
    if APPLY_SMOOTHING:
        data = smooth_lifecycle(data, launch_year, set(constraints.get("zero_years", [])))

    # Ensure full coverage LAUNCH->DEADLINE
    have = {r.get("year") for r in data.get("yearly_estimates", [])}
    for y in range(launch_year, DEADLINE_YEAR+1):
        if y not in have:
            data.setdefault("yearly_estimates", []).append({
                "year": y, "world_sales_units": 0, "europe_sales_units": 0,
                "rationale": "Filled to cover from generation window."
            })
    data["yearly_estimates"] = sorted(data["yearly_estimates"], key=lambda r: r["year"])
    data["start_year"] = launch_year; data["end_year"] = DEADLINE_YEAR

    # Fleet & ICOR
    here = os.path.dirname(os.path.abspath(__file__))
    icor_txt_path = os.path.join(os.getcwd(), "icor_supported_models.txt")
    if not os.path.exists(icor_txt_path):
        icor_txt_path = os.path.join(here, "icor_supported_models.txt")
    icor_map = parse_icor_supported_txt(icor_txt_path)
    icor_status = check_icor_support(icor_map, None, display_model, seed.get("generation",""), launch_year)
    icor_status["plausibility"] = plaus

    # Save (relative to DATA_DIR/CWD)
    safe_model = display_model.replace(" ", "_")
    safe_gen = (normalize_generation(seed.get("generation","GEN")) or "GEN").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"sales_estimates_{safe_model}_{safe_gen}_{launch_year}_{MODEL_NAME}_{timestamp}"
    
    # Actually save files first
    csv_path = save_csv(data, base)
    xlsx_path = save_excel(data, fleet, seed, constraints, icor_status, base)
    
    # Then compute absolute paths
    csv_abs  = os.path.abspath(csv_path)
    xlsx_abs = os.path.abspath(xlsx_path)
    
    print_summary(data, seed, constraints, basis, icor_status, autodetect_note="")
    print(f"\nSaved CSV: {os.path.basename(csv_path)}")
    print(f"Saved Excel: {os.path.basename(xlsx_path)}")
    
    # Also print absolute paths
    print(f"Saved CSV (abs): {csv_abs}")
    print(f"Saved Excel (abs): {xlsx_abs}")
    
    print(f"\nNotes: {plaus.get('source_note','')}")
    print("- Window-first: local Top100 used where available; web seeds only for gaps.")
    print(f"- User requested view-from year: {user['start_year']}; modeled from LAUNCH year: {launch_year}.")



if __name__ == "__main__":
    main()

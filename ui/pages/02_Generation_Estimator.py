#!/usr/bin/env python3
"""
car_sales_estimator_gen_autodetect_alias_icor.py  (EU + World local data)

Enhancements:
  • Loads EU Top100_YYYY.txt AND World Top100_World_YYYY.txt.
  • Uses World local data for the same purposes as EU:
      - model matching, gen window confirmation/extension, start-year seed, history, presence/rank diagnostics.
  • SerpAPI remains the fallback when the model is not present in either EU or World local data
    (and still used for gen autodetect if preferred).
  • Alias inference for the detected generation considers EU+World labels inside the window.
  • World start-year handling:
      - If World local seed exists → set exact World constraint.
      - Else if EU exists → build EU→World range (same logic as before).
  • Outputs unchanged (CSV, Excel with Estimates/Fleet_Repairs/Summary/Seeds_Constraints).

Run (standalone):
  export OPENAI_API_KEY=...; export SERPAPI_KEY=...
  python script2.py
"""

from __future__ import annotations

import os
import re
import csv
import json
import glob
import time
import math
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import difflib
import requests
import pandas as pd
from tabulate import tabulate
from collections import Counter

# ───────── Secrets from environment only ─────────
import os, sys

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")

if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY not found in environment. OpenAI calls will fail.", file=sys.stderr)
if not SERPAPI_KEY:
    print("⚠️  SERPAPI_KEY not found in environment. Web seeding may be disabled.", file=sys.stderr)


# ---------------- Config ----------------
MODEL_NAME            = "gpt-4o-mini"   # set to an available model
DEADLINE_YEAR         = 2035
APPLY_SMOOTHING       = True
SEARCH_TIMEOUT        = 20
MAX_RESULTS_TO_SCAN   = 10
DEBUG                 = False
WORLD_MAX_CAP         = 3_000_000

# Derived metrics
DECAY_RATE            = 0.0556      # 5.56% annual decay starting N+2
REPAIR_RATE           = 0.021       # 2.1% of fleet

# Generation detection policy
PREFER_WEB_FOR_GEN    = True        # web (SerpAPI) is authoritative when gen is blank
MAX_FUTURE_GAP        = 1           # accept Top100 nearest ≥ if ≤ 1 year ahead (fallback only)

SYSTEM_INSTRUCTIONS = (
    "You are an automotive market analyst. Use provided seed data and constraints as anchors. "
    "This forecast is generation-specific; do not mix other generations. "
    "If world is provided as a range, ensure your start-year world is inside it. "
    "For years outside the generation window, output 0 for both Europe and World. "
    "Explain key assumptions briefly."
)

# ---------------- Helpers: normalization ----------------
def normalize_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[/\-]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_generation(g: Optional[str]) -> str:
    if not g:
        return ""
    g = str(g)
    g = g.lower()
    g = re.sub(r"[\(\)\[\]{}]", " ", g)
    g = re.sub(r"mk\s*", "mk", g)      # "Mk 6" -> "mk6"
    g = re.sub(r"\s+", " ", g).strip()
    return g

# ---------------- Local DB loaders (EU & World) ----------------
def _load_json_array(path: str) -> Optional[List[dict]]:
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception as e:
        if DEBUG: print(f"[WARN] Failed to read {path}: {e}")
        return None

def load_local_database_eu(folder: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    EU db: db_eu[normalized_model][year] = {
        'model': str, 'generation': str|None, 'units_europe': int, 'estimated': bool
    }
    Reads Top100_*.txt
    """
    db: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(folder, "Top100_*.txt"))):
        if "_World_" in os.path.basename(path):
            continue
        m = re.search(r"(\d{4})", os.path.basename(path))
        if not m:
            continue
        year = int(m.group(1))
        arr = _load_json_array(path)
        if not arr: continue
        for rec in arr:
            model = rec.get("model") or rec.get("name") or ""
            if not model:
                continue
            gen = rec.get("generation")
            est = bool(rec.get("estimated", False))
            units = rec.get("units_sold", rec.get("projected_units_2025"))
            if not isinstance(units, int):
                try:
                    units = int(units)
                except Exception:
                    continue
            key = normalize_name(model)
            db.setdefault(key, {})
            db[key][year] = {
                "model": model,
                "generation": gen,
                "units_europe": int(units),
                "estimated": est,
            }
    return db

def load_local_database_world(folder: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    World db: db_world[normalized_model][year] = {
        'model': str, 'generation': str|None, 'units_world': int, 'estimated': bool
    }
    Reads Top100_World_*.txt
    """
    db: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(folder, "Top100_World_*.txt"))):
        m = re.search(r"(\d{4})", os.path.basename(path))
        if not m:
            continue
        year = int(m.group(1))
        arr = _load_json_array(path)
        if not arr: continue
        for rec in arr:
            model = rec.get("model") or rec.get("name") or ""
            if not model:
                continue
            gen = rec.get("generation")
            est = bool(rec.get("estimated", False))
            units = rec.get("units_sold", rec.get("projected_units_2025"))
            if not isinstance(units, int):
                try:
                    units = int(units)
                except Exception:
                    continue
            key = normalize_name(model)
            db.setdefault(key, {})
            db[key][year] = {
                "model": model,
                "generation": gen,
                "units_world": int(units),
                "estimated": est,
            }
    return db

# ---------------- Shared lookups ----------------
def find_best_model_key(*dbs: Dict[str, Dict[int, Dict[str, Any]]], user_model: str) -> Optional[str]:
    key = normalize_name(user_model)
    for db in dbs:
        if key in db:
            return key
    # fuzzy across union of keys
    all_keys = sorted(set(k for db in dbs for k in db.keys()))
    cand = difflib.get_close_matches(key, all_keys, n=1, cutoff=0.7)
    return cand[0] if cand else None

def model_presence_years(db: Dict[str, Dict[int, Dict[str, Any]]], model_key: str) -> int:
    return len(db.get(model_key, {}))

def get_year_rank_units(db: Dict[str, Dict[int, Dict[str, Any]]], year: int, units_field: str) -> Dict[str, int]:
    rows = []
    for mk, by_year in db.items():
        if year in by_year:
            rows.append((mk, by_year[year][units_field]))
    rows.sort(key=lambda t: t[1], reverse=True)
    return {mk: i + 1 for i, (mk, _) in enumerate(rows)}

def get_year_rank(db: Dict[str, Dict[int, Dict[str, Any]]], model_key: str, year: int, units_field: str) -> Optional[int]:
    rankmap = get_year_rank_units(db, year, units_field)
    return rankmap.get(model_key)

# ---------------- ICOR TXT parser (map style) ----------------
def parse_icor_supported_txt(path: str) -> Optional[Dict[str, Dict[int, str]]]:
    if not os.path.exists(path):
        return None
    try:
        text = open(path, "r", encoding="utf-8").read()
    except Exception as e:
        print(f"[WARN] Cannot read ICOR TXT at {path}: {e}")
        return None

    i, n = 0, len(text)
    mapping: Dict[str, Dict[int, str]] = {}
    while i < n:
        m = re.search(r'"([^"]+)"\s*:', text[i:])
        if not m:
            break
        model = m.group(1)
        start = i + m.end()
        b = re.search(r'\{', text[start:])
        if not b:
            break
        brace_start = start + b.start()
        depth, j = 0, brace_start
        while j < n:
            if text[j] == '{': depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0: break
            j += 1
        if j >= n: break
        inner = text[brace_start+1:j]
        for y, gen in re.findall(r'(\d{4})\s*:\s*"([^"]+)"', inner):
            year = int(y)
            mapping.setdefault(normalize_name(model), {})[year] = gen
        i = j + 1
    return mapping

# ---------------- Simple list parser (fallback) ----------------
def parse_icor_supported_list(path: str) -> Optional[set]:
    if not os.path.exists(path):
        return None
    try:
        text = open(path, "r", encoding="utf-8").read()
    except Exception:
        return None
    models = set()
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        mpart = ln.split("\t", 1)[0]
        for m in re.split(r"[+/&]", mpart):
            models.add(normalize_name(m.strip()))
    return models or None

# ---------------- Windows from local / ICOR ----------------
def generation_window_from_icor(icor_map: Optional[Dict[str, Dict[int, str]]],
                                model_name: str,
                                target_gen: str,
                                horizon_start: int,
                                horizon_end: int) -> Optional[Tuple[int, int]]:
    if not icor_map:
        return None
    key = normalize_name(model_name)
    if key not in icor_map:
        return None
    pairs = sorted(icor_map[key].items())  # [(year, gen), ...]
    tg = normalize_generation(target_gen)
    start, end = None, None
    for idx, (y, g) in enumerate(pairs):
        if normalize_generation(g) == tg:
            start = y
            end = horizon_end
            for yy, gg in pairs[idx+1:]:
                if normalize_generation(gg) != tg:
                    end = yy - 1
                    break
            break
    if start is None:
        return None
    return (max(start, horizon_start), min(end, horizon_end))

def window_from_local_history(hist: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    if not hist:
        return None
    years = [h["year"] for h in hist]
    return (min(years), max(years))

def combine_windows(win_local: Optional[Tuple[int,int]],
                    win_icor: Optional[Tuple[int,int]],
                    default_from: int,
                    default_to: int) -> Tuple[Tuple[int,int], str]:
    if win_local and win_icor:
        lo = min(win_local[0], win_icor[0]); hi = max(win_local[1], win_icor[1])
        return ((max(lo, default_from), min(hi, default_to))), "union(local,icor_txt)"
    if win_local:
        return ((max(win_local[0], default_from), min(win_local[1], default_to))), "local_top100"
    if win_icor:
        return ((max(win_icor[0], default_from), min(win_icor[1], default_to))), "icor_txt"
    return ((default_from, min(default_from + 8, default_to))), "default_8yr"

# ---------------- Alias inference (EU+World window) ----------------
def infer_local_gen_alias(db_eu, db_world, user_model: str, detected_gen: str, gen_window: Tuple[int, int]) -> Optional[str]:
    mk = find_best_model_key(db_eu, db_world, user_model=user_model)
    if not mk:
        return None
    lo, hi = gen_window
    det_norm = normalize_generation(detected_gen)
    labels = []
    for db, field in ((db_eu, "generation"), (db_world, "generation")):
        by_year = db.get(mk, {})
        for y, rec in by_year.items():
            if lo <= y <= hi:
                g = rec.get(field)
                if g:
                    labels.append(normalize_generation(g))
    labels = [g for g in labels if g]
    if not labels:
        return None
    c = Counter(labels)
    if det_norm in c:
        return None
    top_label, top_count = c.most_common(1)[0]
    total = sum(c.values())
    if len(c) == 1 or (top_count / total) >= 0.70:
        return top_label
    return None

# ---------------- Local seed/history for a specific generation (+alias) ----------------
def local_seed_for_generation(
    db_eu: Dict[str, Dict[int, Dict[str, Any]]],
    db_world: Dict[str, Dict[int, Dict[str, Any]]],
    user_model: str,
    target_gen: str,
    start_year: int,
    accepted_alias: Optional[str] = None,
    window: Optional[Tuple[int,int]] = None
) -> Dict[str, Any]:
    out = {
        "found_model": False, "model_key": None, "display_model": None,
        "eu": None, "world": None,
        "history_europe": [], "history_world": [],
        "rank_eu": None, "rank_world": None,
        "presence_years_eu": 0, "presence_years_world": 0,
    }
    mk = find_best_model_key(db_eu, db_world, user_model=user_model)
    if not mk:
        return out
    out["found_model"] = True
    out["model_key"] = mk
    accepted = {normalize_generation(target_gen)}
    if accepted_alias:
        accepted.add(normalize_generation(accepted_alias))
    def _hist_one(db: Dict[str, Dict[int, Dict[str, Any]]], units_field: str) -> List[Dict[str, Any]]:
        hist = []
        by_year = db.get(mk, {})
        for y, rec in by_year.items():
            if window and not (window[0] <= y <= window[1]):
                continue
            if normalize_generation(rec.get("generation")) in accepted:
                hist.append({"year": y, "units": int(rec.get(units_field, 0)), "estimated": bool(rec.get("estimated", False))})
        hist.sort(key=lambda r: r["year"])
        return hist
    eu_hist = _hist_one(db_eu, "units_europe")
    w_hist  = _hist_one(db_world, "units_world")
    out["history_europe"] = eu_hist
    out["history_world"]  = w_hist
    disp = None
    for db in (db_eu, db_world):
        if mk in db and start_year in db[mk]:
            disp = db[mk][start_year]["model"]; break
    if disp is None:
        for db in (db_eu, db_world):
            if mk in db and db[mk]:
                last_year = sorted(db[mk].keys())[-1]
                disp = db[mk][last_year]["model"]; break
    out["display_model"] = disp
    out["presence_years_eu"] = model_presence_years(db_eu, mk)
    out["presence_years_world"] = model_presence_years(db_world, mk)
    out["rank_eu"] = get_year_rank(db_eu, mk, start_year, "units_europe")
    out["rank_world"] = get_year_rank(db_world, mk, start_year, "units_world")
    eu_seed = None
    if mk in db_eu and start_year in db_eu[mk]:
        if normalize_generation(db_eu[mk][start_year].get("generation")) in accepted:
            eu_seed = int(db_eu[mk][start_year]["units_europe"])
    world_seed = None
    if mk in db_world and start_year in db_world[mk]:
        if normalize_generation(db_world[mk][start_year].get("generation")) in accepted:
            world_seed = int(db_world[mk][start_year]["units_world"])
    if eu_seed is not None:
        out["eu"] = {"value": eu_seed, "source": "local-db (alias OK)" if accepted_alias else "local-db", "is_model_level": True}
    if world_seed is not None:
        out["world"] = {"value": world_seed, "source": "local-db (alias OK)" if accepted_alias else "local-db", "is_model_level": True}
    return out

# ---------------- SerpAPI helpers ----------------
SERP_ENDPOINT = "https://serpapi.com/search.json"
CURRENCY_TOKENS = ["€", "$", "£"]
PRICE_WORDS = ["price", "prices", "pricing", "msrp", "starting", "from", "cost", "lease", "per month"]
SALES_WORDS = ["sales", "sold", "units", "registrations", "deliveries", "volume", "shipments"]
UNIT_WORDS = r"(?:cars|units|vehicles|registrations|sales|deliveries)"

NUM_PATTERN = re.compile(
    rf"(?<![A-Z0-9])(\d{{1,3}}(?:[,\.\s]\d{{3}})+|\d{{4,}})(?=\s*(?:{UNIT_WORDS})?\b)",
    flags=re.IGNORECASE
)

DOMAIN_WEIGHTS = {
    "wikipedia.org": 5,
    "en.wikipedia.org": 6,
    "www.wikipedia.org": 5,
    "ford.co.uk": 4, "www.ford.com": 5,
    "caranddriver.com": 4, "autocar.co.uk": 4, "autoexpress.co.uk": 4, "topgear.com": 4,
    "motor1.com": 3, "parkers.co.uk": 3, "whatcar.com": 3,
}

def looks_like_price(text: str) -> bool:
    t = text.lower()
    if any(sym in text for sym in CURRENCY_TOKENS): return True
    if any(w in t for w in PRICE_WORDS): return True
    if re.search(r"[€$£]\s?\d{1,3}(?:[,\.\s]\d{3})+", text): return True
    return False

def has_sales_context(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in SALES_WORDS)

def extract_candidate_numbers(text: str) -> List[int]:
    def _to_int(num_str: str) -> Optional[int]:
        s = re.sub(r"[,\.\s]", "", num_str)
        return int(s) if s.isdigit() else None
    candidates = []
    for m in NUM_PATTERN.finditer(text or ""):
        val = _to_int(m.group(1))
        if val is not None:
            candidates.append(val)
    return [n for n in candidates if 10 <= n <= 5_000_000]

def serp_search(query: str) -> Dict[str, Any]:
    params = {"engine": "google", "q": query, "api_key": SERPAPI_KEY, "hl": "en", "num": "10", "safe": "active"}
    try:
        r = requests.get(SERP_ENDPOINT, params=params, timeout=SEARCH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "query": query}

def pull_text_blobs(serp_json: Dict[str, Any]) -> List[Dict[str, str]]:
    blobs = []
    def add_blob(source: str, text: str, url: str):
        if not text: return
        if looks_like_price(text):
            if DEBUG: print("\n[DROP price-like]\n", text, "\nURL:", url)
            return
        blobs.append({"source": source, "text": text, "url": url})
    for key in ("answer_box", "knowledge_graph"):
        box = serp_json.get(key)
        if isinstance(box, dict):
            for _, v in box.items():
                if isinstance(v, str):
                    add_blob(key, v, serp_json.get("search_metadata", {}).get("google_url", ""))
    for item in serp_json.get("organic_results", [])[:MAX_RESULTS_TO_SCAN]:
        title = item.get("title") or ""
        snippet = item.get("snippet") or ""
        url = item.get("link") or ""
        combined = f"{title}\n{snippet}".strip()
        add_blob("organic", combined, url)
    return blobs

def contains_phrase(text: str, phrase: str) -> bool:
    tokens = [re.escape(t) for t in phrase.split()]
    pattern = r"\b" + r"\s+".join(tokens) + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None

def build_generation_aliases(gen: str) -> List[str]:
    g = gen.strip()
    aliases = {g}
    gnorm = normalize_generation(g)
    if gnorm.startswith("mk"):
        n = gnorm[2:]
        aliases.update({f"mk{n}", f"mk {n}", f"mark {n}"})
        words = {
            "1":"first","2":"second","3":"third","4":"fourth","5":"fifth","6":"sixth","7":"seventh",
            "8":"eighth","9":"ninth","10":"tenth","11":"eleventh","12":"twelfth"
        }
        if n in words:
            aliases.add(f"{words[n]} generation")
            aliases.add(f"{words[n]}-gen")
            aliases.add(f"{n}th generation")
    return sorted(a for a in aliases if a)

def build_search_queries(model: str, gen: str, year: int) -> List[str]:
    gen_aliases = build_generation_aliases(gen) if gen else []
    gen_part = " OR ".join(f'"{a}"' for a in gen_aliases) if gen_aliases else ""
    base = [
        f'"{model}" {year} sales worldwide -price -prices -MSRP -€ -$',
        f'"{model}" {year} global sales -price -prices -MSRP -€ -$',
        f'"{model}" {year} worldwide deliveries -price -prices -MSRP -€ -$',
        f'"{model}" {year} registrations Europe -price -prices -MSRP -€ -$',
        f'"{model}" {year} sales Europe -price -prices -MSRP -€ -$',
    ]
    if gen_part:
        base = [
            f'"{model}" ({gen_part}) {year} sales worldwide -price -prices -MSRP -€ -$',
            f'"{model}" ({gen_part}) {year} global sales -price -prices -MSRP -€ -$',
            f'"{model}" ({gen_part}) {year} worldwide deliveries -price -prices -MSRP -€ -$',
            f'"{model}" ({gen_part}) {year} registrations Europe -price -prices -MSRP -€ -$',
            f'"{model}" ({gen_part}) {year} sales Europe -price -prices -MSRP -€ -$',
        ]
    preferred = [
        f'site:carsalesbase.com "{model}" {year} sales' if not gen_part else f'site:carsalesbase.com "{model}" ({gen_part}) {year} sales',
        f'site:acea.auto "{model}" {year} registrations' if not gen_part else f'site:acea.auto "{model}" ({gen_part}) {year} registrations',
        f'site:marklines.com "{model}" {year} sales' if not gen_part else f'site:marklines.com "{model}" ({gen_part}) {year} sales',
    ]
    return base + preferred

def best_number_for_region(blobs: List[Dict[str, str]], region: str, model: str, gen: str, year: int) -> Optional[Dict[str, Any]]:
    region_aliases = {"europe": ["europe", "eu", "efta", "uk", "european"], "world": ["world", "global", "worldwide"]}
    aliases = region_aliases["europe"] if region == "europe" else region_aliases["world"]
    y = str(year)
    gen_aliases = build_generation_aliases(gen) if gen else []
    best, best_score = None, -1e9
    for b in blobs:
        text = b.get("text") or ""
        t = text.lower()
        if not has_sales_context(t): continue
        if region == "world" and not any(a in t for a in aliases): continue
        if gen and not any(a.lower() in t for a in gen_aliases): continue
        nums = extract_candidate_numbers(text)
        if not nums: continue
        is_model = contains_phrase(text + " " + (b.get("url") or ""), model)
        has_gen = (not gen) or any(a.lower() in t for a in gen_aliases)
        region_bonus = 2 if any(a in t for a in aliases) else 0
        model_bonus  = 3 if is_model else -2
        gen_bonus    = 2 if has_gen else -3
        year_bonus   = 1 if y in t else 0
        price_pen    = -3 if looks_like_price(text) else 0
        candidate = max(nums)
        domain = ""
        try:
            domain = re.sub(r"^https?://(www\.)?", "", (b.get("url") or "")).split("/")[0].lower()
        except Exception:
            pass
        domain_w = DOMAIN_WEIGHTS.get(domain, 0)
        score = (region_bonus + model_bonus + gen_bonus + year_bonus + price_pen + domain_w) * 10 + math.log10(candidate + 1)
        if score > best_score:
            best_score = score
            best = {"value": candidate, "url": b.get("url") or "", "snippet": text, "is_model_level": is_model and has_gen}
    return best

def serp_seed(model: str, gen: str, year: int) -> Dict[str, Any]:
    if not SERPAPI_KEY:
        return {"model": model, "gen": gen, "year": year, "europe": None, "world": None, "queries": [], "error": "SERPAPI_KEY missing"}
    queries = build_search_queries(model, gen, year)
    blobs_all: List[Dict[str, str]] = []
    for q in queries:
        res = serp_search(q)
        if "error" in res:
            if DEBUG: print("[SERP error]", res["error"])
            continue
        blobs_all.extend(pull_text_blobs(res))
        time.sleep(0.35)
    eu = best_number_for_region(blobs_all, "europe", model, gen, year)
    w  = best_number_for_region(blobs_all, "world",  model, gen, year)
    return {"model": model, "gen": gen, "year": year, "europe": eu, "world": w, "queries": queries}

# ----- Generation autodetect via web (when blank) & local fallbacks -----
ROMAN_MAP = {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10,"xi":11,"xii":12}
ORDINAL_WORDS = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10,"eleventh":11,"twelfth":12}
GEN_PATTERNS = [
    re.compile(r'\b(?:mk|mark)\s?([ivx\d]{1,3})\b', re.I),
    re.compile(r'\bgen(?:eration)?\s*(?:no\.?\s*)?([ivx\d]{1,3})\b', re.I),
]
ORDINAL_PATTERN = re.compile(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+generation\b', re.I)
CODENAME_PATTERN = re.compile(r'\b(chassis|platform|internal|code(?:name)?)\s*(?:is|:)?\s*([A-Z]{1,2}\d{2,3})\b')
YEAR_RANGE_PATTERN = re.compile(r'(20\d{2})\s?(?:–|-|to)\s?(20\d{2})')
SINCE_PATTERN = re.compile(r'(?:since|from)\s+(20\d{2})')

def roman_or_digit_to_int(s: str) -> Optional[int]:
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    return ROMAN_MAP.get(s)

def detect_generation_via_web(model: str, year: int) -> Tuple[Optional[str], Optional[Tuple[int,int]], Optional[Dict[str,Any]]]:
    queries = [
        f'"{model}" {year} generation',
        f'"{model}" {year} mk',
        f'"{model}" generation timeline',
        f'site:wikipedia.org "{model}" generation',
        f'"{model}" {year} facelift generation',
    ]
    best = None
    best_score = -1e9
    best_window = None
    best_blob = None
    for q in queries:
        res = serp_search(q)
        if "error" in res:
            continue
        blobs = pull_text_blobs(res)
        for b in blobs:
            text = (b.get("text") or "") + " " + (b.get("url") or "")
            t = text.lower()
            if not contains_phrase(text, model):
                continue
            gen_label = None
            for pat in GEN_PATTERNS:
                m = pat.search(text)
                if m:
                    n = roman_or_digit_to_int(m.group(1))
                    if n:
                        gen_label = f"Mk{n}"
                        break
            if not gen_label:
                m = ORDINAL_PATTERN.search(text)
                if m:
                    n = ORDINAL_WORDS.get(m.group(1).lower())
                    if n:
                        gen_label = f"Mk{n}"
            if not gen_label:
                m = CODENAME_PATTERN.search(text)
                if m:
                    gen_label = m.group(2)
            if not gen_label:
                continue
            win = None
            m = YEAR_RANGE_PATTERN.search(text)
            if m:
                y1, y2 = int(m.group(1)), int(m.group(2))
                if 1990 <= y1 <= 2100 and 1990 <= y2 <= 2100 and y1 <= y2:
                    win = (y1, y2)
            else:
                m2 = SINCE_PATTERN.search(text)
                if m2:
                    y1 = int(m2.group(1))
                    if 1990 <= y1 <= 2100:
                        win = (y1, DEADLINE_YEAR)
            domain = ""
            try:
                domain = re.sub(r"^https?://(www\.)?", "", (b.get("url") or "")).split("/")[0].lower()
            except Exception:
                pass
            dom_w = DOMAIN_WEIGHTS.get(domain, 0)
            year_hit = 1 if str(year) in t else 0
            score = 10*dom_w + 5*year_hit
            if win: score += 8
            if "wikipedia.org" in domain: score += 5
            if score > best_score:
                best_score = score
                best = gen_label
                best_window = win
                best_blob = b
        time.sleep(0.35)
    return best, best_window, best_blob

# ---------------- Dynamic EU-share prior (model-level from EU Top100) ----------------
def infer_eu_share_bounds(db_eu: Dict[str, Dict[int, Dict[str, Any]]],
                          model_key: Optional[str],
                          start_year: int) -> Tuple[Tuple[float, float], Dict[str, Any]]:
    diag = {"basis": "presence+rank", "rank": None, "presence_years": 0, "bands": None}
    if not model_key:
        diag["bands"] = "unknown_model"
        return (0.01, 0.25), diag
    presence = model_presence_years(db_eu, model_key)
    rank = get_year_rank(db_eu, model_key, start_year, "units_europe")
    diag["rank"] = rank; diag["presence_years"] = presence
    if rank is None:
        if presence >= 5: lo, hi = 0.08, 0.60; diag["bands"] = "freq_present_no_rank"
        elif presence >= 1: lo, hi = 0.04, 0.40; diag["bands"] = "sporadic_present_no_rank"
        else: lo, hi = 0.01, 0.25; diag["bands"] = "never_present"
        return (lo, hi), diag
    if rank <= 10:   lo, hi = 0.25, 0.85; diag["bands"] = "rank<=10"
    elif rank <= 30: lo, hi = 0.15, 0.65; diag["bands"] = "rank<=30"
    elif rank <= 60: lo, hi = 0.08, 0.50; diag["bands"] = "rank<=60"
    else:            lo, hi = 0.03, 0.35; diag["bands"] = "rank<=100"
    return (lo, hi), diag

# ---------------- Build constraints (gen-specific) ----------------
def build_constraints(start_year: int, display_model: str, target_gen: str,
                      gen_window: Tuple[int,int],
                      db_eu: Dict[str, Dict[int, Dict[str, Any]]],
                      local: Dict[str, Any],
                      web: Optional[Dict[str, Any]]) -> Tuple[dict, dict]:
    seed_for_prompt = {
        "model": display_model,
        "generation": target_gen,
        "generation_window": {"start": gen_window[0], "end": gen_window[1]},
        "year": start_year,
        "europe": None,
        "world": None,
        "history_europe": local.get("history_europe", []),
        "history_world": local.get("history_world", []),
        "notes": "Local Top100 EU/World used for start-year seeds when available (accepting local alias). "
                 "If World seed missing, dynamic EU→World prior constrains World. "
                 "Forecast is generation-specific; zero outside the window."
    }
    constraints = {"world": {}, "europe": {}, "zero_years": []}

    zero_years = list(range(1990, gen_window[0])) + list(range(gen_window[1] + 1, DEADLINE_YEAR + 1))
    constraints["zero_years"] = [y for y in zero_years if y >= start_year]

    eu_val = None
    if local.get("eu"):
        eu_val = int(local["eu"]["value"])
        seed_for_prompt["europe"] = {"value": eu_val, "source": local["eu"]["source"], "is_model_level": True}
        constraints["europe"]["exact"] = {start_year: eu_val}

    world_val = None
    if local.get("world"):
        world_val = int(local["world"]["value"])
        seed_for_prompt["world"] = {"value": world_val, "source": local["world"]["source"], "is_model_level": True}
        constraints["world"]["exact"] = {start_year: min(world_val, WORLD_MAX_CAP)}

    world_range = None
    if world_val is None and eu_val is not None:
        (lo_share, hi_share), diag = infer_eu_share_bounds(db_eu, local.get("model_key"), start_year)
        world_min = max(eu_val, int(math.ceil(eu_val / max(hi_share, 1e-6))))
        world_max = int(min(WORLD_MAX_CAP, math.floor(eu_val / max(lo_share, 1e-6))))
        world_range = (world_min, world_max)
        seed_for_prompt["eu_share_prior"] = {"low": lo_share, "high": hi_share,
                                             "rank": diag["rank"], "presence_years": diag["presence_years"]}
        constraints["world"]["range"] = {start_year: world_range}

    # Web seeds only if local has nothing
    if (eu_val is None and world_val is None
        and not seed_for_prompt["history_europe"] and not seed_for_prompt["history_world"]):
        if web and web.get("europe") and web["europe"].get("is_model_level"):
            eu_web = int(web["europe"]["value"])
            seed_for_prompt["europe"] = {"value": eu_web, "source": "web-serp", "is_model_level": True}
            constraints.setdefault("europe", {}).setdefault("exact", {})[start_year] = eu_web
        if web and web.get("world") and web["world"].get("is_model_level"):
            w_web = int(web["world"]["value"])
            seed_for_prompt["world"] = {"value": w_web, "source": "web-serp", "is_model_level": True}
            constraints.setdefault("world", {}).setdefault("exact", {})[start_year] = min(w_web, WORLD_MAX_CAP)
        elif web and web.get("world") and world_range:
            seed_for_prompt["world"] = {"value": int(web["world"]["value"]), "source": "web-serp (bounded by EU prior)", "is_model_level": bool(web["world"].get("is_model_level"))}

    return seed_for_prompt, constraints

# ---------------- GPT interaction ----------------
def build_messages(car_model: str, target_gen: str, start_year: int, seed_for_prompt: dict, constraints: dict) -> List[Dict[str, str]]:
    seed_text = json.dumps(seed_for_prompt, ensure_ascii=False, indent=2)
    constraints_text = json.dumps(constraints, ensure_ascii=False, indent=2)
    user_prompt = (
        f"Model: {car_model}\n"
        f"Generation: {target_gen}\n"
        f"Generation window (inclusive): {seed_for_prompt['generation_window']}\n"
        f"Starting year for forecast: {start_year}\n\n"
        f"Seed & history (THIS GENERATION ONLY):\n{seed_text}\n\n"
        f"HARD RULES:\n{constraints_text}\n"
        f"- This is a GENERATION-ONLY forecast. Do not include sales from other generations.\n"
        f"- For any year outside the generation window, set both Europe and World to 0.\n"
        f"- If 'exact' is provided for Europe/World at start-year, set that exact number.\n"
        f"- If a World (min,max) range is provided, your start-year World must lie within it.\n"
        f"- Ensure Europe ≤ World each year.\n\n"
        f"Task: Estimate annual unit sales from {start_year} through {DEADLINE_YEAR} for this generation only, "
        f"covering World total and Europe (EU+EFTA+UK). Return ONLY one JSON object with fields: "
        f"model, generation_or_trim_context, start_year, end_year, assumptions[], methodology_summary, confidence, "
        f"yearly_estimates:[{{year:int, world_sales_units:int, europe_sales_units:int, rationale:string}}], notes."
    )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_prompt},
    ]

def call_openai(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing; cannot call OpenAI.")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        response_format={"type": "json_object"}
    )
    content = completion.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if "```" in content:
            chunk = content.split("```json")[-1].split("```")[0]
            return json.loads(chunk)
        raise RuntimeError("Model did not return valid JSON.")

# ---------------- Enforcement, zeroing & smoothing ----------------
def enforce_constraints_and_zero(data: dict, constraints: dict, start_year: int) -> dict:
    rows = data.setdefault("yearly_estimates", [])
    row = next((r for r in rows if r.get("year") == start_year), None)
    if row is None:
        row = {"year": start_year, "world_sales_units": 0, "europe_sales_units": 0, "rationale": ""}
        rows.append(row)
    def clamp(val, lo, hi): return max(lo, min(hi, val))
    ce = constraints.get("europe", {})
    if "exact" in ce:
        row["europe_sales_units"] = int(ce["exact"][start_year])
        row["rationale"] = (row.get("rationale","") + " Europe start fixed by local generation seed.").strip()
    cw = constraints.get("world", {})
    if "exact" in cw:
        row["world_sales_units"] = int(cw["exact"][start_year])
        row["rationale"] = (row.get("rationale","") + " World start fixed by local/web seed.").strip()
    elif "range" in cw:
        lo, hi = cw["range"][start_year]
        lo = max(lo, row["europe_sales_units"])
        hi = max(hi, lo)
        cur = int(row.get("world_sales_units", 0))
        row["world_sales_units"] = clamp(cur, lo, hi)
    if row["europe_sales_units"] > row["world_sales_units"]:
        row["world_sales_units"] = row["europe_sales_units"]
    needed = set(range(start_year, DEADLINE_YEAR + 1))
    have = {r.get("year") for r in rows}
    for y in sorted(needed - have):
        rows.append({"year": y, "world_sales_units": 0, "europe_sales_units": 0, "rationale": "Filled by client."})
    rows.sort(key=lambda r: r["year"])
    zero_years = set(constraints.get("zero_years", []))
    for r in rows:
        if r["year"] in zero_years:
            r["world_sales_units"] = 0
            r["europe_sales_units"] = 0
            r["rationale"] = (r.get("rationale","") + " Outside generation window -> zero.").strip()
    return data

def smooth_lifecycle(data: dict, start_year: int, zero_years: set):
    rows = sorted(data.get("yearly_estimates", []), key=lambda r: r["year"])
    if not rows: return data
    def smooth_series(series: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        y0, v0 = series[0]
        out = [(y0, max(0, v0))]
        for i in range(1, len(series)):
            y, suggested = series[i]
            if y in zero_years:
                out.append((y, 0)); continue
            _, v_prev = out[-1]
            years_since = y - y0
            if years_since == 1: growth = 0.35
            elif 2 <= years_since <= 3: growth = 0.12
            elif 4 <= years_since <= 5: growth = 0.05
            else: growth = -0.08
            proposed = int(round(v_prev * (1 + growth)))
            blended = int(max(0, 0.6 * proposed + 0.4 * suggested))
            out.append((y, blended))
        return out
    years = [r["year"] for r in rows]
    world_series  = [(r["year"], int(r.get("world_sales_units", 0))) for r in rows]
    europe_series = [(r["year"], int(r.get("europe_sales_units", 0))) for r in rows]
    world_sm  = smooth_series(world_series)
    europe_sm = smooth_series(europe_series)
    for i, _ in enumerate(years):
        w = max(0, world_sm[i][1])
        e = max(0, europe_sm[i][1])
        if e > w: e = w
        rows[i]["world_sales_units"]  = min(w, WORLD_MAX_CAP)
        rows[i]["europe_sales_units"] = e
    data["yearly_estimates"] = rows
    return data

# ---------------- Fleet & Repairs ----------------
def compute_fleet_and_repairs(rows: List[Dict[str, Any]],
                              decay_rate: float = DECAY_RATE,
                              repair_rate: float = REPAIR_RATE) -> List[Dict[str, Any]]:
    years = [r["year"] for r in rows]
    world_sales = [int(r.get("world_sales_units", 0)) for r in rows]
    europe_sales = [int(r.get("europe_sales_units", 0)) for r in rows]
    n = len(years)
    world_fleet = [0.0] * n
    europe_fleet = [0.0] * n
    for i in range(n):
        cohort_year = years[i]
        ws = world_sales[i]
        es = europe_sales[i]
        for j in range(i, n):
            age = years[j] - cohort_year
            surv = 1.0 if age <= 1 else (1.0 - decay_rate) ** (age - 1)
            world_fleet[j] += ws * surv
            europe_fleet[j] += es * surv
    results = []
    for i in range(n):
        wf = world_fleet[i]
        ef = europe_fleet[i]
        results.append({
            "year": years[i],
            "world_fleet": int(round(wf)),
            "europe_fleet": int(round(ef)),
            "world_repairs": int(round(wf * repair_rate)),
            "europe_repairs": int(round(ef * repair_rate)),
        })
    return results

# ---------------- Output: CSV & Excel ----------------
def save_csv(data: Dict[str, Any], base_name: str) -> str:
    rows = data.get("yearly_estimates", [])
    csv_name = f"{base_name}.csv"
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "world_sales_units", "europe_sales_units"])
        for r in rows:
            writer.writerow([r.get("year"), r.get("world_sales_units"), r.get("europe_sales_units")])
    return csv_name

def save_excel(data: Dict[str, Any],
               fleet_repair: List[Dict[str, Any]],
               seed_for_prompt: Dict[str, Any],
               constraints: Dict[str, Any],
               icor_status: Dict[str, Any],
               base_name: str) -> str:
    estimates = pd.DataFrame([
        {"Year": r["year"],
         "World_Sales": int(r.get("world_sales_units", 0)),
         "Europe_Sales": int(r.get("europe_sales_units", 0)),
         "Rationale": r.get("rationale", "")}
        for r in data.get("yearly_estimates", [])
    ])
    fr = pd.DataFrame(fleet_repair)[["year","world_fleet","europe_fleet","world_repairs","europe_repairs"]] \
           .rename(columns={
               "year":"Year",
               "world_fleet":"World_Fleet",
               "europe_fleet":"Europe_Fleet",
               "world_repairs":"World_Windshield_Repairs",
               "europe_repairs":"Europe_Windshield_Repairs"
           })
    estimates_merged = estimates.merge(fr, on="Year", how="left")
    estimates_merged["ICOR_Supported"] = icor_status.get("supported_flag")
    estimates_merged["ICOR_Match_Type"] = icor_status.get("match_type")
    gen_win = seed_for_prompt.get("generation_window", {})
    gstart, gend = gen_win.get("start"), gen_win.get("end")
    estimates_merged["Gen_Active"] = estimates_merged["Year"].apply(lambda y: bool(gstart is not None and gend is not None and gstart <= y <= gend))
    estimates_merged = estimates_merged[
        ["Year",
         "World_Sales","Europe_Sales",
         "World_Fleet","Europe_Fleet",
         "World_Windshield_Repairs","Europe_Windshield_Repairs",
         "Gen_Active","ICOR_Supported","ICOR_Match_Type",
         "Rationale"]
    ]
    summary_rows = {
        "Model": [data.get("model")],
        "Generation_Input_or_Detected": [seed_for_prompt.get("generation")],
        "Generation_Window_Start": [gstart],
        "Generation_Window_End": [gend],
        "Generation_Window_Basis": [seed_for_prompt.get("generation_window_basis")],
        "Generation_Alias_Used": [seed_for_prompt.get("generation_alias_used")],
        "Generation_Context_From_Model": [data.get("generation_or_trim_context")],
        "Start_Year": [data.get("start_year")],
        "End_Year": [data.get("end_year")],
        "Confidence": [data.get("confidence")],
        "ICOR_Supported": [icor_status.get("supported_flag")],
        "ICOR_Match_Type": [icor_status.get("match_type")],
        "ICOR_Matched_Row": [json.dumps(icor_status.get("matched_row"), ensure_ascii=False)],
    }
    summary = pd.DataFrame(summary_rows)
    seeds_constraints = pd.DataFrame({
        "Seed_or_Constraint": ["seed", "constraints"],
        "JSON": [json.dumps(seed_for_prompt, indent=2), json.dumps(constraints, indent=2)],
    })
    xlsx_name = f"{base_name}.xlsx"
    with pd.ExcelWriter(xlsx_name, engine="openpyxl") as writer:
        estimates_merged.to_excel(writer, sheet_name="Estimates", index=False)
        fr.to_excel(writer, sheet_name="Fleet_Repairs", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        seeds_constraints.to_excel(writer, sheet_name="Seeds_Constraints", index=False)
    return xlsx_name

# ---------------- ICOR support (by model+year; label may differ) ----------------
def load_icor_catalog_csv_json(path: str) -> Optional[pd.DataFrame]:
    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        elif path.lower().endswith((".json",)):
            return pd.read_json(path)
    except Exception as e:
        print(f"[WARN] Failed to read ICOR CSV/JSON at {path}: {e}")
    return None

def check_icor_support(icor_map: Optional[Dict[str, Dict[int, str]]],
                       icor_df: Optional[pd.DataFrame],
                       model_name: str,
                       generation_input: str,
                       start_year: int) -> Dict[str, Any]:
    if icor_map:
        norm_model = normalize_name(model_name)
        if norm_model in icor_map:
            years = sorted(icor_map[norm_model].keys())
            active_years = [y for y in years if y <= start_year]
            if active_years:
                y_active = max(active_years)
                icor_gen_raw  = icor_map[norm_model][y_active]
                icor_gen_norm = normalize_generation(icor_gen_raw)
                input_gen_norm = normalize_generation(generation_input)
                match_type = "by_year_exact_gen" if (input_gen_norm and input_gen_norm == icor_gen_norm) else "by_year_diff_gen_label"
                return {
                    "supported_flag": True,
                    "match_type": match_type,
                    "matched_row": {"model": model_name, "icor_gen_code": icor_gen_raw, "icor_gen_from_year": y_active}
                }
            else:
                return {
                    "supported_flag": False,
                    "match_type": "model_present_no_year_coverage",
                    "matched_row": {"model": model_name, "first_year_in_icor": years[0] if years else None}
                }
        else:
            return {"supported_flag": False, "match_type": "no_model_match", "matched_row": None}
    if icor_df is None or icor_df.empty:
        return {"supported_flag": "unknown", "match_type": "no_catalog", "matched_row": None}
    norm_target_model = normalize_name(model_name)
    norm_target_gen   = normalize_generation(generation_input)
    cols = {c.lower(): c for c in icor_df.columns}
    if "model" not in cols:
        return {"supported_flag": "unknown", "match_type": "no_model_column", "matched_row": None}
    gen_col = cols.get("generation")
    model_rows = icor_df[icor_df[cols["model"]].apply(lambda v: normalize_name(str(v)) == norm_target_model)]
    if model_rows.empty:
        return {"supported_flag": False, "match_type": "no_model_match", "matched_row": None}
    if gen_col and any(not str(g).strip() for g in model_rows[gen_col].fillna("")):
        row = model_rows.iloc[0]
        return {"supported_flag": True, "match_type": "all_gens", "matched_row": {"model": row[cols["model"]], "generation": ""}}
    if gen_col and norm_target_gen:
        for _, row in model_rows.iterrows():
            if normalize_generation(str(row[gen_col])) == norm_target_gen:
                return {"supported_flag": True, "match_type": "exact_gen", "matched_row": {"model": row[cols["model"]], "generation": row[gen_col]}}
    row = model_rows.iloc[0]
    return {"supported_flag": True, "match_type": "model_present_diff_gen_no_year_map", "matched_row": {"model": row[cols["model"]], "generation": (row[gen_col] if gen_col else "")}}

# ---------------- Print summary ----------------
def print_summary(data: Dict[str, Any], seed_for_prompt: Dict[str, Any], constraints: Dict[str, Any],
                  gen_basis: str, icor_status: Dict[str, Any], autodetect_note: str = ""):
    print("\n=== Generation-Specific Car Sales Estimates (Modelled) ===")
    print(f"Model: {data.get('model', 'N/A')}  |  Generation input/detected: {seed_for_prompt.get('generation')}  [basis: {gen_basis}]")
    if autodetect_note:
        print(f"Auto-detect note: {autodetect_note}")
    if data.get("generation_or_trim_context"):
        print(f"Model's own generation context: {data['generation_or_trim_context']}")
    gw = seed_for_prompt.get("generation_window", {})
    print(f"Generation window (inclusive): {gw.get('start')}–{gw.get('end')}")
    alias_used = seed_for_prompt.get("generation_alias_used")
    if alias_used:
        print(f"Using local Top100 alias '{alias_used}' as equivalent to detected generation for EU/World history/anchor.")
    print(f"Coverage (forecast): {data.get('start_year')}–{data.get('end_year')}")
    print(f"Confidence: {data.get('confidence', 'N/A')}")

    print("\nSeeds & history (this generation only):")
    if seed_for_prompt.get("europe"):
        e = seed_for_prompt["europe"]["value"]
        print(f"  Europe {seed_for_prompt['year']}: ~{e:,}  [{seed_for_prompt['europe']['source']}]")
    else:
        print("  Europe: (no start-year figure for this generation)")
    hist_eu = seed_for_prompt.get("history_europe") or []
    if hist_eu:
        tail = ", ".join(f"{h['year']}:{h['units']:,}{'*' if h.get('estimated') else ''}" for h in hist_eu[-8:])
        print(f"  Europe history (gen): {tail}")

    if seed_for_prompt.get("world"):
        w = seed_for_prompt["world"]["value"]
        print(f"  World {seed_for_prompt['year']}: ~{w:,}  [{seed_for_prompt['world']['source']}]")
    else:
        print("  World: (no start-year figure for this generation)")
    hist_w = seed_for_prompt.get("history_world") or []
    if hist_w:
        tail = ", ".join(f"{h['year']}:{h['units']:,}{'*' if h.get('estimated') else ''}" for h in hist_w[-8:])
        print(f"  World history (gen): {tail}")

    print("\nStart-year constraints:")
    print(json.dumps({k: v for k, v in constraints.items() if k in ("europe", "world")}, indent=2))
    if constraints.get("zero_years"):
        zy = f"{min(constraints['zero_years'])}..{max(constraints['zero_years'])}" if constraints["zero_years"] else "-"
        print(f"Zeroed years outside window: {zy}")

    rows = data.get("yearly_estimates", [])
    if rows:
        table = [[r.get("year"), r.get("world_sales_units"), r.get("europe_sales_units")] for r in rows]
        print("\nYearly estimates (units):")
        print(tabulate(table, headers=["Year", "World", "Europe"], tablefmt="github", numalign="right", stralign="right"))

    print(f"\nICOR support: {icor_status.get('supported_flag')} (match={icor_status.get('match_type')})")
    if icor_status.get("matched_row"):
        print(f"Matched ICOR row: {icor_status['matched_row']}")


# ---------------- Inputs, auto-deduction, main ----------------
def ask_user_inputs() -> Dict[str, Any]:
    """
    Standalone CLI inputs:
      line 1: car model (e.g., 'Ford Fiesta')
      line 2: generation tag (optional; e.g., 'Mk6')
      line 3: start year (e.g., 2013)
    """
    # If there are already three lines on stdin (from Streamlit wrapper), read them:
    if not sys.stdin.isatty():
        try:
            lines = []
            for _ in range(3):
                ln = sys.stdin.readline()
                if ln == "":
                    break
                lines.append(ln.rstrip("\n"))
            if len(lines) >= 2:
                model = (lines[0] or "").strip()
                generation = (lines[1] or "").strip()
                year_str = (lines[2] or "").strip() if len(lines) >= 3 else ""
                if model:
                    try:
                        year = int(year_str) if year_str else 2013
                    except Exception:
                        year = 2013
                    return {"car_model": model, "generation": generation, "start_year": year}
        except Exception:
            pass

    # Fallback to interactive prompt
    model = input("Enter the car model (e.g., 'Ford Fiesta'): ").strip()
    generation = input("Enter the generation tag (e.g., 'Mk6', 'Mk7', 'A5'; leave blank to auto-detect): ").strip()
    year_str = input("Enter the starting year (e.g., 2013): ").strip()

    if not model:
        print("You must provide a car model.", file=sys.stderr)
        sys.exit(1)
    try:
        year = int(year_str)
    except ValueError:
        print("Starting year must be an integer.", file=sys.stderr)
        sys.exit(1)
    if year < 1990 or year > DEADLINE_YEAR:
        print(f"Starting year must be between 1990 and {DEADLINE_YEAR}.", file=sys.stderr)
        sys.exit(1)
    return {"car_model": model, "generation": generation, "start_year": year}


def autodetect_generation(db_eu, db_world, icor_map, model, start_year, user_gen) -> Tuple[str, Tuple[int, int], str, str]:
    """
    Returns: (gen_label, window(start,end), basis, note)
    basis ∈ {"user_input","web_serp","local_top100","icor_txt","future_top100","default_8yr"}
    Uses EU+World local data for fallbacks/confirmation.
    """
    # If user already supplied a generation, respect it; build window from local+ICOR.
    if user_gen:
        gen_label = user_gen
        eu_hist = local_seed_for_generation(db_eu, db_world, model, gen_label, start_year).get("history_europe", [])
        w_hist  = local_seed_for_generation(db_eu, db_world, model, gen_label, start_year).get("history_world", [])
        win_local_eu = window_from_local_history(eu_hist)
        win_local_w  = window_from_local_history(w_hist)
        win_local = None
        if win_local_eu and win_local_w:
            win_local = (min(win_local_eu[0], win_local_w[0]), max(win_local_eu[1], win_local_w[1]))
        else:
            win_local = win_local_eu or win_local_w
        win_icor  = generation_window_from_icor(icor_map, model, gen_label, start_year, DEADLINE_YEAR)
        window, _ = combine_windows(win_local, win_icor, start_year, DEADLINE_YEAR)
        return gen_label, window, "user_input", "Generation provided by user."

    mk = find_best_model_key(db_eu, db_world, user_model=model)

    # Prefer web SERP when available
    web_gen, win_serp, _blob = detect_generation_via_web(model, start_year)
    if PREFER_WEB_FOR_GEN and web_gen:
        gen_label = web_gen
        # Confirm/extend with local EU+World
        det_norm = normalize_generation(gen_label)

        def _hist_for(db):
            hist = []
            if mk and mk in db:
                for y, rec in db[mk].items():
                    if normalize_generation(rec.get("generation")) == det_norm:
                        hist.append({"year": y, "units": rec.get("units_europe", rec.get("units_world", 0))})
            return hist

        eu_hist = _hist_for(db_eu)
        w_hist = _hist_for(db_world)
        win_local_eu = window_from_local_history(eu_hist)
        win_local_w = window_from_local_history(w_hist)
        win_local = None
        if win_local_eu and win_local_w:
            win_local = (min(win_local_eu[0], win_local_w[0]), max(win_local_eu[1], win_local_w[1]))
        else:
            win_local = win_local_eu or win_local_w
        win_icor_agree = generation_window_from_icor(icor_map, model, gen_label, start_year, DEADLINE_YEAR)

        base_start, base_end = (win_serp if win_serp else (start_year, min(start_year + 8, DEADLINE_YEAR)))
        lo, hi = base_start, base_end
        if win_local:
            lo, hi = min(lo, win_local[0]), max(hi, win_local[1])
        if win_icor_agree:
            lo, hi = min(lo, win_icor_agree[0]), max(hi, win_icor_agree[1])
        window = (max(lo, start_year), min(hi, DEADLINE_YEAR))

        note_bits = ["Detected from web SERP"]
        if win_serp:
            note_bits.append("window from snippet")
        if win_local:
            note_bits.append("extended with Top100 EU+World (agreed)")
        if win_icor_agree:
            note_bits.append("extended with ICOR (agreed)")
        return gen_label, window, "web_serp", "; ".join(note_bits) + "."

    # Local Top100 exact at start-year (EU first, then World)
    if mk:
        for db in (db_eu, db_world):
            if mk in db and start_year in db[mk] and db[mk][start_year].get("generation"):
                gen_label = db[mk][start_year]["generation"]
                eu_hist = local_seed_for_generation(db_eu, db_world, model, gen_label, start_year).get("history_europe", [])
                w_hist = local_seed_for_generation(db_eu, db_world, model, gen_label, start_year).get("history_world", [])
                win_local_eu = window_from_local_history(eu_hist)
                win_local_w = window_from_local_history(w_hist)
                win_local = None
                if win_local_eu and win_local_w:
                    win_local = (min(win_local_eu[0], win_local_w[0]), max(win_local_eu[1], win_local_w[1]))
                else:
                    win_local = win_local_eu or win_local_w
                win_icor = generation_window_from_icor(icor_map, model, gen_label, start_year, DEADLINE_YEAR)
                window, _ = combine_windows(win_local, win_icor, start_year, DEADLINE_YEAR)
                return gen_label, window, "local_top100", "Detected from Top100 at the start year."

    # Nearest previous year with gen
    if mk:
        years = sorted(set(list(db_eu.get(mk, {}).keys()) + list(db_world.get(mk, {}).keys())))
        prev_years = [y for y in years if y <= start_year and (
            (mk in db_eu and y in db_eu[mk] and db_eu[mk][y].get("generation")) or
            (mk in db_world and y in db_world[mk] and db_world[mk][y].get("generation"))
        )]
        if prev_years:
            y0 = prev_years[-1]
            gen_label = db_eu.get(mk, {}).get(y0, db_world.get(mk, {}).get(y0))["generation"]
            eu_hist = local_seed_for_generation(db_eu, db_world, model, gen_label, start_year).get("history_europe", [])
            w_hist = local_seed_for_generation(db_eu, db_world, model, gen_label, start_year).get("history_world", [])
            win_local_eu = window_from_local_history(eu_hist)
            win_local_w = window_from_local_history(w_hist)
            win_local = None
            if win_local_eu and win_local_w:
                win_local = (min(win_local_eu[0], win_local_w[0]), max(win_local_eu[1], win_local_w[1]))
            else:
                win_local = win_local_eu or win_local_w
            win_icor = generation_window_from_icor(icor_map, model, gen_label, start_year, DEADLINE_YEAR)
            window, _ = combine_windows(win_local, win_icor, start_year, DEADLINE_YEAR)
            return gen_label, window, "local_top100", f"Detected from Top100 nearest ≤ year ({y0})."

    # ICOR-only
    if icor_map:
        norm_model = normalize_name(model)
        if norm_model in icor_map:
            years = sorted(icor_map[norm_model].keys())
            yr = None
            for y in years:
                if y <= start_year:
                    yr = y
            if yr is None and years:
                yr = years[0]
            if yr is not None:
                gen_label = icor_map[norm_model][yr]
                win_icor = generation_window_from_icor(icor_map, model, gen_label, start_year, DEADLINE_YEAR)
                window, _ = combine_windows(None, win_icor, start_year, DEADLINE_YEAR)
                return gen_label, window, "icor_txt", f"Detected from ICOR map (year {yr})."

    # Nearest future (≤ 1y)
    if mk:
        years = sorted(set(list(db_eu.get(mk, {}).keys()) + list(db_world.get(mk, {}).keys())))
        next_years = [y for y in years if y >= start_year and (
            (mk in db_eu and y in db_eu[mk] and db_eu[mk][y].get("generation")) or
            (mk in db_world and y in db_world[mk] and db_world[mk][y].get("generation"))
        )]
        if next_years and (next_years[0] - start_year) <= MAX_FUTURE_GAP:
            y1 = next_years[0]
            gen_label = db_eu.get(mk, {}).get(y1, db_world.get(mk, {}).get(y1))["generation"]
            eu_hist = local_seed_for_generation(db_eu, db_world, model, gen_label, start_year).get("history_europe", [])
            w_hist = local_seed_for_generation(db_eu, db_world, model, gen_label, start_year).get("history_world", [])
            win_local_eu = window_from_local_history(eu_hist)
            win_local_w = window_from_local_history(w_hist)
            win_local = None
            if win_local_eu and win_local_w:
                win_local = (min(win_local_eu[0], win_local_w[0]), max(win_local_eu[1], win_local_w[1]))
            else:
                win_local = win_local_eu or win_local_w
            win_icor = generation_window_from_icor(icor_map, model, gen_label, start_year, DEADLINE_YEAR)
            window, _ = combine_windows(win_local, win_icor, start_year, DEADLINE_YEAR)
            return gen_label, window, "future_top100", f"Detected from Top100 nearest ≥ year ({y1})."

    # Default fallback
    return "GEN", (start_year, min(start_year + 8, DEADLINE_YEAR)), "default_8yr", "Fallback default 8-year window."


def main():
    user = ask_user_inputs()

    # Load local Top100 databases (EU and World) from CWD
    cwd = os.getcwd()
    db_eu = load_local_database_eu(cwd)
    db_world = load_local_database_world(cwd)

    # ICOR TXT: try CWD first, then script dir
    here = os.path.dirname(os.path.abspath(__file__))
    icor_txt_path = os.path.join(cwd, "icor_supported_models.txt")
    if not os.path.exists(icor_txt_path):
        icor_txt_path = os.path.join(here, "icor_supported_models.txt")

    # Preferred: map (year→gen)
    icor_map = parse_icor_supported_txt(icor_txt_path)

    # Fallback: simple list → tiny DF used by checker
    icor_df = None
    if icor_map is None:
        icor_simple = parse_icor_supported_list(icor_txt_path)
        if icor_simple:
            icor_df = pd.DataFrame({"model": sorted(icor_simple), "generation": [""] * len(icor_simple)})

    # Generation autodetect using EU+World local data (and web if preferred)
    gen_label, gen_window, gen_basis, autodetect_note = autodetect_generation(
        db_eu, db_world, icor_map, user["car_model"], user["start_year"], user["generation"]
    )

    # Infer local alias across EU+World inside window
    alias = infer_local_gen_alias(db_eu, db_world, user["car_model"], gen_label, gen_window)

    # Local seeds/history for the generation (accept alias)
    local = local_seed_for_generation(
        db_eu, db_world, user["car_model"], gen_label, user["start_year"],
        accepted_alias=alias, window=gen_window
    )

    # If the model is missing in BOTH EU and World local DBs, query web; else keep web only as diagnostic
    use_web = not (local.get("history_europe") or local.get("history_world") or local.get("eu") or local.get("world"))
    web = serp_seed(local.get("display_model") or user["car_model"], gen_label, user["start_year"]) if use_web else {}

    # Build prompt seeds & constraints
    seed_for_prompt, constraints = build_constraints(
        user["start_year"],
        local.get("display_model") or user["car_model"],
        gen_label, gen_window,
        db_eu, local, web
    )
    seed_for_prompt["generation_window_basis"] = gen_basis
    if alias:
        seed_for_prompt["generation_alias_used"] = alias

    # GPT call
    messages = build_messages(local.get("display_model") or user["car_model"], gen_label,
                              user["start_year"], seed_for_prompt, constraints)
    data = call_openai(messages)

    # Enforce constraints & zeroing
    data = enforce_constraints_and_zero(data, constraints, user["start_year"])

    # Optional smoothing
    if APPLY_SMOOTHING:
        zero_set = set(constraints.get("zero_years", []))
        data = smooth_lifecycle(data, user["start_year"], zero_set)

    # Ensure full horizon coverage
    needed = set(range(user["start_year"], DEADLINE_YEAR + 1))
    have = {r.get("year") for r in data.get("yearly_estimates", [])}
    for y in sorted(needed - have):
        data.setdefault("yearly_estimates", []).append(
            {"year": y, "world_sales_units": 0, "europe_sales_units": 0, "rationale": "Filled by client."}
        )
    data["yearly_estimates"] = sorted(data["yearly_estimates"], key=lambda r: r["year"])
    data["start_year"] = user["start_year"]
    data["end_year"] = DEADLINE_YEAR

    # Fleet & repairs
    fleet_repair = compute_fleet_and_repairs(data["yearly_estimates"], DECAY_RATE, REPAIR_RATE)

    # ICOR support check
    icor_status = check_icor_support(icor_map, icor_df,
                                     local.get("display_model") or user["car_model"],
                                     gen_label,
                                     user["start_year"])

    # Save outputs
    safe_model_name = (data.get("model") or local.get("display_model") or user["car_model"]).replace(" ", "_")
    safe_gen = normalize_generation(gen_label or "GEN").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"sales_estimates_{safe_model_name}_{safe_gen}_{user['start_year']}_{MODEL_NAME}_{timestamp}"

    csv_path = save_csv(data, base)
    xlsx_path = save_excel(data, fleet_repair, seed_for_prompt, constraints, icor_status, base)

    # Console summary
    print_summary(data, seed_for_prompt, constraints, gen_basis, icor_status, autodetect_note)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved Excel: {xlsx_path}")
    print("\nNotes:")
    print(f"- Fleet uses decay={DECAY_RATE*100:.2f}% starting N+2; Repairs are {REPAIR_RATE*100:.1f}% of fleet.")
    print("- Gen detection order: Web (SERP) → Top100 exact → Top100 ≤ → ICOR → (Top100 ≥ if ≤1y) → default.")
    print("- Within the detected window, a dominant local Top100 label can be used as an alias for EU/World seeding.\n")


if __name__ == "__main__":
    main()

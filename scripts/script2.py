#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation-specific sales estimator (EU + World)
- LOCAL-FIRST generation detection (EU+World); web only when local has no labels.
- Start of estimates = DETECTED GENERATION LAUNCH YEAR (not the user's view year).
- Web seeds get plausibility checks vs. prior-gen EU avg and model-level EU totals.
- Exports Estimates, Fleet_Repairs, Summary, Seeds_Constraints with plausibility fields.

This version adds:
- Digit-aware strict model matching (Q5 will not match Q3).
- Safer filenames (use the user's typed model).
- Estimation starts at generation launch year (window start), not the user's year.
- Canonical generation identity (mk ii / 2nd gen / second generation / II → 'gen2').
- Slash/alternative handling ('XV / XVI' → {'gen15','gen16'}).
- Contiguity expansion across unlabeled years to avoid collapsing the window.
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
PREFER_WEB_FOR_GEN  = True   # used only when local has no labels at all
MAX_FUTURE_GAP      = 1

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

# --- Robust Generation Normalizers -----------------
_ROMAN_MAP = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
_ORD_WORDS = {
    "first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,
    "seventh":7,"eighth":8,"ninth":9,"tenth":10,"eleventh":11,"twelfth":12,
    "thirteenth":13,"fourteenth":14,"fifteenth":15,"sixteenth":16,
    "seventeenth":17,"eighteenth":18,"nineteenth":19,"twentieth":20
}

def _roman_to_int_any(s: str) -> int | None:
    if not s: return None
    s = s.upper()
    if not all(ch in _ROMAN_MAP for ch in s): return None
    total, prev = 0, 0
    for ch in reversed(s):
        val = _ROMAN_MAP[ch]
        if val < prev: total -= val
        else: total += val; prev = val
    return total if 0 < total <= 3999 else None

def normalize_generation(s: str | None) -> str:
    if not s: return ""
    g = str(s)
    g = re.sub(r"\((?:\s*(?:c\.\s*)?\d{3,4}(?:\s*[–\-\/]\s*(?:\d{2,4})?)?\s*)\)\s*$", "", g)
    g = g.replace("—", "-").replace("–", "-").strip()
    g = re.sub(r"[/|]", " / ", g)
    g = re.sub(r"[\(\)\[\]\{\}]", " ", g)
    g = re.sub(r"\bmk\s*", "mk", g, flags=re.I)
    g = re.sub(r"\s+", " ", g).strip()
    return g

_GEN_NUMBER_PATTERNS = [
    re.compile(r"\b(?:mk|mark)\s*([ivxlcdm]+|\d{1,3})\b", re.I),
    re.compile(r"\bgen(?:eration)?\s*(?:no\.?\s*)?([ivxlcdm]+|\d{1,3})\b", re.I),
    re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)\b(?:\s+gen(?:eration)?)?", re.I),
    re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)?\b(?:\s+gen(?:eration)?)?", re.I),
    re.compile(r"\(\s*(\d{1,3})(?:st|nd|rd|th)\s*\)", re.I),
]

def _extract_all_gen_numbers_from_token(tok: str) -> list[int]:
    nums: list[int] = []
    for pat in _GEN_NUMBER_PATTERNS:
        for m in pat.finditer(tok):
            g = m.group(1) if m.groups() else None
            if g is None:
                word = m.group(0).strip().split()[0].lower()
                n = _ORD_WORDS.get(word)
                if n: nums.append(n)
                continue
            g = g.strip()
            if g.isdigit():
                nums.append(int(g))
            else:
                rn = _roman_to_int_any(g)
                if rn: nums.append(rn)
    return nums

def canonical_gen_set(label: str | None) -> set[str]:
    if not label: return set()
    s = normalize_generation(label)
    parts = re.split(r"\s*[\/|,&]\s*", s)
    nums: set[int] = set()
    for part in parts:
        stripped = part.strip()
        if re.fullmatch(r"[ivxlcdm]+", stripped, flags=re.I):
            rn = _roman_to_int_any(stripped)
            if rn: nums.add(rn)
        elif re.fullmatch(r"\d{1,3}", stripped):
            nums.add(int(stripped))
        for n in _extract_all_gen_numbers_from_token(stripped):
            nums.add(n)
    return {f"gen{n}" for n in sorted(nums)}

def same_generation(a: str | None, b: str | None) -> bool:
    if not a or not b: return False
    A, B = canonical_gen_set(a), canonical_gen_set(b)
    return bool(A and B and (A & B))

def canonical_gen_label(g: str | None) -> str:
    S = canonical_gen_set(g)
    if not S: return (normalize_generation(g) if g else "")
    def _n(x: str) -> int: return int(x.replace("gen",""))
    return sorted(S, key=_n)[0]

def build_generation_aliases(gen: str | None) -> List[str]:
    if not gen: return []
    can_set = canonical_gen_set(gen)
    if not can_set:
        return [gen]
    out: set[str] = set()
    ord_map = {1:"first",2:"second",3:"third",4:"fourth",5:"fifth",6:"sixth",
               7:"seventh",8:"eighth",9:"ninth",10:"tenth",11:"eleventh",12:"twelfth",
               13:"thirteenth",14:"fourteenth",15:"fifteenth",16:"sixteenth",
               17:"seventeenth",18:"eighteenth",19:"nineteenth",20:"twentieth"}
    for can in can_set:
        n = int(can.replace("gen",""))
        out.update({
            f"mk{n}", f"mk {n}", f"mark {n}",
            f"{n} gen", f"gen {n}", f"gen{n}",
            f"{n}th generation",
        })
        if n in ord_map:
            out.add(f"{ord_map[n]} generation")
        # crude roman converter up to 20
        roman_pairs = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
        rn, m = "", n
        while m >= 10: rn += "X"; m -= 10
        for val, sym in roman_pairs:
            while m >= val:
                rn += sym; m -= val
        if rn:
            out.update({rn, f"mk{rn.lower()}", f"mk {rn.lower()}", f"gen {rn.lower()}"})
    raw = normalize_generation(gen)
    if re.search(r"\b[A-Z]{1,3}\d{1,3}\b", raw, flags=re.I):
        out.add(raw)
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
        # guards
        is_country_only = bool(re.search(r"\b(uk|germany|france|italy|spain|poland|netherlands|sweden|norway)\b", t))
        is_monthly = bool(re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|q[1-4]|month|monthly)\b", t))
        has_year = str(year) in t
        # scoring
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

def serp_seed(model: str, gen: str, year: int) -> Dict[str, Any]:
    blobs_all: List[Dict[str,str]] = []
    for q in build_search_queries(model, gen, year):
        res = serp_search(q)
        if "error" not in res: blobs_all.extend(pull_text_blobs(res))
        time.sleep(0.35)
    eu = best_number_for_region(blobs_all, "europe", model, gen, year)
    w  = best_number_for_region(blobs_all, "world",  model, gen, year)
    return {"model": model, "gen": gen, "year": year, "europe": eu, "world": w}

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

# ------------------ Local seeds/history -------------
def infer_local_gen_alias(db_eu, db_world, user_model: str, detected_gen: str, gen_window: Tuple[int,int]) -> Optional[str]:
    mk = find_best_model_key(db_eu, db_world, user_model=user_model)
    if not mk: return None
    lo, hi = gen_window
    det_can = canonical_gen_label(detected_gen)
    labels = []
    for db in (db_eu, db_world):
        for y, rec in db.get(mk, {}).items():
            if lo <= y <= hi:
                g = rec.get("generation")
                if g: labels.append(canonical_gen_label(g))
    labels = [g for g in labels if g]
    if not labels: return None
    c = Counter(labels)
    if det_can in c: return None
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

    accepted = {canonical_gen_label(target_gen)}
    if accepted_alias: accepted.add(canonical_gen_label(accepted_alias))

    def _hist(db, units_field):
        hist = []
        for y, rec in db.get(mk, {}).items():
            if window and not (window[0] <= y <= window[1]): 
                continue
            g = rec.get("generation")
            # Accept unlabeled rows inside the computed window
            if g and canonical_gen_label(g) not in accepted:
                continue
            hist.append({"year": y, "units": int(rec.get(units_field,0)), "estimated": bool(rec.get("estimated",False))})
        return sorted(hist, key=lambda r: r["year"])

    eu_hist = _hist(db_eu, "units_europe"); w_hist = _hist(db_world, "units_world")
    out["history_europe"] = eu_hist; out["history_world"] = w_hist

    # display model
    disp = None
    for db in (db_eu, db_world):
        if mk in db and start_year in db[mk]:
            disp = db[mk][start_year]["model"]; break
    if disp is None:
        for db in (db_eu, db_world):
            if mk in db and db[mk]:
                disp = db[mk][sorted(db[mk].keys())[-1]]["model"]; break
    out["display_model"] = disp

    # seeds at start year (accept unlabeled rows inside window)
    if mk in db_eu and start_year in db_eu[mk]:
        g = db_eu[mk][start_year].get("generation")
        if (not g) or (canonical_gen_label(g) in accepted):
            out["eu"] = {"value": int(db_eu[mk][start_year]["units_europe"]), "source": "local-db", "is_model_level": True}
    if mk in db_world and start_year in db_world[mk]:
        g = db_world[mk][start_year].get("generation")
        if (not g) or (canonical_gen_label(g) in accepted):
            out["world"] = {"value": int(db_world[mk][start_year]["units_world"]), "source": "local-db", "is_model_level": True}
    return out

# ------------------ Generation detection ------------
def _combined_rec(db_eu, db_world, mk, y):
    """Return a combined record (if any) for model key mk at year y from EU/World DBs."""
    rec_eu = db_eu.get(mk, {}).get(y)
    rec_w  = db_world.get(mk, {}).get(y)
    if rec_eu and rec_w:
        if rec_eu.get("generation"): return rec_eu
        if rec_w.get("generation"):  return rec_w
        return rec_eu
    return rec_eu or rec_w

def _expand_window_by_contiguity(db_eu, db_world, mk, pivot_year, det_can):
    """
    Starting from pivot_year (where we saw this generation), walk backward/forward
    across years where the model exists and the generation is either:
      - same canonical label (== det_can), or
      - missing/empty (treated as same-gen),
    stopping when we hit a conflicting labeled generation.
    """
    years_all = sorted(set(list(db_eu.get(mk, {}).keys()) + list(db_world.get(mk, {}).keys())))
    if not years_all:
        return (pivot_year, pivot_year)

    # Backward
    lo = pivot_year
    y = pivot_year - 1
    while y in years_all:
        rec = _combined_rec(db_eu, db_world, mk, y)
        if not rec: break
        g = rec.get("generation")
        if g:
            can = canonical_gen_label(g)
            if can == det_can:
                lo = y
            else:
                break
        else:
            lo = y
        y -= 1

    # Forward
    hi = pivot_year
    y = pivot_year + 1
    while y in years_all:
        rec = _combined_rec(db_eu, db_world, mk, y)
        if not rec: break
        g = rec.get("generation")
        if g:
            can = canonical_gen_label(g)
            if can == det_can:
                hi = y
            else:
                break
        else:
            hi = y
        y += 1

    return (lo, hi)

def autodetect_generation(db_eu, db_world, _icor_map_unused, model, user_year, user_gen):
    """
    Generation detection (LOCAL FIRST; ICOR ignored).
    Returns 4-tuple: (gen_label: str, window:(start,end), basis: str, note: str)

    We DO NOT clip the detected window to the user's year.
    We expand across unlabeled contiguous years to avoid window collapse.
    """
    def _window_from_local_for_label(gen_label: str):
        det_can = canonical_gen_label(gen_label)
        mk = find_best_model_key(db_eu, db_world, user_model=model)
        years_same = []
        years_labeled = []
        if mk:
            for db in (db_eu, db_world):
                for y, rec in db.get(mk, {}).items():
                    g = rec.get("generation")
                    if g:
                        years_labeled.append(y)
                        if canonical_gen_label(g) == det_can:
                            years_same.append(y)

        if years_same:
            pivot = min(years_same)  # earliest labeled occurrence
            lo, hi = _expand_window_by_contiguity(db_eu, db_world, mk, pivot, det_can)
            return (lo, hi)
        if years_labeled:
            pivot = min(years_labeled, key=lambda y: abs(y - user_year))
            lo, hi = _expand_window_by_contiguity(db_eu, db_world, mk, pivot, det_can)
            return (lo, hi)
        return None

    def _clip_to_horizon(window):
        gs, ge = window
        return (max(1990, gs), min(ge, DEADLINE_YEAR))

    # 0) User-provided generation
    if user_gen:
        gen_label = user_gen
        win_local = _window_from_local_for_label(gen_label)
        if win_local:
            return gen_label, _clip_to_horizon(win_local), "user_input", "Generation provided by user."
        return gen_label, (max(1990, user_year-1), min(user_year+7, DEADLINE_YEAR)), "user_input_default", "User gen; default 8y window."

    mk = find_best_model_key(db_eu, db_world, user_model=model)

    # 1) Local Top100 at/near user_year
    if mk:
        # exact at user_year
        for db in (db_eu, db_world):
            if mk in db and user_year in db[mk] and db[mk][user_year].get("generation"):
                gen_label = db[mk][user_year]["generation"]
                win_local = _window_from_local_for_label(gen_label)
                if win_local:
                    return gen_label, _clip_to_horizon(win_local), "local_top100", "Detected from Top100 at the user year."
        # nearest previous ≤ user_year
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
                return gen_label, _clip_to_horizon(win_local), "local_top100", f"Detected from Top100 nearest ≤ year ({y0})."
        # nearest future (≤ 1y gap)
        next_years = [y for y in years if y >= user_year and (
            (mk in db_eu and y in db_eu[mk] and db_eu[mk][y].get("generation")) or
            (mk in db_world and y in db_world[mk] and db_world[mk][y].get("generation"))
        )]
        if next_years and (next_years[0] - user_year) <= MAX_FUTURE_GAP:
            y1 = next_years[0]
            gen_label = db_eu.get(mk, {}).get(y1, db_world.get(mk, {}).get(y1))["generation"]
            win_local = _window_from_local_for_label(gen_label)
            if win_local:
                return gen_label, _clip_to_horizon(win_local), "future_top100", f"Detected from Top100 nearest ≥ year ({y1})."

    # 2) Web SERP (only if local truly has no labels)
    web_gen, win_serp, _ = detect_generation_via_web(model, user_year)
    if web_gen:
        window = win_serp or (user_year, min(user_year+8, DEADLINE_YEAR))
        return web_gen, _clip_to_horizon(window), "web_serp", "Detected from web SERP."

    # 3) Fallback default
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
    det_can = canonical_gen_label(detected_gen)
    vals = []
    for y in range(start_year-1, max(1990, start_year - lookback_years) - 1, -1):
        rec = db_eu.get(mk, {}).get(y)
        if not rec: continue
        g = rec.get("generation")
        if g and canonical_gen_label(g) == det_can:  # same-gen -> skip
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
        "notes": "Local Top100 EU/World are authoritative for generation. "
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

    # If local misses seeds entirely → consider web
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

    # Plausibility check for EU seed (if present and came from web)
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

    # If World exact missing but EU exists → build EU→World range
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

def detect_generation_via_web(model: str, year: int) -> Tuple[Optional[str], Optional[Tuple[int,int]], Optional[Dict[str,Any]]]:
    # lightweight web gen detector (kept from earlier version)
    queries = [
        f'"{model}" {year} generation',
        f'"{model}" {year} mk',
        f'site:wikipedia.org "{model}" generation',
    ]
    best = None; best_score = -1e9; best_window=None; best_blob=None
    ROMAN = {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10,"xi":11,"xii":12}
    def roman_or_digit_to_int(s):
        s=s.lower().strip()
        return int(s) if s.isdigit() else ROMAN.get(s)
    GEN_PAT = [re.compile(r'\b(?:mk|mark)\s?([ivx\d]{1,3})\b', re.I),
               re.compile(r'\bgen(?:eration)?\s*(?:no\.?\s*)?([ivx\d]{1,3})\b', re.I)]
    ORDINAL = re.compile(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+generation\b', re.I)
    ORD_WORDS = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10,"eleventh":11,"twelfth":12}
    YEAR_RANGE = re.compile(r'(20\d{2})\s?(?:–|-|to)\s?(20\d{2})')
    SINCE = re.compile(r'(?:since|from)\s+(20\d{2})')
    for q in queries:
        res = serp_search(q)
        if "error" in res: continue
        for b in pull_text_blobs(res):
            text = (b.get("text") or "") + " " + (b.get("url") or "")
            if normalize_name(model) not in normalize_name(text): continue
            gen_label = None
            for pat in GEN_PAT:
                m = pat.search(text)
                if m:
                    n = roman_or_digit_to_int(m.group(1))
                    if n: gen_label=f"Mk{n}"; break
            if not gen_label:
                m = ORDINAL.search(text)
                if m:
                    n = ORD_WORDS.get(m.group(1).lower()); 
                    if n: gen_label=f"Mk{n}"
            win=None
            m = YEAR_RANGE.search(text)
            if m:
                y1,y2 = int(m.group(1)), int(m.group(2))
                if 1990<=y1<=2100 and y1<=y2<=2100: win=(y1,y2)
            else:
                m2 = SINCE.search(text)
                if m2:
                    y1 = int(m2.group(1)); 
                    if 1990<=y1<=2100: win=(y1, DEADLINE_YEAR)
            dom_bonus = 5 if "wikipedia.org" in text.lower() else 0
            yr_bonus  = 2 if str(year) in text else 0
            score = 10*dom_bonus + 5*yr_bonus + (8 if win else 0)
            if gen_label and score>best_score:
                best_score=score; best=gen_label; best_window=win; best_blob=b
        time.sleep(0.3)
    return best, best_window, best_blob

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

    # Gen detect (LOCAL FIRST) — expanded across unlabeled contiguous years
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

    # Guarantee rows cover every year from LAUNCH through DEADLINE
    all_years = {r["year"] for r in data.get("yearly_estimates", [])}
    for y in range(launch_year, DEADLINE_YEAR + 1):
        if y not in all_years:
            data.setdefault("yearly_estimates", []).append({
                "year": y, "world_sales_units": 0, "europe_sales_units": 0,
                "rationale": "Filled to cover from generation launch."
            })
    data["yearly_estimates"] = sorted(data["yearly_estimates"], key=lambda r: r["year"])

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
    print("- Local Top100 EU/World are authoritative for generation; web is fallback only.")
    print(f"- User requested view-from year: {user['start_year']}; modeled from LAUNCH year: {launch_year}.")

if __name__ == "__main__":
    main()

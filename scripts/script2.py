#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script2.py — Generation-specific sales estimator (EU + World)

Hard rule:
- The generation window comes ONLY from wikipedia_gen.
  If the module exposes detect_via_wikipedia(), we call it.
  Otherwise, we run wikipedia_gen.py as a CLI with --json and parse its output.

If wikipedia_gen fails, we exit with a clear error.

"""

from __future__ import annotations
import csv
import difflib
import glob
import json
import math
import os
import re
import sys
import time
import subprocess
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tabulate import tabulate

# --- repo root on sys.path so "scripts.*" works when run directly
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

# Try to import wikipedia_gen as a module; if the function isn't present,
# we’ll call the CLI as a fallback (bridge).
_wg_mod = None
_wg_path = None
try:
    import scripts.wikipedia_gen as _wg_mod  # type: ignore
except Exception:
    try:
        import wikipedia_gen as _wg_mod  # type: ignore
    except Exception:
        _wg_mod = None

if _wg_mod is not None:
    _wg_path = getattr(_wg_mod, "__file__", None)

def _detect_via_wikipedia_bridge(model: str, year: int, lang: str = "en") -> Tuple[str, Tuple[int,int], Dict[str,Any]]:
    """
    Bridge that prefers module API detect_via_wikipedia(); if missing,
    executes wikipedia_gen.py CLI with --json and parses the last JSON line.
    """
    # 1) Module API present?
    if _wg_mod is not None and hasattr(_wg_mod, "detect_via_wikipedia"):
        return _wg_mod.detect_via_wikipedia(model, year, lang=lang)  # type: ignore

    # 2) CLI fallback
    if not _wg_path or not os.path.exists(_wg_path):
        raise RuntimeError("Cannot locate wikipedia_gen.py to run as a CLI.")

    py = sys.executable or "python3"
    cmd = [py, _wg_path, "--year", str(year), "--lang", lang, "--json", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except Exception as e:
        raise RuntimeError(f"Failed to execute wikipedia_gen CLI: {e}")

    if proc.returncode != 0:
        raise RuntimeError(f"wikipedia_gen CLI returned {proc.returncode}:\nSTDERR:\n{proc.stderr.strip()}\nSTDOUT:\n{proc.stdout.strip()}")

    # Parse last JSON line from stdout
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    json_obj = None
    for ln in reversed(lines):
        # Heuristic: last line that looks like JSON
        if ln.startswith("{") and ln.endswith("}"):
            try:
                json_obj = json.loads(ln)
                break
            except Exception:
                continue
    if not json_obj:
        raise RuntimeError("wikipedia_gen CLI did not emit the expected JSON line (--json).")

    label = json_obj.get("label")
    start = json_obj.get("start")
    end   = json_obj.get("end")
    diag  = json_obj.get("diag", {})

    if not isinstance(label, str) or not isinstance(start, int) or not isinstance(end, int):
        raise RuntimeError(f"Invalid JSON payload from wikipedia_gen CLI: {json_obj}")

    return label, (start, end), diag

# ------------------ Secrets (env) ------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY not set", file=sys.stderr)

# ------------------ Config -------------------------
MODEL_NAME          = "gpt-5"
DEADLINE_YEAR       = 2035
APPLY_SMOOTHING     = True
WORLD_MAX_CAP       = 3_000_000
DECAY_RATE          = 0.0556
REPAIR_RATE         = 0.021

SYSTEM_INSTRUCTIONS = (
    "You are an automotive market analyst. Use provided seed data and constraints as anchors. "
    "This forecast is generation-specific; do not mix other generations. "
    "If world is provided as a range, ensure your start-year world is inside it. "
    "For years outside the generation window, output 0 for both Europe and World. "
    "Explain key assumptions briefly."
)

# ------------------ Small utils --------------------
def safe_slug(s: str, max_len: int = 80) -> str:
    s = str(s)
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^A-Za-z0-9\-_\. ]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    if len(s) > max_len:
        s = s[:max_len]
    return s or "untitled"

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

# ------------------ Local DB loaders ----------------
def _load_json_array(path: str) -> Optional[List[dict]]:
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return None

def load_local_database_eu(folder: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    db: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for path in sorted(glob.glob(os.path.join(folder, "Top100_*.txt"))):
        if "_World_" in os.path.basename(path): continue
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

# ------------------ Model key matching ----------------
def find_best_model_key(*dbs: Dict[str, Dict[int, Dict[str, Any]]], user_model: str) -> Optional[str]:
    key = normalize_name(user_model)
    for db in dbs:
        if key in db:
            return key
    all_keys = sorted(set(k for db in dbs for k in db.keys()))
    if not all_keys: return None
    def digit_tokens(s: str) -> set[str]:
        toks = s.split()
        return {t for t in toks if t.isdigit() or re.fullmatch(r"[a-z]+?\d+", t)}
    target_digits = digit_tokens(key)
    filtered = all_keys
    if target_digits:
        same = [k for k in all_keys if digit_tokens(k) == target_digits]
        if same: filtered = same
    best, best_r = None, -1.0
    for cand in filtered:
        r = difflib.SequenceMatcher(None, key, cand).ratio()
        if r > best_r:
            best, best_r = cand, r
    if best and best_r >= 0.88:
        return best
    return None

# ------------------ Seeding helpers ----------------
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
        if normalize_generation(rec.get("generation")) == det_norm:
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

# ------------------ Wikipedia-only generation window ----------------
def autodetect_generation(model: str, user_year: int, lang: str = "en") -> Tuple[str, Tuple[int,int], str, str]:
    """
    Window ONLY from wikipedia_gen (module function or CLI bridge).
    Returns (gen_label, (start,end_capped), basis, note).
    """
    try:
        label, (start, end), diag = _detect_via_wikipedia_bridge(model, user_year, lang=lang)
    except Exception as e:
        msg = f"[FATAL] wikipedia_gen failed for '{model}' @ {user_year}: {e}"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg) from e

    end_capped = 2035 if end == 9999 else min(end, 2035)
    start_capped = max(1990, start)
    basis = "wikipedia_only"
    note = "Generation window taken from wikipedia_gen (module or CLI). end==9999 capped to horizon."
    return label, (start_capped, end_capped), basis, note

# ------------------ Constraints builder ----------------
def build_constraints(start_year: int, display_model: str, target_gen: str,
                      gen_window: Tuple[int,int], db_eu, local, web) -> Tuple[dict, dict, dict]:
    seed = {
        "model": display_model,
        "generation": target_gen,
        "generation_window": {"start": gen_window[0], "end": gen_window[1]},
        "generation_window_basis": "wikipedia_only",
        "year": start_year,
        "europe": None,
        "world": None,
        "history_europe": local.get("history_europe", []),
        "history_world": local.get("history_world", []),
        "notes": "Window from wikipedia_gen; local Top100 EU/World used for seeds/history if present."
    }
    constraints = {"world": {}, "europe": {}, "zero_years": []}
    plaus = {"flag": False, "reason": "", "source_note": ""}

    zero_years = list(range(1990, gen_window[0])) + list(range(gen_window[1]+1, 2035+1))
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

    if world_val is None and eu_val is not None:
        (lo_share, hi_share), diag = infer_eu_share_bounds(db_eu, local.get("model_key"), start_year)
        world_min = max(eu_val, int(math.ceil(eu_val / max(hi_share, 1e-6))))
        world_max = int(min(WORLD_MAX_CAP, math.floor(eu_val / max(lo_share, 1e-6))))
        constraints["world"]["range"] = {start_year: (world_min, world_max)}
        seed["eu_share_prior"] = {"low": lo_share, "high": hi_share,
                                  "rank": diag["rank"], "presence_years": diag["presence_years"]}
    return seed, constraints, plaus

# ------------------ Local gen history & seeds ----------------
def local_seed_for_generation(db_eu, db_world, user_model, target_gen, start_year, accepted_alias=None, window=None) -> Dict[str,Any]:
    out = {"found_model": False, "model_key": None, "display_model": None,
           "eu": None, "world": None, "history_europe": [], "history_world": []}
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
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    csv_name = f"{base}.csv"
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["year","world_sales_units","europe_sales_units"])
        for r in data.get("yearly_estimates", []):
            w.writerow([r.get("year"), r.get("world_sales_units"), r.get("europe_sales_units")])
    return csv_name

def save_excel(data: Dict[str,Any], fleet_repair: List[Dict[str,Any]],
               seed: Dict[str,Any], constraints: Dict[str,Any],
               diag: Dict[str,Any], base: str) -> str:
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
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

    summary = pd.DataFrame({
        "Model": [data.get("model")],
        "Generation_Input_or_Detected": [seed.get("generation")],
        "Generation_Window_Start": [gstart],
        "Generation_Window_End": [gend],
        "Generation_Window_Basis": [seed.get("generation_window_basis")],
        "Generation_Context_From_Model": [data.get("generation_or_trim_context")],
        "Start_Year": [data.get("start_year")],
        "End_Year": [data.get("end_year")],
        "Confidence": [data.get("confidence")],
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

# ------------------ Summary print ------------------
def print_summary(data, seed, constraints, gen_basis, autodetect_note=""):
    print("\n=== Generation-Specific Car Sales Estimates ===")
    print(f"Model: {data.get('model','N/A')} | Generation: {seed.get('generation')}  [basis: {gen_basis}]")
    if autodetect_note: print(f"Note: {autodetect_note}")
    gw = seed.get("generation_window",{})
    print(f"Generation window (modeling): {gw.get('start')}–{gw.get('end')}")
    print(f"Coverage: {data.get('start_year')}–{data.get('end_year')}")
    print(f"Confidence: {data.get('confidence','N/A')}")
    rows = data.get("yearly_estimates",[])
    if rows:
        tbl = [[r["year"], r["world_sales_units"], r["europe_sales_units"]] for r in rows[:10]]
        print(tabulate(tbl, headers=["Year","World","Europe"], tablefmt="github", numalign="right"))

# ------------------ CLI inputs & main --------------
def ask_user_inputs() -> Dict[str,Any]:
    model = input("Enter the car model: ").strip()
    generation = input("Enter the generation tag (blank to auto): ").strip()  # display only
    year = int(input("Enter the starting year you'd like to view from (we will model from launch): ").strip())
    if year < 1990 or year > DEADLINE_YEAR:
        print(f"Year must be between 1990 and {DEADLINE_YEAR}.", file=sys.stderr); sys.exit(1)
    return {"car_model": model, "generation": generation, "start_year": year}

def main():
    user = ask_user_inputs()

    # Local DBs from current working directory
    db_eu = load_local_database_eu(os.getcwd())
    db_world = load_local_database_world(os.getcwd())

    # ---- Generation detection: WIKIPEDIA ONLY (API or CLI) ----
    gen_label, gen_window, gen_basis, autodetect_note = autodetect_generation(
        user["car_model"], user["start_year"], lang="en"
    )

    # Model from launch
    launch_year = max(1990, gen_window[0])

    # Local seeds/history for that generation at launch year
    local = local_seed_for_generation(
        db_eu, db_world, user["car_model"], gen_label, launch_year,
        accepted_alias=None, window=gen_window
    )

    # Build seeds & constraints at launch year
    seed, constraints, _ = build_constraints(
        launch_year, local.get("display_model") or user["car_model"],
        gen_label, gen_window, db_eu, local, web={}
    )

    # Call OpenAI
    messages = build_messages(local.get("display_model") or user["car_model"], gen_label, launch_year, seed, constraints)
    data = call_openai(messages)

    # Enforce, smooth, horizon fill
    data = enforce_constraints_and_zero(data, constraints, launch_year)
    if APPLY_SMOOTHING:
        data = smooth_lifecycle(data, launch_year, set(constraints.get("zero_years", [])))
    need = set(range(launch_year, DEADLINE_YEAR+1))
    have = {r.get("year") for r in data.get("yearly_estimates", [])}
    for y in sorted(need - have):
        data.setdefault("yearly_estimates", []).append({"year": y, "world_sales_units": 0, "europe_sales_units": 0, "rationale": "Filled by client."})
    data["yearly_estimates"] = sorted(data["yearly_estimates"], key=lambda r: r["year"])
    data["start_year"] = launch_year
    data["end_year"] = DEADLINE_YEAR
    data["model"] = local.get("display_model") or user["car_model"]

    # Fleet/repairs
    fleet = compute_fleet_and_repairs(data["yearly_estimates"], DECAY_RATE, REPAIR_RATE)

    # Save (safe filenames)
    out_dir = "."
    base = os.path.join(
        out_dir,
        f"sales_estimates_{safe_slug(user['car_model'])}_{safe_slug(gen_label)}_{gen_window[0]}_{MODEL_NAME}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    csv_path = save_csv(data, base)
    xlsx_path = save_excel(data, fleet, seed, constraints, {}, base)

    # Summary
    print_summary(data, seed, constraints, gen_basis, autodetect_note)
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved Excel: {xlsx_path}")
    print("\nNotes:")
    print("- Generation window comes strictly from wikipedia_gen (module or CLI).")

if __name__ == "__main__":
    main()

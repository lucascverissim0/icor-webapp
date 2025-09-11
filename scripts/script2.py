#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web-first generation detector + window finder + sales filler (EU + World)

Flow:
1) Ask user for model + reference year.
2) Web search (SerpAPI → Wikipedia first). Find which generation is active in that year.
3) From the same page text, parse the generation's launch year and end year.
4) If end year == 2025, estimate whether sales continue. If yes, extend window.
5) Build per-year estimates only for years the generation is active:
   - Use local DB numbers when available (authoritative).
   - Fill missing years with lifecycle growth/decay (configurable).
6) Save CSV + Excel (Estimates, Fleet_Repairs, Summary).

Local DB is used ONLY for sales numbers and the 2025 continuation heuristic.
"""

import os, re, json, glob, csv, time, math, sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd
from tabulate import tabulate
from collections import Counter

# ------------------ Secrets (env) ------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # not used in this version
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")
if not SERPAPI_KEY:
    print("⚠️  SERPAPI_KEY not set — web search will fail. Set SERPAPI_KEY in your environment.", file=sys.stderr)

# ------------------ Config -------------------------
DEADLINE_YEAR         = 2035
WORLD_MAX_CAP         = 3_000_000
DECAY_RATE            = 0.0556  # fleet decay for repairs sheet
REPAIR_RATE           = 0.021
SEARCH_TIMEOUT        = 20
MAX_RESULTS_TO_SCAN   = 10
USER_PROMPT_MIN_YEAR  = 1990

# Lifecycle (for filling years missing in local DB)
EARLY_GROWTH_YEARS    = 2
EARLY_GROWTH_RATE     = 0.12
MID_GROWTH_YEARS      = 2
MID_GROWTH_RATE       = 0.05
LATE_DECAY_RATE       = -0.08  # after early+mid years

# 2025 continuation heuristic
CONTINUE_SINCE_2025_MIN_AVG_EU  = 30000   # if avg EU last 2 local years >= this, assume continuation
CONTINUE_EXTENSION_YEARS        = 3       # extend window by this many years when continuing
CONTINUE_DECAY_PER_YEAR         = 0.12    # decay applied for extended years

# ------------------ Utilities ----------------------
def normalize_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[/\-]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def current_year() -> int:
    return datetime.now().year

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

# ------------------ Web search helpers -------------
SERP_ENDPOINT = "https://serpapi.com/search.json"

def serp_search(query: str) -> Dict[str, Any]:
    params = {"engine":"google","q":query,"api_key":SERPAPI_KEY,"hl":"en","num":"10","safe":"active"}
    try:
        r = requests.get(SERP_ENDPOINT, params=params, timeout=SEARCH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "query": query}

def pull_serp_links(serp_json: Dict[str, Any]) -> List[str]:
    links = []
    for item in serp_json.get("organic_results", [])[:MAX_RESULTS_TO_SCAN]:
        url = item.get("link")
        if url: links.append(url)
    return links

def fetch_text(url: str) -> Optional[str]:
    try:
        r = requests.get(url, timeout=SEARCH_TIMEOUT, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        html = r.text
        # crude HTML → text
        html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
        html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
        text = re.sub(r"(?is)<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return None

# ------------------ Generation parsing -------------
_ROMAN = {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10,"xi":11,"xii":12,"xiii":13,"xiv":14,"xv":15,"xvi":16,"xvii":17,"xviii":18,"xix":19,"xx":20}
_ORD  = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,"sixth":6,
         "seventh":7,"eighth":8,"ninth":9,"tenth":10,"eleventh":11,"twelfth":12,
         "thirteenth":13,"fourteenth":14,"fifteenth":15,"sixteenth":16,
         "seventeenth":17,"eighteenth":18,"nineteenth":19,"twentieth":20}

def roman_to_int(tok: str) -> Optional[int]:
    return _ROMAN.get(tok.lower())

def extract_year_ranges(text: str) -> List[Tuple[int, Optional[int], str]]:
    """
    Returns list of tuples: (start_year, end_year_or_None, nearby_label_text)
    We consider patterns like:
      "(Mk4; 2019–2025)", "(2019–present)", "Production 2019–2025", "Model years 2020–present"
    """
    ranges: List[Tuple[int, Optional[int], str]] = []
    # common separators: – — - to
    sep = r"(?:–|—|-|to)"
    # 'present' markers
    present = r"(?:present|ongoing|current)"
    pat1 = re.compile(rf"(?:Production|Produced|Model years|Years)\s*[:]?\s*(20\d{{2}})\s*{sep}\s*(20\d{{2}}|{present})", re.I)
    pat2 = re.compile(rf"\(\s*(?:Mk\s*[ivx\d]+;?\s*)?(20\d{{2}})\s*{sep}\s*(20\d{{2}}|{present})\s*\)", re.I)
    pat3 = re.compile(rf"\b(20\d{{2}})\s*{sep}\s*(20\d{{2}}|{present})\b", re.I)
    for pat in (pat1, pat2, pat3):
        for m in pat.finditer(text):
            y1 = int(m.group(1))
            raw2 = m.group(2).lower()
            y2 = None if re.fullmatch(present, raw2) else int(raw2)
            snippet = m.group(0)
            if 1990 <= y1 <= 2100 and (y2 is None or (1990 <= y2 <= 2100 and y2 >= y1)):
                ranges.append((y1, y2, snippet))
    return ranges

def extract_generation_labels(text: str) -> List[Tuple[str,str]]:
    """
    Find phrases that look like 'Fourth generation (Mk4; 2019–2025)',
    'Mk V (2016–2021)', 'Second generation (2013–present)', etc.
    Returns list of (label, raw_snippet).
    """
    labels: List[Tuple[str,str]] = []
    # e.g., "Fourth generation (Mk4; 2019–2025)" or "Fifth generation (2018–present)"
    patA = re.compile(r"((?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)\s+generation)\s*\((.*?)\)", re.I)
    # e.g., "Mk4 (2019–2025)" or "Mk V (2016–2021)"
    patB = re.compile(r"\b(Mk\s*(?:[ivx]+|\d{1,2}))\s*\((.*?)\)", re.I)
    # e.g., "Generation 4 (2019–2026)"
    patC = re.compile(r"\b(gen(?:eration)?\s*(?:no\.?\s*)?(?:[ivx]+|\d{1,2}))\s*\((.*?)\)", re.I)
    for pat in (patA, patB, patC):
        for m in pat.finditer(text):
            labels.append((m.group(1), m.group(0)))
    return labels

def generation_number_from_label(label: str) -> Optional[int]:
    t = label.lower()
    # mk/mark
    m = re.search(r"\b(?:mk|mark)\s*([ivx]+|\d{1,2})\b", t)
    if m:
        tok = m.group(1)
        return int(tok) if tok.isdigit() else roman_to_int(tok)
    # generation N
    m = re.search(r"\bgen(?:eration)?\s*(?:no\.?\s*)?([ivx]+|\d{1,2})\b", t)
    if m:
        tok = m.group(1)
        return int(tok) if tok.isdigit() else roman_to_int(tok)
    # ordinal word
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)\s+generation\b", t)
    if m:
        return _ORD.get(m.group(1))
    return None

def find_active_generation_from_wikipedia(model: str, ref_year: int) -> Tuple[Optional[str], Optional[Tuple[int,int]], Optional[str]]:
    """
    Search Wikipedia for the model and parse which generation was active in ref_year.
    Returns: (generation_label, (start, end_or_ongoing), source_url)
             If ongoing, end is set to None.
    """
    queries = [
        f'site:wikipedia.org "{model}" generation',
        f'site:wikipedia.org {model} mk generation',
        f'site:wikipedia.org {model} production years',
        f'site:wikipedia.org "{model}"',
    ]
    urls: List[str] = []
    for q in queries:
        res = serp_search(q)
        if "error" in res: continue
        urls.extend(pull_serp_links(res))
        time.sleep(0.2)

    seen = set()
    urls = [u for u in urls if u not in seen and not seen.add(u)]

    best = (None, None, None)  # label, (start,end/None), url
    best_score = -1

    for url in urls:
        if "wikipedia.org" not in url: continue
        text = fetch_text(url)
        if not text: continue

        # 1) Pull labeled generation blocks and ranges nearby
        labels = extract_generation_labels(text)
        ranges = extract_year_ranges(text)

        # score blocks that contain a range matching ref_year
        # tie-breaker: label with explicit generation number
        for (label, raw) in labels:
            # search for the closest range near this label occurrence
            # simple approach: use all ranges; choose those covering ref_year
            for (y1, y2, snip) in ranges:
                y_end = y2 if y2 is not None else current_year()
                if y1 <= ref_year <= y_end:
                    # prefer matching where raw+snip appear close (crude check)
                    near = 1 if snip in raw or label in snip else 0
                    has_num = 1 if generation_number_from_label(label) else 0
                    ongoing_bonus = 1 if y2 is None else 0
                    score = 10*near + 5*has_num + 3*ongoing_bonus
                    if score > best_score:
                        best_score = score
                        best = (label, (y1, (y2 if y2 is not None else None)), url)

        # 2) If nothing matched, consider bare ranges without labels
        if best_score < 0 and ranges:
            for (y1, y2, _snip) in ranges:
                y_end = y2 if y2 is not None else current_year()
                if y1 <= ref_year <= y_end:
                    score = 1 if y2 is None else 0
                    if score > best_score:
                        best_score = score
                        best = (None, (y1, (y2 if y2 is not None else None)), url)

    return best

# ------------------ Continuation heuristic ---------
def should_continue_after_2025(model_key: str, db_eu: Dict[str,Dict[int,Dict[str,Any]]]) -> bool:
    """
    If the average EU sales of the last two available years (<=2025) >= threshold,
    assume the generation will continue beyond 2025 for a few years.
    """
    if model_key not in db_eu: return False
    ys = sorted([y for y in db_eu[model_key].keys() if y <= 2025])
    if len(ys) < 2: return False
    last2 = ys[-2:]
    vals = [int(db_eu[model_key][y].get("units_europe",0)) for y in last2]
    avg = sum(vals)/2 if vals else 0
    return avg >= CONTINUE_SINCE_2025_MIN_AVG_EU

# ------------------ Model key matching -------------
def find_best_model_key(*dbs: Dict[str, Dict[int, Dict[str, Any]]], user_model: str) -> Optional[str]:
    key = normalize_name(user_model)
    for db in dbs:
        if key in db:
            return key
    # fuzzy pass: pick highest token overlap (very conservative)
    all_keys = sorted(set(k for db in dbs for k in db.keys()))
    best, best_score = None, -1
    key_tokens = set(key.split())
    for cand in all_keys:
        s = len(key_tokens & set(cand.split()))
        if s > best_score:
            best, best_score = cand, s
    return best

# ------------------ Sales assembly -----------------
def build_generation_sales_series(
    model: str,
    gen_window: Tuple[int, Optional[int]],
    db_eu: Dict[str,Dict[int,Dict[str,Any]]],
    db_world: Dict[str,Dict[int,Dict[str,Any]]],
    extend_if_2025: bool
) -> Tuple[List[Dict[str,Any]], Tuple[int,int]]:
    """
    Returns rows with Europe+World per year inside active window.
    Missing years are filled with lifecycle model based on nearest available local value.
    """
    mk = find_best_model_key(db_eu, db_world, user_model=model)
    start, end = gen_window[0], gen_window[1]
    # extension if needed
    if end is None:
        end = min(DEADLINE_YEAR, current_year())
    if extend_if_2025 and end == 2025 and mk and should_continue_after_2025(mk, db_eu):
        end = min(2025 + CONTINUE_EXTENSION_YEARS, DEADLINE_YEAR)

    # collect local seeds
    eu_local: Dict[int,int] = {}
    world_local: Dict[int,int] = {}
    if mk and mk in db_eu:
        for y, rec in db_eu[mk].items():
            if start <= y <= end and isinstance(rec.get("units_europe"), int):
                eu_local[y] = int(rec["units_europe"])
    if mk and mk in db_world:
        for y, rec in db_world[mk].items():
            if start <= y <= end and isinstance(rec.get("units_world"), int):
                world_local[y] = int(rec["units_world"])

    # base series by EU (priority) else World
    years = list(range(start, end+1))
    rows: List[Dict[str,Any]] = []
    # find an anchor value (prefer EU)
    anchor_year = None
    anchor_val_eu = None
    for y in years:
        if y in eu_local:
            anchor_year = y; anchor_val_eu = eu_local[y]; break
    if anchor_year is None:
        # fallback to world → we will copy to EU with cap
        for y in years:
            if y in world_local:
                anchor_year = y; anchor_val_eu = int(world_local[y]*0.5)  # assume EU ~50% if only World exists
                break

    def lifecycle_value(year_idx_from_anchor: int, base: int) -> int:
        """
        Simple lifecycle growth/decay from the anchor.
        """
        if year_idx_from_anchor < 0:
            # going backward: mirror late decay
            steps = -year_idx_from_anchor
            v = base
            for _ in range(steps):
                v = int(max(0, v * (1 + LATE_DECAY_RATE)))
            return v
        v = base
        # early growth
        for i in range(min(year_idx_from_anchor, EARLY_GROWTH_YEARS)):
            v = int(max(0, v * (1 + EARLY_GROWTH_RATE)))
        # mid growth
        rem = max(0, year_idx_from_anchor - EARLY_GROWTH_YEARS)
        for i in range(min(rem, MID_GROWTH_YEARS)):
            v = int(max(0, v * (1 + MID_GROWTH_RATE)))
        # late decay
        rem2 = max(0, year_idx_from_anchor - EARLY_GROWTH_YEARS - MID_GROWTH_YEARS)
        for i in range(rem2):
            v = int(max(0, v * (1 + LATE_DECAY_RATE)))
        return v

    for y in years:
        eu_val = eu_local.get(y)
        w_val  = world_local.get(y)
        if eu_val is None and anchor_year is not None and anchor_val_eu is not None:
            eu_val = lifecycle_value(y - anchor_year, anchor_val_eu)
        if eu_val is None: eu_val = 0
        if w_val is None:
            # naive EU→World band: 25%..85% share; pick center to synthesize
            lo_share, hi_share = 0.25, 0.85
            share = (lo_share + hi_share)/2.0
            w_val = max(eu_val, int(eu_val / max(share,1e-6)))
        # caps and Europe ≤ World
        w_val = min(w_val, WORLD_MAX_CAP)
        if eu_val > w_val: w_val = eu_val
        rows.append({"year": y, "world_sales_units": int(w_val), "europe_sales_units": int(eu_val), "rationale": "local db + lifecycle fill"})
    return rows, (start, end)

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

def save_excel(data: Dict[str,Any], fleet_repair: List[Dict[str,Any]], base: str) -> str:
    estimates = pd.DataFrame([
        {"Year": r["year"], "World_Sales": int(r.get("world_sales_units",0)),
         "Europe_Sales": int(r.get("europe_sales_units",0)), "Rationale": r.get("rationale","")}
        for r in data.get("yearly_estimates", [])
    ])
    fr = pd.DataFrame(fleet_repair)[["year","world_fleet","europe_fleet","world_repairs","europe_repairs"]] \
           .rename(columns={"year":"Year","world_fleet":"World_Fleet","europe_fleet":"Europe_Fleet",
                            "world_repairs":"World_Windshield_Repairs","europe_repairs":"Europe_Windshield_Repairs"})
    xlsx = f"{base}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        estimates.to_excel(writer, sheet_name="Estimates", index=False)
        fr.to_excel(writer, sheet_name="Fleet_Repairs", index=False)
    return xlsx

# ------------------ CLI inputs ---------------------
def ask_user_inputs() -> Dict[str,Any]:
    model = input("Enter the car model: ").strip()
    year = int(input("Enter the reference year (we will detect the active generation for that year): ").strip())
    if year < USER_PROMPT_MIN_YEAR or year > DEADLINE_YEAR:
        print(f"Year must be between {USER_PROMPT_MIN_YEAR} and {DEADLINE_YEAR}.", file=sys.stderr); sys.exit(1)
    return {"car_model": model, "ref_year": year}

# ------------------ Main ---------------------------
def main():
    if not SERPAPI_KEY:
        sys.exit(2)

    user = ask_user_inputs()

    # 1) WEB FIRST: detect active generation + window from Wikipedia
    label, window, source_url = find_active_generation_from_wikipedia(user["car_model"], user["ref_year"])
    if not window:
        print("❌ Could not detect a generation window from the web.", file=sys.stderr)
        sys.exit(3)

    start, end = window
    ongoing = (end is None)
    print(f"Detected from web: label={label or '(unlabeled)'} window={start}–{('present' if ongoing else end)}")
    if source_url: print(f"Source: {source_url}")

    # 2) Local DB load (ONLY for sales numbers)
    db_eu = load_local_database_eu(os.getcwd())
    db_world = load_local_database_world(os.getcwd())

    # 3) Build sales series for active years (extend if 2025 and heuristic passes)
    rows, final_window = build_generation_sales_series(
        user["car_model"],
        (start, end),
        db_eu,
        db_world,
        extend_if_2025=True
    )

    # 4) Compose output object
    data = {
        "model": user["car_model"],
        "detected_generation_label": label,
        "source_url": source_url,
        "start_year": final_window[0],
        "end_year": final_window[1],
        "yearly_estimates": rows,
        "notes": "Web-first generation/window via Wikipedia; local DB used only for sales numbers. Missing years filled with lifecycle."
    }

    # 5) Save files
    safe_model = user["car_model"].replace(" ", "_")
    win_txt = f"{final_window[0]}_{final_window[1]}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"webfirst_estimates_{safe_model}_{win_txt}_{timestamp}"

    csv_path = save_csv(data, base)
    fleet = compute_fleet_and_repairs(rows, DECAY_RATE, REPAIR_RATE)
    xlsx_path = save_excel(data, fleet, base)

    # 6) Console summary
    print("\n=== Web-First Generation Sales (EU+World) ===")
    print(f"Model: {data['model']}")
    print(f"Generation (label): {data['detected_generation_label']}")
    print(f"Window: {data['start_year']}–{data['end_year']}")
    print(f"Source: {data['source_url']}")
    tbl = [[r["year"], r["world_sales_units"], r["europe_sales_units"]] for r in data["yearly_estimates"][:12]]
    print(tabulate(tbl, headers=["Year","World","Europe"], tablefmt="github", numalign="right"))
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved Excel:{xlsx_path}")
    print("\nNotes:")
    print("- Generation/window detected from Wikipedia text (via SerpAPI search).")
    print("- Local DB used only for sales values; lifecycle fills gaps.")
    print(f"- If detected end=2025 and recent EU avg ≥ {CONTINUE_SINCE_2025_MIN_AVG_EU:,}, window extends by {CONTINUE_EXTENSION_YEARS} yr(s) with {int(CONTINUE_DECAY_PER_YEAR*100)}%/yr decay.")
    print("- You can tune thresholds at the top of the script.")

if __name__ == "__main__":
    main()

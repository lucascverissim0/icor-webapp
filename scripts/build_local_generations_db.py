#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build/refresh a local JSON database of generation windows for car models,
using your existing wikipedia_gen.py logic and the Top100_* source files.

Inputs (expected in /data):
  data/Top100_YYYY.txt
  data/Top100_World_YYYY.txt
  ... for 2015..2025 (inclusive)

Output:
  data/generations.json  (schema below)

Schema:
{
  "models": [
    {
      "model": "Car Name",
      "parent_model": null,
      "generations": [
        { "name": "Mk7", "start_year": 2012, "end_year": 2019 },
        { "name": "Mk8", "start_year": 2019, "end_year": null }
      ]
    }
  ],
  "meta": { "version": 1, "updated_at": "YYYY-MM-DD" }
}

Run (from repo root):
  python scripts/build_local_generations_db.py --years 2015-2025 --lang en --print-lines
"""
import os, sys, json, re, argparse
from datetime import date
from typing import Dict, Any, List, Set, Tuple, Optional

# --- paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(REPO_ROOT, "data")

# Use your existing scraper
sys.path.append(SCRIPT_DIR)
from wikipedia_gen import scrape_and_cache, read_generation_cache  # noqa: E402

DEFAULT_TOPLIST_DIR = DATA_DIR                       # <-- reads straight from /data
DEFAULT_OUTPUT_PATH = os.path.join(DATA_DIR, "generations.json")

# ---------- helpers ----------
def _normalize_model_name(model: str) -> str:
    return re.sub(r"\s+", " ", model.strip())

def _short_gen_name(label_text: str) -> str:
    s = str(label_text).strip()

    # Mk / Mark + number or roman
    m = re.search(r'\b(?:mk|mark)\s*([ivxlcdm\d]{1,4})\b', s, flags=re.I)
    if m:
        return f"Mk{m.group(1).upper()}"

    # "First/Second/... generation"
    m2 = re.search(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+generation\b', s, re.I)
    if m2:
        roman = {"first":"I","second":"II","third":"III","fourth":"IV","fifth":"V","sixth":"VI",
                 "seventh":"VII","eighth":"VIII","ninth":"IX","tenth":"X"}
        return roman.get(m2.group(1).lower(), m2.group(1).title())

    # Platform/code like 8Y, W177, F3, etc.
    m3 = re.search(r'\b([A-Z]{1,3}\d{1,3}|[A-Z]\d{1,3}[A-Z]?)\b', s)
    if m3:
        return m3.group(1)

    # Fallback: heading snippet, no trailing parens
    s = re.sub(r'\s*\(.*?\)\s*', '', s).strip()
    return (s[:32] + "…") if len(s) > 32 else s

def _read_toplist_models(toplist_dir: str, years: Tuple[int,int]) -> List[str]:
    """Collect unique `model` strings from Top100_YYYY.txt and Top100_World_YYYY.txt in /data."""
    y0, y1 = years
    patterns = []
    for y in range(y0, y1 + 1):
        patterns.append(os.path.join(toplist_dir, f"Top100_{y}.txt"))
        patterns.append(os.path.join(toplist_dir, f"Top100_World_{y}.txt"))

    seen: Set[str] = set()
    models: List[str] = []

    for path in patterns:
        if not os.path.exists(path):
            continue
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
        except Exception as e:
            print(f"[warn] could not read {path}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            print(f"[warn] {path} is not a JSON list, skipping", file=sys.stderr)
            continue
        for row in data:
            if not isinstance(row, dict): continue
            model = row.get("model")
            if not model: continue
            m = _normalize_model_name(model)
            if m not in seen:
                seen.add(m)
                models.append(m)

    return models

def _load_db(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            pass
    return {"models": [], "meta": {"version": 1}}

def _find_entry(db: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    for m in db.get("models", []):
        if m.get("model") == model:
            return m
    return None

def _upsert_entry(db: Dict[str, Any], model: str, generations: List[Dict[str, Any]]) -> None:
    entry = _find_entry(db, model)
    if entry is None:
        db["models"].append({"model": model, "parent_model": None, "generations": generations})
    else:
        entry["generations"] = generations

def _windows_to_generations(wins: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert wikipedia_gen cache windows → minimal schema (end 9999 → null)."""
    gens: List[Dict[str, Any]] = []
    for w in wins:
        try:
            s = int(w["start"])
            e_raw = int(w["end"])
        except Exception:
            continue
        gens.append({
            "name": _short_gen_name(w.get("label", "GEN")),
            "start_year": s,
            "end_year": None if e_raw == 9999 else e_raw
        })
    gens.sort(key=lambda g: (g["start_year"], g["end_year"] if g["end_year"] is not None else 9999))
    # de-dup exact repeats
    out, seen = [], set()
    for g in gens:
        key = (g["name"], g["start_year"], g["end_year"])
        if key in seen: continue
        seen.add(key); out.append(g)
    return out

def _print_compact(model: str, gens: List[Dict[str, Any]]) -> None:
    parts = [f"generation {i}: {g['start_year']}-{g['end_year'] if g['end_year'] is not None else 'present'}"
             for i, g in enumerate(gens, 1)]
    print(f"{model}: " + ", ".join(parts))

# ---------- main build ----------
def build_db(toplist_dir: str,
             output_path: str,
             years: Tuple[int,int],
             lang: str,
             max_models: int = 0,
             print_lines: bool = False) -> Dict[str, Any]:

    models = _read_toplist_models(toplist_dir, years)
    if max_models and len(models) > max_models:
        models = models[:max_models]

    print(f"[info] unique models from toplists {years[0]}–{years[1]}: {len(models)}")

    db = _load_db(output_path)
    hint_year = years[1]  # used by wikipedia_gen to pick the right window if needed

    for idx, model in enumerate(models, 1):
        try:
            # scrape_and_cache writes/refreshes cache using your logic (multi-language + inference)
            scrape_and_cache(model, hint_year, lang=lang)
            cached = read_generation_cache(model)
            if not cached or not cached.get("windows"):
                print(f"[warn] no windows cached for '{model}', skipping", file=sys.stderr)
                continue

            gens = _windows_to_generations(cached["windows"])
            if not gens:
                print(f"[warn] empty generations after transform for '{model}', skipping", file=sys.stderr)
                continue

            _upsert_entry(db, model, gens)
            if print_lines:
                _print_compact(model, gens)

            if idx % 10 == 0:
                print(f"[progress] {idx}/{len(models)} done")

        except Exception as e:
            print(f"[error] {model}: {e}", file=sys.stderr)
            continue

    db.setdefault("meta", {})["version"] = db.get("meta", {}).get("version", 1)
    db["meta"]["updated_at"] = date.today().isoformat()
    db["models"].sort(key=lambda m: m["model"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote {output_path} with {len(db['models'])} models")
    return db

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toplist-dir", default=DEFAULT_TOPLIST_DIR,
                    help="Directory containing Top100_YYYY.txt and Top100_World_YYYY.txt (defaults to ./data)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT_PATH,
                    help="Output JSON path (defaults to ./data/generations.json)")
    ap.add_argument("--lang", default="en", help="Preferred Wikipedia language to try first")
    ap.add_argument("--years", default="2015-2025",
                    help="Inclusive range, e.g. 2015-2025 (or a single year like 2025)")
    ap.add_argument("--max", type=int, default=0, help="Process only the first N models (0 = no limit)")
    ap.add_argument("--print-lines", action="store_true", help="Print 'Car model XYZ: generation 1: …' lines")
    return ap.parse_args()

def main():
    args = parse_args()
    if "-" in args.years:
        y0_s, y1_s = args.years.split("-", 1)
        years = (int(y0_s), int(y1_s))
    else:
        y = int(args.years)
        years = (y, y)

    build_db(
        toplist_dir=args.toplist_dir,
        output_path=args.output,
        years=years,
        lang=args.lang,
        max_models=args.max,
        print_lines=args.print_lines
    )

if __name__ == "__main__":
    main()

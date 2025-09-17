#!/usr/bin/env python3
"""
wikipedia_gen.py
----------------
Scrape generation windows (launch/end years) for a given car model from Wikipedia,
normalize them, and write them into a JSON cache file under cache/gen_windows/.

Usage:
    python scripts/wikipedia_gen.py --year 2022 "Volkswagen Golf" --write-cache --json

Features:
- Multilingual support (en, fr, de, es, it, pt).
- Detects headings like "First generation (2008–2013)" and infobox spans.
- Infers missing end years as (next launch year - 1).
- Outputs structured JSON with all generations.
- Can emit the active generation for a given year (--json).
"""

import sys, re, html, os, json, time, argparse, datetime, urllib.parse
from typing import Optional, Tuple
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

CACHE_DIR = "cache/gen_windows"

# ---------- HTTP session ----------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "icor-gen-scraper/1.0"})
retries = Retry(total=4, backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"])
SESSION.mount("https://", HTTPAdapter(max_retries=retries))
REQUEST_DELAY_SEC = 0.2

def _get(url, params):
    time.sleep(REQUEST_DELAY_SEC)
    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r

# ---------- Regexes ----------
PRESENT_WORDS = {"present", "current", "ongoing", "présent", "actuel", "presente", "attuale", "oggi",
                 "atual", "aktuell", "heute", "bis heute"}
PARENS_RANGE_FLEX_RE = re.compile(r"\((?:[^()]*?;?\s*)?(?P<start>\d{4})\s*[–—-]\s*(?P<end>\d{4}|"
                                  r"(?:present|current|ongoing|présent|actuel|presente|attuale|oggi|"
                                  r"atual|aktuell|heute|bis heute))?\)", re.IGNORECASE)
YEAR_RANGE_FLEX_RE = re.compile(r"(?P<start>\b\d{4}\b)\s*[–—-]\s*(?P<end>\b\d{4}\b|"
                                r"(?:present|current|ongoing|présent|actuel|presente|attuale|oggi|"
                                r"atual|aktuell|heute|bis heute))", re.IGNORECASE)
MK_TEXT_RE = re.compile(r"\b(?:mk|mark)\s*(?:[ivxlcdm]+|\d+)\b", re.IGNORECASE)
GEN_KEYWORDS = ["generation", "génération", "generación", "geração", "generazione"]

# ---------- Helpers ----------
def _safe_slug(s: str, max_len: int = 80) -> str:
    s = str(s).replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^A-Za-z0-9\-_\. ]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:max_len] if s else "untitled"

def extract_years_any(text: str):
    t = text.replace("—", "–").replace("-", "–")
    m = PARENS_RANGE_FLEX_RE.search(t) or YEAR_RANGE_FLEX_RE.search(t)
    if not m:
        return None, None
    start, end = m.group("start"), m.group("end")
    if not end or str(end).lower() in PRESENT_WORDS:
        end = "present"
    return start, end

def heading_looks_like_generation(h: str) -> bool:
    low = h.lower()
    return bool(MK_TEXT_RE.search(low) or any(kw in low for kw in GEN_KEYWORDS) or PARENS_RANGE_FLEX_RE.search(low))

def extract_generation_headings(soup: BeautifulSoup):
    return [tag.get_text(" ", strip=True)
            for tag in soup.find_all(["h2", "h3", "h4"])
            if heading_looks_like_generation(tag.get_text(" ", strip=True))]

def extract_infobox_years(soup: BeautifulSoup):
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
    if not infobox: return None, None
    for row in infobox.find_all("tr"):
        th, td = row.find("th"), row.find("td")
        if not th or not td: continue
        label = th.get_text(" ", strip=True).lower()
        if "production" in label or "model years" in label:
            return extract_years_any(td.get_text(" ", strip=True))
    return None, None

def parse_generations_from_html(html_content: str):
    soup = BeautifulSoup(html_content, "html.parser")
    out, seen = [], set()
    for heading in extract_generation_headings(soup):
        s, e = extract_years_any(heading)
        if not s: continue
        key = (s, e, heading)
        if key in seen: continue
        seen.add(key)
        out.append({"label": heading, "launch_year": s, "end_year": e, "source": "heading"})
    if not out:
        s, e = extract_infobox_years(soup)
        if s:
            out.append({"label": "Infobox production/model years", "launch_year": s, "end_year": e, "source": "infobox"})
    def sort_key(it): return int(it["launch_year"]) if it["launch_year"] else 99999
    out.sort(key=sort_key)
    return out

def infer_end_years_inplace(items):
    items.sort(key=lambda g: int(g["launch_year"]))
    for i in range(len(items)-1):
        curr, nxt = items[i], items[i+1]
        if not curr.get("end_year") or curr["end_year"] in ("present",""):
            curr["end_year"] = str(int(nxt["launch_year"]) - 1)
            curr["end_year_inferred"] = True

# ---------- Cache ----------
def write_generation_cache(model: str, lang: str, windows: list, cache_dir: str = CACHE_DIR) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "model": model,
        "lang": lang,
        "scraped_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "windows": [
            {"label": w["label"], "start": int(w["launch_year"]),
             "end": (9999 if str(w["end_year"]).lower()=="present" else int(w["end_year"])),
             "source": w.get("source","unknown")}
            for w in windows
        ]
    }
    fname = os.path.join(cache_dir, f"{_safe_slug(model)}.json")
    with open(fname, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    return fname

def scrape_and_cache(model: str, year: int, lang: str = "en", cache_dir: str = CACHE_DIR):
    global WIKI_API
    WIKI_API = f"https://{lang}.wikipedia.org/w/api.php"
    pid, title = search_wikipedia_page(model)
    if not pid: raise RuntimeError(f"No page found for {model}")
    html_main = fetch_page_html_by_pageid(pid)
    gens = parse_generations_from_html(html_main)
    infer_end_years_inplace(gens)
    path = write_generation_cache(model, lang, gens, cache_dir)
    return path, gens

# ---------- Wikipedia API ----------
def search_wikipedia_page(query: str):
    params = {"action":"query","list":"search","srsearch":query,
              "srlimit":5,"format":"json","srnamespace":0}
    r = _get(WIKI_API, params)
    results = r.json().get("query",{}).get("search",[])
    if not results: return None, None
    top = results[0]
    return top["pageid"], html.unescape(top["title"])

def fetch_page_html_by_pageid(pageid: int) -> str:
    params = {"action":"parse","pageid":pageid,"prop":"text","format":"json"}
    r = _get(WIKI_API, params)
    return r.json().get("parse",{}).get("text",{}).get("*","")

def get_pageid_by_title(title: str):
    params = {"action":"query","titles":title,"format":"json"}
    r = _get(WIKI_API, params)
    pages = r.json().get("query",{}).get("pages",{})
    for pid,p in pages.items():
        if pid != "-1": return int(pid)
    return None

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--write-cache", action="store_true")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--json", action="store_true")
    return ap.parse_args()

# ---------- Lightweight API for other modules ----------
def _cache_path_for(model: str, cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"{_safe_slug(model)}.json")

def read_generation_cache(model: str, cache_dir: str = CACHE_DIR) -> Optional[dict]:
    path = _cache_path_for(model, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _pick_window_for_year(payload: dict, year: int) -> Optional[dict]:
    wins = payload.get("windows") or []
    if not wins:
        return None
    # Prefer the window covering the year; else pick the closest (by distance)
    covering = [w for w in wins if w["start"] <= year <= (w["end"] if w["end"] != 9999 else 9999)]
    if covering:
        # If multiple cover, choose the narrowest, then latest start
        covering.sort(key=lambda w: (((w["end"] if w["end"] != 9999 else 9999) - w["start"]), -w["start"]))
        return covering[0]
    # No covering window: choose the nearest by boundary distance
    def dist(w):
        s, e = w["start"], (w["end"] if w["end"] != 9999 else 9999)
        if year < s: return s - year
        if year > e: return year - e
        return 0
    wins_sorted = sorted(wins, key=lambda w: (dist(w), -w["start"]))
    return wins_sorted[0] if wins_sorted else None

def _mk_label_from_text(label_text: str) -> str:
    # Extract a compact generation tag like Mk7, MkVII, etc., else "GEN"
    m = re.search(r'\b(?:mk|mark|gen(?:eration)?)\s*([ivxlcdm\d]{1,4})\b', str(label_text), flags=re.I)
    if m:
        val = m.group(1).upper()
        return f"Mk{val}"
    return "GEN"

def detect_via_wikipedia(model: str, year: int, lang: str = "en",
                         cache_dir: str = CACHE_DIR) -> Tuple[Optional[str], Optional[Tuple[int, int]], dict]:
    """
    Public function used by script2.
    Returns: (gen_label, (start, end), diag)
      - end may be 9999 when the generation is 'present' (open-ended).
    Behavior:
      1) Try cache (cache/gen_windows/<model>.json)
      2) If missing, call scrape_and_cache(...) to build it, then re-read
    """
    # 1) Try cache
    cached = read_generation_cache(model, cache_dir)
    basis = "cache"
    if not cached:
        # 2) Build cache
        try:
            scrape_and_cache(model, year, lang, cache_dir)
        except Exception as e:
            return None, None, {"basis": "wikipedia_gen", "status": "scrape_failed", "error": str(e)}
        cached = read_generation_cache(model, cache_dir)
        basis = "scraped"
        if not cached:
            # Unexpected: scraper ran but nothing could be read
            return None, None, {"basis": "wikipedia_gen", "status": "no_cache_after_scrape"}

    pick = _pick_window_for_year(cached, year)
    if not pick:
        return None, None, {"basis": "wikipedia_gen", "status": "no_window_in_cache", "cache_model": cached.get("model")}

    start = int(pick["start"])
    end = int(pick["end"])  # may be 9999
    label_text = pick.get("label") or ""
    gen_label = _mk_label_from_text(label_text)

    diag = {
        "basis": "wikipedia_gen",
        "source": basis,
        "cache_lang": cached.get("lang"),
        "scraped_at": cached.get("scraped_at"),
        "picked_label": label_text,
        "picked_source": pick.get("source"),
    }
    return gen_label, (start, end), diag

def main():
    args = parse_args()
    path, gens = scrape_and_cache(args.model, args.year, args.lang, args.cache_dir)
    print(f"[cache] wrote: {path}")
    if args.json:
        cached = json.load(open(path,"r",encoding="utf-8"))
        wins = cached["windows"]; yr = args.year
        pick = next((w for w in wins if w["start"] <= yr <= (w["end"] if w["end"]!=9999 else 9999)), wins[0])
        print(json.dumps({
            "model": cached["model"], "label": pick["label"],
            "start": pick["start"], "end": pick["end"],
            "diag": {"basis":"cache","lang":cached["lang"],"scraped_at":cached["scraped_at"]}
        }))

if __name__ == "__main__":
    main()

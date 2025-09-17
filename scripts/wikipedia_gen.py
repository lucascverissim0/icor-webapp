#!/usr/bin/env python3
"""
wikipedia_gen.py
----------------
Scrape generation windows (launch/end years) for a given car model from Wikipedia,
normalize them, and write them into a JSON cache file.

Also exposes:
    detect_via_wikipedia(model, year, lang="en", cache_dir=CACHE_DIR)
which is consumed by script2 to get a generation label + window for a given view year.

CLI:
    python scripts/wikipedia_gen.py --year 2023 "Volkswagen Golf" --json
"""

import sys, re, html, os, json, time, argparse, datetime
from typing import Optional, Tuple, List, Dict

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

# =============== Paths (anchor to repo root; allow override) ===============
_THIS_FILE   = os.path.abspath(__file__)
_REPO_ROOT   = os.path.dirname(os.path.dirname(_THIS_FILE))  # parent of /scripts

# Choose default cache home (A = repo-root; B = ui/)
_DEFAULT_CACHE_DIR = os.path.join(_REPO_ROOT, "cache", "gen_windows")   # recommended
# _DEFAULT_CACHE_DIR = os.path.join(_REPO_ROOT, "ui", "cache", "gen_windows")  # alt

CACHE_DIR = os.environ.get("ICOR_CACHE_DIR", _DEFAULT_CACHE_DIR)

# =============== HTTP session ===============
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "icor-gen-scraper/1.2"})
retries = Retry(
    total=4,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
SESSION.mount("https://", HTTPAdapter(max_retries=retries))
REQUEST_DELAY_SEC = 0.2

def _get(url, params):
    time.sleep(REQUEST_DELAY_SEC)
    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r

# =============== Regexes ===============
PRESENT_WORDS = {
    "present", "current", "ongoing", "présent", "actuel", "presente",
    "attuale", "oggi", "atual", "aktuell", "heute", "bis heute"
}
PARENS_RANGE_FLEX_RE = re.compile(
    r"\((?:[^()]*?;?\s*)?(?P<start>\d{4})\s*[–—-]\s*(?P<end>\d{4}|"
    r"(?:present|current|ongoing|présent|actuel|presente|attuale|oggi|atual|aktuell|heute|bis heute))?\)",
    re.IGNORECASE
)
YEAR_RANGE_FLEX_RE = re.compile(
    r"(?P<start>\b\d{4}\b)\s*[–—-]\s*(?P<end>\b\d{4}\b|"
    r"(?:present|current|ongoing|présent|actuel|presente|attuale|oggi|atual|aktuell|heute|bis heute))",
    re.IGNORECASE
)
MK_TEXT_RE = re.compile(r"\b(?:mk|mark)\s*(?:[ivxlcdm]+|\d+)\b", re.IGNORECASE)
GEN_KEYWORDS = ["generation", "génération", "generación", "geração", "generazione"]

# Robust full-text pattern for "MkX (2019–present)" or "Eighth generation (Mk8; 2019–present)"
FULLTEXT_MK_RANGE_RE = re.compile(
    r"(?P<label>(?:[^()\n]{0,40})?"
    r"(?:generation|génération|generación|geração|generazione|mk|mark)\s*"
    r"([ivxlcdm\d]{1,4})"
    r"(?:[^()\n]{0,60}))"
    r"\s*\((?:[^()]*)?(?P<start>19\d{2}|20\d{2})\s*[–—-]\s*"
    r"(?P<end>19\d{2}|20\d{2}|present|current|ongoing|présent|actuel|presente|attuale|oggi|atual|aktuell|heute|bis heute)\)",
    re.IGNORECASE
)

# =============== Helpers ===============
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
    return [
        tag.get_text(" ", strip=True)
        for tag in soup.find_all(["h2", "h3", "h4"])
        if heading_looks_like_generation(tag.get_text(" ", strip=True))
    ]

def extract_infobox_years(soup: BeautifulSoup):
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
    if not infobox:
        return None, None
    for row in infobox.find_all("tr"):
        th, td = row.find("th"), row.find("td")
        if not th or not td:
            continue
        label = th.get_text(" ", strip=True).lower()
        if "production" in label or "model years" in label:
            return extract_years_any(td.get_text(" ", strip=True))
    return None, None

def _normalize_end(end: Optional[str]) -> str:
    if not end:
        return "present"
    e = str(end).strip().lower()
    return "present" if e in PRESENT_WORDS else end

def parse_generations_from_html(html_content: str) -> List[Dict[str, str]]:
    """
    Returns list of dicts:
      {"label": str, "launch_year": str, "end_year": str, "source": "heading|infobox|fulltext"}
    """
    soup = BeautifulSoup(html_content, "html.parser")
    out: List[Dict[str, str]] = []
    seen = set()

    # Pass 1: headings that look like generations
    for heading in extract_generation_headings(soup):
        s, e = extract_years_any(heading)
        if not s:
            continue
        e = _normalize_end(e)
        key = (s, e, heading)
        if key in seen:
            continue
        seen.add(key)
        out.append({"label": heading, "launch_year": s, "end_year": e, "source": "heading"})

    # Pass 2: full-text Mk + range
    if len(out) <= 1:
        fulltext = soup.get_text(" ", strip=True)
        for m in FULLTEXT_MK_RANGE_RE.finditer(fulltext):
            label = re.sub(r"\s+", " ", m.group("label")).strip()
            s = m.group("start")
            e = _normalize_end(m.group("end"))
            key = (s, e, label)
            if key in seen:
                continue
            seen.add(key)
            out.append({"label": label, "launch_year": s, "end_year": e, "source": "fulltext"})

    # Pass 3: infobox (model-wide), only if nothing else
    if not out:
        s, e = extract_infobox_years(soup)
        if s:
            out.append({
                "label": "Infobox production/model years",
                "launch_year": s,
                "end_year": _normalize_end(e),
                "source": "infobox"
            })

    def sort_key(it):
        try:
            return int(it["launch_year"])
        except Exception:
            return 99999

    out.sort(key=sort_key)
    return out

def infer_end_years_inplace(items: List[Dict[str, str]]) -> None:
    """If a generation has 'present'/missing end, set end = next launch - 1."""
    items.sort(key=lambda g: int(g["launch_year"]))
    for i in range(len(items) - 1):
        curr, nxt = items[i], items[i + 1]
        if not curr.get("end_year") or str(curr["end_year"]).lower() in ("present", ""):
            curr["end_year"] = str(int(nxt["launch_year"]) - 1)
            curr["end_year_inferred"] = True

# =============== Cache ===============
def _cache_path_for(model: str, cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"{_safe_slug(model)}.json")

def write_generation_cache(model: str, lang: str, windows: List[Dict[str, str]], cache_dir: str = CACHE_DIR) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    if os.environ.get("ICOR_DEBUG_CACHE") == "1":
        print(f"[wikipedia_gen] write -> {os.path.abspath(cache_dir)}", file=sys.stderr)
    payload = {
        "model": model,
        "lang": lang,
        "scraped_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "windows": [
            {
                "label": w["label"],
                "start": int(w["launch_year"]),
                "end": (9999 if str(w["end_year"]).lower() == "present" else int(w["end_year"])),
                "source": w.get("source", "unknown"),
            }
            for w in windows
        ],
    }
    fname = _cache_path_for(model, cache_dir)
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fname

def read_generation_cache(model: str, cache_dir: str = CACHE_DIR) -> Optional[dict]:
    path = _cache_path_for(model, cache_dir)
    if os.environ.get("ICOR_DEBUG_CACHE") == "1":
        print(f"[wikipedia_gen] read -> {os.path.abspath(path)}", file=sys.stderr)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# =============== Wikipedia API ===============
def search_wikipedia_page(query: str, lang: str = "en"):
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "format": "json",
        "srlimit": 5,
        "srsearch": query,
        "srnamespace": 0,
    }
    r = _get(api, params)
    results = r.json().get("query", {}).get("search", [])
    if not results:
        return None, None, api
    top = results[0]
    return top["pageid"], html.unescape(top["title"]), api

def fetch_page_html_by_pageid(pageid: int, api: str) -> str:
    params = {"action": "parse", "pageid": pageid, "prop": "text", "format": "json", "disableeditsection": 1}
    r = _get(api, params)
    return r.json().get("parse", {}).get("text", {}).get("*", "")

# ---- subpage helpers (Mk pages) ----
def wiki_search(query: str, lang: str = "en", limit: int = 8) -> List[Dict]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "format": "json", "srlimit": limit, "srsearch": query, "srnamespace": 0}
    r = _get(api, params)
    return r.json().get("query", {}).get("search", [])

def fetch_page_html_by_title(title: str, lang: str = "en") -> str:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {"action": "parse", "page": title, "prop": "text", "format": "json", "disableeditsection": 1}
    r = _get(api, params)
    return r.json().get("parse", {}).get("text", {}).get("*", "")

def parse_infobox_years_from_html(html_content: str) -> Optional[Tuple[int, int]]:
    soup = BeautifulSoup(html_content, "html.parser")
    start, end = extract_infobox_years(soup)
    if not start:
        return None
    try:
        s = int(start)
        e = 9999 if (not end or str(end).lower() in PRESENT_WORDS) else int(end)
        return (s, e)
    except Exception:
        return None

def harvest_mk_subpages(model: str, lang: str = "en") -> List[Dict[str, str]]:
    """
    Search subpages like 'Volkswagen Golf Mk8' and parse their infobox/model-year ranges
    as generation windows.
    """
    queries = [
        f'"{model}" Mk',
        f'"{model}" Mark',
        f'{model} generation Mk',
    ]
    seen_titles = set()
    windows: List[Dict[str, str]] = []

    for q in queries:
        for hit in wiki_search(q, lang=lang, limit=10):
            title = html.unescape(hit.get("title", ""))
            low = title.lower()
            if title in seen_titles:
                continue
            if not re.search(r'\b(mk|mark)\s*[ivxlcdm\d]{1,4}\b', low):
                continue
            if model.lower().split()[0] not in low:
                continue
            seen_titles.add(title)

            try:
                html_page = fetch_page_html_by_title(title, lang)
            except Exception:
                continue

            rng = parse_infobox_years_from_html(html_page)
            if not rng:
                gens = parse_generations_from_html(html_page)
                if gens:
                    try:
                        s = int(gens[0]["launch_year"])
                        e_raw = gens[0]["end_year"]
                        e = 9999 if str(e_raw).lower() == "present" else int(e_raw)
                        rng = (s, e)
                    except Exception:
                        rng = None

            if rng:
                s, e = rng
                windows.append({
                    "label": title,
                    "launch_year": str(s),
                    "end_year": ("present" if e == 9999 else str(e)),
                    "source": "mk_subpage_infobox",
                })

    # de-dup by (start, end)
    dedup, out = set(), []
    for w in windows:
        key = (w["launch_year"], w["end_year"])
        if key in dedup:
            continue
        dedup.add(key)
        out.append(w)

    out.sort(key=lambda w: int(w["launch_year"]))
    return out

# =============== Scrape + cache orchestrator ===============
def scrape_and_cache(model: str, year: int, lang: str = "en", cache_dir: str = CACHE_DIR):
    pageid, title, api = search_wikipedia_page(model, lang)
    if not pageid:
        raise RuntimeError(f"No Wikipedia page found for {model} [{lang}]")
    html_main = fetch_page_html_by_pageid(pageid, api)

    gens = parse_generations_from_html(html_main)

    # If we only got 0–1 windows (often model-wide), try Mk subpages
    if len(gens) <= 1:
        mk_windows = harvest_mk_subpages(model, lang=lang)
        if mk_windows:
            gens = mk_windows

    if gens:
        infer_end_years_inplace(gens)

    path = write_generation_cache(model, lang, gens, cache_dir)
    return path, gens

# =============== Public API for script2 ===============
def _pick_window_for_year(payload: dict, year: int) -> Optional[dict]:
    wins = payload.get("windows") or []
    if not wins:
        return None
    covering = [w for w in wins if w["start"] <= year <= (w["end"] if w["end"] != 9999 else 9999)]
    if covering:
        covering.sort(key=lambda w: (((w["end"] if w["end"] != 9999 else 9999) - w["start"]), -w["start"]))
        return covering[0]

    def dist(w):
        s, e = w["start"], (w["end"] if w["end"] != 9999 else 9999)
        if year < s: return s - year
        if year > e: return year - e
        return 0

    wins_sorted = sorted(wins, key=lambda w: (dist(w), -w["start"]))
    return wins_sorted[0] if wins_sorted else None

def _mk_label_from_text(label_text: str) -> str:
    m = re.search(r'\b(?:mk|mark|gen(?:eration)?)\s*([ivxlcdm\d]{1,4})\b', str(label_text), flags=re.I)
    if m:
        return f"Mk{m.group(1).upper()}"
    return "GEN"

def detect_via_wikipedia(
    model: str, year: int, lang: str = "en", cache_dir: str = CACHE_DIR
) -> Tuple[Optional[str], Optional[Tuple[int, int]], dict]:
    """
    Returns: (gen_label, (start, end), diag)
    - Reads JSON cache first; if missing, scrapes and writes it, then reads.
    - 'end' may be 9999 for 'present'.
    """
    cached = read_generation_cache(model, cache_dir)
    basis = "cache"
    if not cached:
        try:
            scrape_and_cache(model, year, lang, cache_dir)
        except Exception as e:
            return None, None, {"basis": "wikipedia_gen", "status": "scrape_failed", "error": str(e)}
        cached = read_generation_cache(model, cache_dir)
        basis = "scraped"
        if not cached:
            return None, None, {"basis": "wikipedia_gen", "status": "no_cache_after_scrape"}

    pick = _pick_window_for_year(cached, year)
    if not pick:
        return None, None, {"basis": "wikipedia_gen", "status": "no_window_in_cache", "cache_model": cached.get("model")}

    start = int(pick["start"])
    end = int(pick["end"])  # 9999 means 'present'
    gen_label = _mk_label_from_text(pick.get("label", ""))

    diag = {
        "basis": "wikipedia_gen",
        "source": basis,
        "cache_lang": cached.get("lang"),
        "scraped_at": cached.get("scraped_at"),
        "picked_label": pick.get("label"),
        "picked_source": pick.get("source"),
    }
    return gen_label, (start, end), diag

# =============== CLI ===============
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--write-cache", action="store_true")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--json", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    # Force using the given cache dir (if provided)
    path, gens = scrape_and_cache(args.model, args.year, args.lang, args.cache_dir)
    print(f"[cache] wrote: {path}")
    if os.environ.get("ICOR_DEBUG_CACHE") == "1":
        print(f"[wikipedia_gen] wrote {len(gens)} windows to {os.path.abspath(path)}", file=sys.stderr)
    if args.json:
        cached = json.load(open(path, "r", encoding="utf-8"))
        wins = cached["windows"]; yr = args.year
        covering = [w for w in wins if w["start"] <= yr <= (w["end"] if w["end"] != 9999 else 9999)]
        pick = covering[0] if covering else wins[0]
        print(json.dumps({
            "model": cached["model"], "label": pick["label"],
            "start": pick["start"], "end": pick["end"],
            "diag": {"basis": "cache", "lang": cached["lang"], "scraped_at": cached["scraped_at"]}
        }))

if __name__ == "__main__":
    main()

__all__ = [
    "detect_via_wikipedia",
    "scrape_and_cache",
    "read_generation_cache",
]

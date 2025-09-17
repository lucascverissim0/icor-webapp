#!/usr/bin/env python3
"""
wikipedia_gen.py
----------------
Scrape generation windows (launch/end years) for a given car model from Wikipedia,
normalize them, and write them into a JSON cache file.

Public API (for script2):
    detect_via_wikipedia(model, year, lang="en", cache_dir=CACHE_DIR)
      -> (gen_label, (start, end), diag)

CLI:
    python scripts/wikipedia_gen.py --year 2023 "Volkswagen Golf" --json
"""

import sys, re, html, os, json, time, argparse, datetime
from typing import Optional, Tuple, List, Dict

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

# ================= Paths (repo-anchored; env override) =================
_THIS_FILE   = os.path.abspath(__file__)
_REPO_ROOT   = os.path.dirname(os.path.dirname(_THIS_FILE))  # parent of /scripts

_DEFAULT_CACHE_DIR = os.path.join(_REPO_ROOT, "cache", "gen_windows")  # recommended
# _DEFAULT_CACHE_DIR = os.path.join(_REPO_ROOT, "ui", "cache", "gen_windows")  # alt if desired

CACHE_DIR = os.environ.get("ICOR_CACHE_DIR", _DEFAULT_CACHE_DIR)

# ================= HTTP session =================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "icor-gen-scraper/1.3"})
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

# ================= Regexes =================
PRESENT_WORDS = {"present","current","ongoing","présent","actuel","presente","attuale",
                 "oggi","atual","aktuell","heute","bis heute","to present"}
PARENS_RANGE_FLEX_RE = re.compile(
    r"\((?:[^()]*?;?\s*)?(?P<start>\d{4})\s*[–—-]\s*(?P<end>\d{4}|"
    r"(?:present|current|ongoing|présent|actuel|presente|attuale|oggi|atual|aktuell|heute|bis heute|to present))?\)",
    re.I)
YEAR_RANGE_FLEX_RE = re.compile(
    r"(?P<start>\b\d{4}\b)\s*[–—-]\s*(?P<end>\b\d{4}\b|"
    r"(?:present|current|ongoing|présent|actuel|presente|attuale|oggi|atual|aktuell|heute|bis heute|to present))",
    re.I)
MK_TEXT_RE = re.compile(r"\b(?:mk|mark)\s*(?:[ivxlcdm]+|\d+)\b", re.I)
GEN_KEYWORDS = ["generation","génération","generación","geração","generazione"]

# Robust full-text pattern e.g. "Mk8 (2019–present)" or "Second generation (2023– )"
FULLTEXT_MK_RANGE_RE = re.compile(
    r"(?P<label>(?:[^()\n]{0,40})?(?:generation|génération|generación|geração|generazione|mk|mark)"
    r"\s*(?:[ivxlcdm\d]+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
    r"(?:[^()\n]{0,60}))\s*\((?:[^()]*)?(?P<start>19\d{2}|20\d{2})\s*[–—-]\s*(?P<end>19\d{2}|20\d{2}|"
    r"present|current|ongoing|présent|actuel|presente|attuale|oggi|atual|aktuell|heute|bis heute|to present)\)",
    re.I
)

# ================= Helpers =================
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

def extract_generation_headings(soup: BeautifulSoup) -> List[str]:
    return [
        tag.get_text(" ", strip=True)
        for tag in soup.find_all(["h2","h3","h4"])
        if heading_looks_like_generation(tag.get_text(" ", strip=True))
    ]

def extract_infobox_years(soup: BeautifulSoup):
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
    if not infobox: return None, None
    for row in infobox.find_all("tr"):
        th, td = row.find("th"), row.find("td")
        if not th or not td: continue
        label = th.get_text(" ", strip=True).lower()
        if "production" in label or "model years" in label or "années de production" in label:
            return extract_years_any(td.get_text(" ", strip=True))
    return None, None

def _normalize_end(end: Optional[str]) -> str:
    if not end: return "present"
    e = str(end).strip().lower()
    return "present" if e in PRESENT_WORDS else end

def parse_generations_from_html(html_content: str) -> List[Dict[str, str]]:
    """Return list of dicts: {"label","launch_year","end_year","source"}"""
    soup = BeautifulSoup(html_content, "html.parser")
    out: List[Dict[str,str]] = []; seen = set()

    # Pass 1: headings
    for heading in extract_generation_headings(soup):
        s, e = extract_years_any(heading)
        if not s: continue
        e = _normalize_end(e)
        key = (s, e, heading)
        if key in seen: continue
        seen.add(key)
        out.append({"label": heading, "launch_year": s, "end_year": e, "source": "heading"})

    # Pass 2: full-text (covers "Second generation (2023–)")
    if len(out) <= 1:
        text = soup.get_text(" ", strip=True)
        for m in FULLTEXT_MK_RANGE_RE.finditer(text):
            label = re.sub(r"\s+", " ", m.group("label")).strip()
            s = m.group("start"); e = _normalize_end(m.group("end"))
            key = (s, e, label)
            if key in seen: continue
            seen.add(key)
            out.append({"label": label, "launch_year": s, "end_year": e, "source": "fulltext"})

    # Pass 3: infobox (model-wide) if nothing else
    if not out:
        s, e = extract_infobox_years(soup)
        if s:
            out.append({"label":"Infobox production/model years","launch_year":s,"end_year":_normalize_end(e),"source":"infobox"})

    out.sort(key=lambda w: int(w["launch_year"]) if w["launch_year"] else 99999)
    return out

def infer_end_years_inplace(items: List[Dict[str,str]]) -> None:
    items.sort(key=lambda g: int(g["launch_year"]))
    for i in range(len(items)-1):
        curr, nxt = items[i], items[i+1]
        if not curr.get("end_year") or str(curr["end_year"]).lower() in ("present",""):
            curr["end_year"] = str(int(nxt["launch_year"]) - 1)
            curr["end_year_inferred"] = True

# ================= Cache =================
def _cache_path_for(model: str, cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"{_safe_slug(model)}.json")

def write_generation_cache(model: str, lang: str, windows: List[Dict[str,str]], cache_dir: str = CACHE_DIR) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    if os.environ.get("ICOR_DEBUG_CACHE") == "1":
        print(f"[wikipedia_gen] write -> {os.path.abspath(cache_dir)}", file=sys.stderr)
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
    fname = _cache_path_for(model, cache_dir)
    with open(fname, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    return fname

def read_generation_cache(model: str, cache_dir: str = CACHE_DIR) -> Optional[dict]:
    path = _cache_path_for(model, cache_dir)
    if os.environ.get("ICOR_DEBUG_CACHE") == "1":
        print(f"[wikipedia_gen] read -> {os.path.abspath(path)}", file=sys.stderr)
    if not os.path.exists(path): return None
    try:
        return json.load(open(path,"r",encoding="utf-8"))
    except Exception:
        return None

# ================= Wikipedia API (per language) =================
def _api_base(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"

def search_wikipedia_page(query: str, lang: str):
    api = _api_base(lang)
    params = {"action":"query","list":"search","srsearch":query,"srlimit":5,"format":"json","srnamespace":0}
    r = _get(api, params)
    results = r.json().get("query",{}).get("search",[])
    if not results: return None, None, api
    top = results[0]
    return top["pageid"], html.unescape(top["title"]), api

def fetch_page_html_by_pageid(pageid: int, api: str) -> str:
    params = {"action":"parse","pageid":pageid,"prop":"text","format":"json","disableeditsection":1}
    r = _get(api, params)
    return r.json().get("parse",{}).get("text",{}).get("*","")

def wiki_search(query: str, lang: str, limit: int = 8) -> List[Dict]:
    api = _api_base(lang)
    params = {"action":"query","list":"search","format":"json","srlimit":limit,"srsearch":query,"srnamespace":0}
    r = _get(api, params)
    return r.json().get("query", {}).get("search", [])

def fetch_page_html_by_title(title: str, lang: str) -> str:
    api = _api_base(lang)
    params = {"action":"parse","page":title,"prop":"text","format":"json","disableeditsection":1}
    r = _get(api, params)
    return r.json().get("parse",{}).get("text",{}).get("*","")

def parse_infobox_years_from_html(html_content: str) -> Optional[Tuple[int,int]]:
    soup = BeautifulSoup(html_content, "html.parser")
    s, e = extract_infobox_years(soup)
    if not s: return None
    try:
        start = int(s)
        end = 9999 if (not e or str(e).lower() in PRESENT_WORDS) else int(e)
        return (start, end)
    except Exception:
        return None

def harvest_mk_subpages(model: str, lang: str) -> List[Dict[str,str]]:
    queries = [f'"{model}" Mk', f'"{model}" Mark', f'{model} generation Mk']
    seen = set(); windows: List[Dict[str,str]] = []
    for q in queries:
        for hit in wiki_search(q, lang=lang, limit=10):
            title = html.unescape(hit.get("title","")); low = title.lower()
            if title in seen: continue
            if not re.search(r'\b(mk|mark)\s*[ivxlcdm\d]{1,4}\b', low): continue
            if model.lower().split()[0] not in low: continue
            seen.add(title)
            try: html_page = fetch_page_html_by_title(title, lang)
            except Exception: continue
            rng = parse_infobox_years_from_html(html_page)
            if not rng:
                gens = parse_generations_from_html(html_page)
                if gens:
                    try:
                        s = int(gens[0]["launch_year"])
                        e_raw = gens[0]["end_year"]
                        e = 9999 if str(e_raw).lower()=="present" else int(e_raw)
                        rng = (s, e)
                    except Exception:
                        rng = None
            if rng:
                s, e = rng
                windows.append({
                    "label": title,
                    "launch_year": str(s),
                    "end_year": ("present" if e==9999 else str(e)),
                    "source": "mk_subpage_infobox"
                })
    # de-dup & sort
    out, seen2 = [], set()
    for w in windows:
        key = (w["launch_year"], w["end_year"])
        if key in seen2: continue
        seen2.add(key); out.append(w)
    out.sort(key=lambda w:int(w["launch_year"]))
    return out

# ================= Multi-language scrape orchestrator =================
LANG_ORDER = ["en","fr","de","es","it","pt"]

def _scrape_once(model: str, lang: str) -> List[Dict[str,str]]:
    """Scrape one language: model page + Mk subpages fallback."""
    pid, title, api = search_wikipedia_page(model, lang)
    if not pid:
        return []
    html_main = fetch_page_html_by_pageid(pid, api)
    gens = parse_generations_from_html(html_main)
    if len(gens) <= 1:
        # try Mk subpages (works for Mk-named gens)
        mk_win = harvest_mk_subpages(model, lang=lang)
        if mk_win:
            gens = mk_win
    if gens:
        infer_end_years_inplace(gens)
    return gens

def scrape_and_cache(model: str, year: int, lang: str = "en", cache_dir: str = CACHE_DIR):
    """Try preferred language first; if ≤1 window, try other languages and prefer the first with ≥2 windows."""
    try_order = [lang] + [l for l in LANG_ORDER if l != lang]
    best_lang, best_gens = None, []
    for lg in try_order:
        gens = _scrape_once(model, lg)
        if gens:
            if not best_gens:
                best_gens, best_lang = gens, lg
            # Prefer a language that yields multiple windows (true gen-splits)
            if len(gens) >= 2:
                best_gens, best_lang = gens, lg
                break
    if not best_gens:
        raise RuntimeError(f"No Wikipedia page/windows found for '{model}' in langs {try_order}")

    path = write_generation_cache(model, best_lang or lang, best_gens, cache_dir)
    return path, best_gens

# ================= Public API =================
def _pick_window_for_year(payload: dict, year: int) -> Optional[dict]:
    wins = payload.get("windows") or []
    if not wins: return None
    covering = [w for w in wins if w["start"] <= year <= (w["end"] if w["end"]!=9999 else 9999)]
    if covering:
        covering.sort(key=lambda w: (((w["end"] if w["end"]!=9999 else 9999) - w["start"]), -w["start"]))
        return covering[0]
    def dist(w):
        s, e = w["start"], (w["end"] if w["end"]!=9999 else 9999)
        if year < s: return s - year
        if year > e: return year - e
        return 0
    wins_sorted = sorted(wins, key=lambda w: (dist(w), -w["start"]))
    return wins_sorted[0] if wins_sorted else None

def _mk_label_from_text(label_text: str) -> str:
    m = re.search(r'\b(?:mk|mark|gen(?:eration)?)\s*([ivxlcdm\d]{1,4})\b', str(label_text), flags=re.I)
    if m: return f"Mk{m.group(1).upper()}"
    return "GEN"

def detect_via_wikipedia(model: str, year: int, lang: str = "en", cache_dir: str = CACHE_DIR) -> Tuple[Optional[str], Optional[Tuple[int,int]], dict]:
    """
    Returns (gen_label, (start,end), diag). If cache has ≤1 window, we re-scrape with multi-language
    strategy to split generations, then overwrite the cache.
    """
    cached = read_generation_cache(model, cache_dir)
    if not cached or len(cached.get("windows", [])) <= 1:
        try:
            scrape_and_cache(model, year, lang, cache_dir)
        except Exception as e:
            return None, None, {"basis":"wikipedia_gen","status":"scrape_failed","error":str(e)}
        cached = read_generation_cache(model, cache_dir)
        if not cached:
            return None, None, {"basis":"wikipedia_gen","status":"no_cache_after_scrape"}
        source = "scraped"
    else:
        source = "cache"

    pick = _pick_window_for_year(cached, year)
    if not pick:
        return None, None, {"basis":"wikipedia_gen","status":"no_window_in_cache","cache_model": cached.get("model")}

    start = int(pick["start"]); end = int(pick["end"])  # 9999 = present
    gen_label = _mk_label_from_text(pick.get("label",""))
    diag = {
        "basis": "wikipedia_gen",
        "source": source,
        "cache_lang": cached.get("lang"),
        "scraped_at": cached.get("scraped_at"),
        "picked_label": pick.get("label"),
        "picked_source": pick.get("source"),
        "windows_count": len(cached.get("windows", [])),
    }
    return gen_label, (start, end), diag

# ================= CLI =================
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
    path, gens = scrape_and_cache(args.model, args.year, args.lang, args.cache_dir)
    print(f"[cache] wrote: {path}")
    if os.environ.get("ICOR_DEBUG_CACHE") == "1":
        print(f"[wikipedia_gen] wrote {len(gens)} windows to {os.path.abspath(path)}", file=sys.stderr)
    if args.json:
        cached = json.load(open(path,"r",encoding="utf-8"))
        wins = cached["windows"]; yr = args.year
        covering = [w for w in wins if w["start"] <= yr <= (w["end"] if w["end"]!=9999 else 9999)]
        pick = covering[0] if covering else wins[0]
        print(json.dumps({
            "model": cached["model"], "label": pick["label"],
            "start": pick["start"], "end": pick["end"],
            "diag": {"basis":"cache","lang":cached["lang"],"scraped_at":cached["scraped_at"],
                     "windows_count": len(wins)}
        }))

if __name__ == "__main__":
    main()

__all__ = ["detect_via_wikipedia","scrape_and_cache","read_generation_cache"]

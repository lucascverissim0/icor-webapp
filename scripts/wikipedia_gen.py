#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wikipedia_gen.py
----------------
High-accuracy generation window detector used by script2.py.

Public API (what script2 imports):
    detect_via_wikipedia(model: str, user_year: int, lang: str = "en")
      -> (gen_label: str | None,
          window: tuple[int, int] | None,
          diag: dict)

- Reuses your robust scraping:
  * Descriptive User-Agent + retries (avoids 403).
  * Multilingual parsing (EN/FR/ES/DE/IT/PT).
  * Parses main page headings + infobox.
  * Discovers & parses Mk/“generation” subpages if the main page is too broad.
  * Normalizes open-ended ranges to "present" and infers missing end years
    as (next launch year - 1).
- Selection prefers per-generation/Mk windows over broad infobox spans.

You can also run this file directly as a CLI:
    python scripts/wikipedia_gen.py --year 2022 "Volkswagen Golf"
"""

from __future__ import annotations
import sys
import re
import html
import time
import json
import urllib.parse
import argparse
from typing import Optional, Tuple, List, Dict

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

# ---------- Constants ----------
DEADLINE_YEAR = 2035
REQUEST_DELAY_SEC = 0.2

# ---------- HTTP session (polite UA + retries) ----------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "ICOR-GenWindow/2.6 (+contact: your-email@your-domain.com)"
})
retries = Retry(
    total=4,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
SESSION.mount("https://", HTTPAdapter(max_retries=retries))
SESSION.mount("http://", HTTPAdapter(max_retries=retries))

def _get(url: str, params: dict) -> requests.Response:
    time.sleep(REQUEST_DELAY_SEC)
    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r

# ------------- Regexes & i18n -------------
PRESENT_WORDS = {
    "present", "current", "ongoing",
    "présent", "actuel", "en cours",
    "presente", "attuale", "oggi",
    "atual", "em curso",
    "aktuell", "heute", "bis heute"
}

# (2012–2019), (2012-2019), (2012–), (2012-)
PARENS_RANGE_FLEX_RE = re.compile(
    r"\((?:[^()]*?;?\s*)?(?P<start>\d{4})\s*[–—-]\s*(?P<end>\d{4}|"
    r"(?:present|current|ongoing|"
    r"présent|actuel|en cours|"
    r"presente|attuale|oggi|"
    r"atual|em curso|"
    r"aktuell|heute|bis heute"
    r"))?\)", re.IGNORECASE
)

# 2012–2019 or 2012–present (no parentheses)
YEAR_RANGE_FLEX_RE = re.compile(
    r"(?P<start>\b\d{4}\b)\s*[–—-]\s*(?P<end>\b\d{4}\b|"
    r"(?:present|current|ongoing|"
    r"présent|actuel|en cours|"
    r"presente|attuale|oggi|"
    r"atual|em curso|"
    r"aktuell|heute|bis heute"
    r"))", re.IGNORECASE
)

# Mk / Mark detection
MK_TEXT_RE = re.compile(r"\b(?:mk|mark)\s*(?:[ivxlcdm]+|\d+)\b", re.IGNORECASE)

# “generation” keywords (multi-lingual)
GEN_KEYWORDS = [
    "generation", "génération", "generación", "geração", "generazione"
]

# ---------- Helpers ----------
def _to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _is_open_end(token: str) -> bool:
    return (token or "").strip().lower() in PRESENT_WORDS

def normalize_dash(s: str) -> str:
    return (s or "").replace("—", "–").replace("-", "–")

def clean_text(s: str) -> str:
    return " ".join((s or "").split())

def heading_looks_like_generation(text: str) -> bool:
    low = text.lower()
    if MK_TEXT_RE.search(low): return True
    if any(kw in low for kw in GEN_KEYWORDS): return True
    if PARENS_RANGE_FLEX_RE.search(low): return True
    return False

# ---------- Wikipedia API wrapper ----------
def _api_base(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"

def search_wikipedia_page(model: str, lang: str) -> tuple[Optional[int], Optional[str]]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": model,
        "srlimit": 5,
        "format": "json",
        "srnamespace": 0,
    }
    r = _get(_api_base(lang), params)
    results = r.json().get("query", {}).get("search", [])
    if not results:
        return None, None

    # Filter out disambiguation pages
    pageids = [str(res["pageid"]) for res in results]
    titles = [html.unescape(res["title"]) for res in results]
    props_params = {
        "action": "query", "pageids": "|".join(pageids),
        "prop": "pageprops", "format": "json",
    }
    r2 = _get(_api_base(lang), props_params)
    pageinfo = r2.json().get("query", {}).get("pages", {})

    for pid, title in zip(pageids, titles):
        page = pageinfo.get(pid, {})
        if "disambiguation" not in page.get("pageprops", {}):
            return int(pid), title

    top = results[0]
    return top["pageid"], html.unescape(top["title"])

def get_pageid_by_title(title: str, lang: str) -> Optional[int]:
    params = {"action": "query", "titles": title, "format": "json"}
    r = _get(_api_base(lang), params)
    pages = r.json().get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if pid != "-1":
            return int(pid)
    return None

def fetch_page_html_by_pageid(pageid: int, lang: str) -> str:
    params = {
        "action": "parse",
        "pageid": pageid,
        "prop": "text",
        "format": "json",
        "disableeditsection": 1,
    }
    r = _get(_api_base(lang), params)
    return r.json().get("parse", {}).get("text", {}).get("*", "") or ""

# ---------- Parsing ----------
def extract_years_any(text: str) -> tuple[Optional[str], Optional[str]]:
    t = normalize_dash(text)
    m = PARENS_RANGE_FLEX_RE.search(t) or YEAR_RANGE_FLEX_RE.search(t)
    if not m:
        # also catch “since 2019 / seit / depuis”
        m2 = re.search(r"\b(depuis|seit|since)\s+(?P<start>\d{4})\b", t, re.IGNORECASE)
        if m2: return m2.group("start"), "present"
        return None, None
    start = m.group("start")
    end = m.group("end")
    if (not end) or _is_open_end(end):
        end = "present"
    return start, end

def extract_generation_headings(soup: BeautifulSoup) -> List[str]:
    return [
        tag.get_text(" ", strip=True)
        for tag in soup.find_all(["h2","h3","h4"])
        if heading_looks_like_generation(tag.get_text(" ", strip=True))
    ]

def extract_infobox_years(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
    if not infobox:
        return None, None
    for row in infobox.find_all("tr"):
        th, td = row.find("th"), row.find("td")
        if not th or not td:
            continue
        label = th.get_text(" ", strip=True).lower()
        if label in (
            "production","model years","production years",
            "années de production","años de producción",
            "produktionszeitraum","anni di produzione","anos de produção"
        ):
            return extract_years_any(td.get_text(" ", strip=True))
    return None, None

def parse_generations_from_html(html_content: str) -> List[Dict]:
    soup = BeautifulSoup(html_content, "html.parser")
    out, seen = [], set()

    # Headings first
    for heading in extract_generation_headings(soup):
        s, e = extract_years_any(heading)
        if s or e:
            key = (s or "", e or "", heading)
            if key in seen: continue
            seen.add(key)
            out.append({
                "source": "heading",
                "title_hint": None,
                "generation_heading": clean_text(heading),
                "launch_year": s,
                "end_year": e,
            })

    # Infobox fallback
    if not out:
        s, e = extract_infobox_years(soup)
        if s or e:
            out.append({
                "source": "infobox",
                "title_hint": None,
                "generation_heading": "Infobox production/model years",
                "launch_year": s,
                "end_year": e,
            })

    def sort_key(item):
        try: return int(item["launch_year"]) if item["launch_year"] else 99999
        except Exception: return 99999
    out.sort(sort_key)
    return out

# ---------- Subpage discovery ----------
def discover_generation_subpages_from_links(main_html: str, main_title: str) -> List[str]:
    soup = BeautifulSoup(main_html, "html.parser")
    content_root = soup.find(id="mw-content-text") or soup
    candidates = set()
    base = main_title.split("(")[0].strip().lower()

    for a in content_root.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        if not text: continue
        href = a["href"]
        if not href.startswith("/wiki/"): continue
        if any(prefix in href for prefix in ("/wiki/Help:","/wiki/Special:","/wiki/Talk:","/wiki/Category:","/wiki/File:","/wiki/Portal:","/wiki/Template:")):
            continue
        if heading_looks_like_generation(text):
            title = a.get("title") or urllib.parse.unquote(href.split("/wiki/")[-1]).replace("_"," ")
            if base.split()[0] in title.lower() or base in title.lower():
                candidates.add(title)

    return sorted(candidates)

def discover_generation_subpages_via_search(main_title: str, lang: str) -> List[str]:
    queries = [
        f'intitle:"{main_title}" intitle:Mk',
        f'intitle:"{main_title}" intitle:Mark',
        f'"{main_title}" Mk',
        f'"{main_title}" "first generation"',
        f'"{main_title}" "second generation"',
        f'"{main_title}" "première génération"',
        f'"{main_title}" "seconde génération"',
    ]
    titles = set()
    for q in queries:
        params = {"action":"query","list":"search","srsearch":q,"srlimit":20,"format":"json","srnamespace":0}
        r = _get(_api_base(lang), params)
        for hit in r.json().get("query",{}).get("search",[]):
            title = html.unescape(hit["title"])
            low = title.lower()
            if (" mk" in low or "mk" in low or "génération" in low or "generation" in low) and main_title.split("(")[0].strip().split()[0].lower() in low:
                titles.add(title)
    return sorted(titles)

# ---------- Inference & selection ----------
def infer_end_years_inplace(items: List[Dict]) -> None:
    items.sort(key=lambda g: _to_int(g.get("launch_year")) or 99999)
    for i in range(len(items)-1):
        curr, nxt = items[i], items[i+1]
        s_curr = _to_int(curr.get("launch_year"))
        e_curr = _to_int(curr.get("end_year"))
        s_next = _to_int(nxt.get("launch_year"))
        if not s_curr or not s_next: continue
        needs = (curr.get("end_year") in (None,"","present")) or (e_curr is not None and e_curr >= s_next)
        if needs:
            curr["end_year"] = str(max(s_curr, s_next-1))
            curr["end_year_inferred"] = True

def year_in_span(year: int, start_str, end_str) -> bool:
    s = _to_int(start_str)
    e = _to_int(end_str) if (end_str and not _is_open_end(end_str)) else 9999
    return s is not None and s <= year <= e

def _label_from_text(txt: str) -> Optional[str]:
    m = re.search(r'\b(?:mk|mark)\s*([ivxlcdm]+|\d+)\b', txt or "", re.IGNORECASE)
    if not m: return None
    val = m.group(1)
    return f"Mk{val.upper()}"

# ---------- Public API ----------
def detect_via_wikipedia(model: str, user_year: int, lang: str = "en") -> tuple[Optional[str], Optional[tuple[int,int]], Dict]:
    """
    Return (gen_label, (start,end), diag). End is numeric; 'present' -> DEADLINE_YEAR.
    """
    base_url = _api_base(lang)
    # 1) main page
    pageid, title = search_wikipedia_page(model, lang)
    if not pageid:
        return None, None, {"basis":"wikipedia","status":"no_results","query":model,"lang":lang}

    main_html = fetch_page_html_by_pageid(pageid, lang)
    results = parse_generations_from_html(main_html)

    # 2) if only broad infobox span, pull subpages
    only_infobox = (len(results)==1 and results[0]["source"]=="infobox")
    subpages_merged = []
    if (not results) or only_infobox:
        link_titles   = discover_generation_subpages_from_links(main_html, title)
        search_titles = discover_generation_subpages_via_search(title, lang)
        sub_titles    = sorted(set(link_titles) | set(search_titles))
        for t in sub_titles:
            pid = get_pageid_by_title(t, lang)
            if not pid: continue
            sub_html = fetch_page_html_by_pageid(pid, lang)
            gens = parse_generations_from_html(sub_html)
            for g in gens:
                g["title_hint"] = t
                results.append(g)
            if gens:
                subpages_merged.append(t)

    if not results:
        return None, None, {"basis":"wikipedia","status":"no_windows","title":title,"lang":lang}

    # Deduplicate + order + infer end years
    seen, dedup = set(), []
    for g in results:
        key = ((g.get("launch_year") or ""), (g.get("end_year") or ""), (g.get("title_hint") or ""), (g.get("generation_heading") or ""))
        if key in seen: continue
        seen.add(key); dedup.append(g)
    dedup.sort(key=lambda it: _to_int(it.get("launch_year")) or 99999)
    infer_end_years_inplace(dedup)

    # Prefer generation/Mk items for the given year
    def is_infobox_item(g: Dict) -> bool:
        return (g.get("source") or "").lower() == "infobox"

    covering_heads = [g for g in dedup if not is_infobox_item(g) and year_in_span(user_year, g.get("launch_year"), g.get("end_year"))]
    covering_any   = [g for g in dedup if year_in_span(user_year, g.get("launch_year"), g.get("end_year"))]

    pick = None
    if covering_heads:
        covering_heads.sort(key=lambda g: ((_to_int(g.get("end_year")) or 9999) - (_to_int(g.get("launch_year")) or 9999), -(_to_int(g.get("launch_year")) or 0)))
        pick = covering_heads[0]
    elif covering_any:
        covering_any.sort(key=lambda g: ((_to_int(g.get("end_year")) or 9999) - (_to_int(g.get("launch_year")) or 9999), -(_to_int(g.get("launch_year")) or 0)))
        pick = covering_any[0]
    else:
        # nearest-by-gap fallback (prefer non-infobox)
        def dist(g: Dict) -> int:
            s = _to_int(g.get("launch_year")) or 9999
            e = _to_int(g.get("end_year")) if (g.get("end_year") and not _is_open_end(g.get("end_year"))) else 9999
            if user_year < s: return s - user_year
            if user_year > e: return user_year - e
            return 0
        heads = [g for g in dedup if not is_infobox_item(g)]
        pool = heads or dedup
        pool.sort(key=lambda g: (dist(g), -(_to_int(g.get("launch_year")) or 0)))
        pick = pool[0]

    s = _to_int(pick.get("launch_year"))
    e = _to_int(pick.get("end_year")) if (pick.get("end_year") and not _is_open_end(pick.get("end_year"))) else DEADLINE_YEAR
    heading = pick.get("title_hint") or pick.get("generation_heading") or ""
    gen_label = _label_from_text(heading) or _label_from_text(title) or "GEN"

    diag = {
        "basis": "wikipedia",
        "title": title,
        "lang": lang,
        "subpages_merged": subpages_merged,
        "windows_found": [
            {"start": g.get("launch_year"), "end": g.get("end_year"),
             "label": g.get("title_hint") or g.get("generation_heading"),
             "source": g.get("source")} for g in dedup
        ],
        "picked": {"start": s, "end": e, "label_text": heading, "gen_label": gen_label},
        "note": f"Active selection for {user_year} (generation pages preferred)."
    }
    return gen_label, (s, e), diag

# ---------- Optional CLI (for debugging) ----------
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", help='Car model, e.g. "Volkswagen Golf"')
    p.add_argument("--year", type=int, required=True, help="Target year, e.g. 2022")
    p.add_argument("--lang", default="en", help="Wikipedia language (default: en)")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    label, window, diag = detect_via_wikipedia(args.model, args.year, args.lang)
    print(json.dumps({
        "model": args.model,
        "year": args.year,
        "label": label,
        "window": window,
        "diag": diag
    }, indent=2, ensure_ascii=False))

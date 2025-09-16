#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wikipedia_gen.py
----------------
High-accuracy generation window detector for car models using Wikipedia.

Public API:
    detect_via_wikipedia(model: str, user_year: int)
        -> (gen_label: str|None, window: (start:int,end:int)|None, diag: dict)

Key behavior:
- Uses a descriptive User-Agent (avoids 403).
- Prefers the model's main page (not per-generation) when searching.
- Parses BOTH generation sections and the page infobox.
- NEW: Looks into the text **after each generation/Mk heading** to find year
  ranges like "since 2019" when they are not inside the heading itself.
- Selector **prefers generation/Mk windows**; only falls back to infobox if
  no generation window covers `user_year`.
"""

from __future__ import annotations
import re
import html
import json
from typing import Optional, Tuple, List, Dict

import requests
from bs4 import BeautifulSoup, Tag

# ------------------ Config -------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
SEARCH_TIMEOUT = 20
DEADLINE_YEAR = 2035  # caller may clip further

# Wikipedia requires a descriptive UA
WIKI_HEADERS = {
    "User-Agent": "ICOR-GenWindow/1.2 (+contact: your-email@your-domain.com)"
}

# ------------------ Regexes ------------------------
YEAR_RANGE_RE = re.compile(
    r"(?P<start>\b\d{4}\b)\s*[–—-]\s*(?P<end>\b\d{4}\b|present|current|ongoing|to\s+present)",
    re.IGNORECASE,
)
PARENS_RANGE_RE = re.compile(
    r"\((?:[^()]*?;?\s*)?(?P<start>\d{4})\s*[–—-]\s*(?P<end>\d{4}|present|current|ongoing|to\s+present)\)",
    re.IGNORECASE,
)
SINCE_RE = re.compile(r"\bsince\s+(?P<start>\d{4})\b", re.IGNORECASE)
FROM_TO_RE = re.compile(
    r"\bfrom\s+(?P<start>\d{4})\s+(?:to|-|until)\s+(?P<end>\d{4}|present|current|ongoing)\b",
    re.IGNORECASE,
)
GEN_TOKEN_RE = re.compile(
    r'\b(mk\s*\d+|mark\s*\d+|gen(?:eration)?\s*\w+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b',
    re.IGNORECASE,
)
YEAR_TOKEN_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')

# ------------------ Helpers ------------------------
def _normalize_dash(s: str) -> str:
    return s.replace("—", "–").replace("-", "–")

def _clean_text(s: str) -> str:
    return " ".join((s or "").split())

def _to_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _is_open_end(token: str) -> bool:
    return token.lower() in {"present", "current", "ongoing", "to present"}

def _looks_like_generation_title(title: str) -> bool:
    t = title.lower()
    if re.search(r'\bmk\s*\d+\b', t): return True
    if "generation" in t: return True
    if re.search(r'\([^)0-9]*\d{4}', t): return True
    return False

def _sanitize_query(model: str) -> str:
    s = model.strip()
    s = GEN_TOKEN_RE.sub("", s)
    s = YEAR_TOKEN_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" -–—")
    return s

# ------------------ Wikipedia fetchers --------------
def _search_wikipedia_page(query: str) -> tuple[Optional[int], Optional[str]]:
    q = _sanitize_query(query)
    params = {
        "action": "query",
        "list": "search",
        "format": "json",
        "srlimit": 5,
        "srsearch": f'intitle:"{q}" {q}',
        "srnamespace": 0,
    }
    r = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=SEARCH_TIMEOUT)
    r.raise_for_status()
    results = r.json().get("query", {}).get("search", [])
    if not results:
        return None, None
    for res in results:
        title = res.get("title", "")
        if title and not _looks_like_generation_title(title):
            return res.get("pageid"), html.unescape(title)
    top = results[0]
    return top.get("pageid"), html.unescape(top.get("title", ""))

def _parse_page_html(pageid: int) -> str:
    params = {
        "action": "parse",
        "pageid": pageid,
        "prop": "text",
        "format": "json",
        "disableeditsection": 1,
    }
    r = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=SEARCH_TIMEOUT)
    r.raise_for_status()
    return r.json().get("parse", {}).get("text", {}).get("*", "") or ""

# ------------------ Extractors ----------------------
def _extract_infobox_years(soup: BeautifulSoup) -> List[tuple[int, int, str]]:
    windows: List[tuple[int, int, str]] = []
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
    if not infobox:
        return windows

    def parse_value_text(text: str) -> List[tuple[int, int]]:
        text = _normalize_dash(html.unescape(text or ""))
        wins: List[tuple[int, int]] = []

        for m in PARENS_RANGE_RE.finditer(text):
            s = _to_int(m.group("start")); e_s = m.group("end")
            if not s: continue
            e = DEADLINE_YEAR if _is_open_end(e_s) else (_to_int(e_s) or DEADLINE_YEAR)
            if 1900 <= s <= 2100 and 1900 <= e <= 2100 and s <= e:
                wins.append((s, e))

        for m in YEAR_RANGE_RE.finditer(text):
            s = _to_int(m.group("start")); e_s = m.group("end")
            if not s: continue
            e = DEADLINE_YEAR if _is_open_end(e_s) else (_to_int(e_s) or DEADLINE_YEAR)
            if 1900 <= s <= 2100 and 1900 <= e <= 2100 and s <= e:
                wins.append((s, e))
        return wins

    for row in infobox.find_all("tr"):
        th = row.find("th")
        if not th: continue
        label = _clean_text(th.get_text(" ", strip=True))
        ll = label.lower()
        if "production" in ll or "model years" in ll:
            td = row.find("td")
            if not td: continue
            val_text = _clean_text(td.get_text(" ", strip=True))
            for (s, e) in parse_value_text(val_text):
                windows.append((s, e, f"Infobox {label}"))

    return windows

def _scan_following_for_years(h: Tag, max_nodes: int = 10) -> Optional[tuple[int,int]]:
    """
    Look at the first few siblings after a heading until the next heading of the
    same or higher level; try to spot year ranges or 'since YYYY'.
    """
    # Determine levels (h2=2, h3=3, h4=4)
    level = int(h.name[-1]) if h.name and h.name[-1].isdigit() else 7
    count = 0
    for sib in h.next_siblings:
        if isinstance(sib, Tag):
            if sib.name in ("h2", "h3", "h4"):
                # Stop at next heading of same or higher hierarchy
                nxt_level = int(sib.name[-1])
                if nxt_level <= level:
                    break
            text = _normalize_dash(_clean_text(sib.get_text(" ", strip=True)))
            if not text:
                continue
            # Try patterns
            m = PARENS_RANGE_RE.search(text) or YEAR_RANGE_RE.search(text)
            if m:
                s = _to_int(m.group("start")); e_s = m.group("end")
                if s:
                    e = DEADLINE_YEAR if _is_open_end(e_s) else (_to_int(e_s) or DEADLINE_YEAR)
                    return (s, e)
            m = FROM_TO_RE.search(text)
            if m:
                s = _to_int(m.group("start")); e_s = m.group("end")
                if s:
                    e = DEADLINE_YEAR if _is_open_end(e_s) else (_to_int(e_s) or DEADLINE_YEAR)
                    return (s, e)
            m = SINCE_RE.search(text)
            if m:
                s = _to_int(m.group("start"))
                if s:
                    return (s, DEADLINE_YEAR)
            count += 1
            if count >= max_nodes:
                break
    return None

def _extract_generation_sections(soup: BeautifulSoup) -> List[tuple[int, int, str]]:
    """
    Gather windows around headings that look like "generation" or "MkN".
    Looks in the heading AND the immediate section content after it.
    """
    windows: List[tuple[int, int, str]] = []
    for tag in soup.find_all(["h2", "h3", "h4"]):
        heading = tag.get_text(" ", strip=True)
        h_norm = _normalize_dash(heading)
        lower = h_norm.lower()
        if ("generation" in lower) or re.search(r"\bmk\s*\d+\b", lower):
            # Inside heading
            m = PARENS_RANGE_RE.search(h_norm) or YEAR_RANGE_RE.search(h_norm)
            if m:
                s = _to_int(m.group("start")); e_s = m.group("end")
                if s:
                    e = DEADLINE_YEAR if _is_open_end(e_s) else (_to_int(e_s) or DEADLINE_YEAR)
                    if 1900 <= s <= 2100 and 1900 <= e <= 2100 and s <= e:
                        windows.append((s, e, _clean_text(heading)))
                        continue
            # Not in heading → look just after it
            se = _scan_following_for_years(tag)
            if se:
                windows.append((se[0], se[1], _clean_text(heading)))
    return windows

def _label_from_heading(heading: str) -> Optional[str]:
    m = re.search(r'\b(?:mk|mark|gen(?:eration)?)\s*([ivx\d]{1,3})\b', heading, flags=re.IGNORECASE)
    if m:
        return f"Mk{m.group(1).upper()}"
    ord_map = {
        "first":"Mk1","second":"Mk2","third":"Mk3","fourth":"Mk4","fifth":"Mk5",
        "sixth":"Mk6","seventh":"Mk7","eighth":"Mk8","ninth":"Mk9","tenth":"Mk10"
    }
    for k,v in ord_map.items():
        if re.search(rf'\b{k}\b', heading, re.IGNORECASE):
            return v
    return None

# ------------------ Selection logic -----------------
def _merge_and_dedupe(windows: List[tuple[int,int,str]]) -> List[tuple[int,int,str]]:
    seen = set()
    out: List[tuple[int,int,str]] = []
    for s,e,h in windows:
        key = (s,e,_clean_text(h))
        if key in seen: continue
        seen.add(key)
        out.append((s,e,_clean_text(h)))
    out.sort(key=lambda t: (t[0], t[1]))
    return out

def _select_window_for_year(windows: List[tuple[int,int,str]], user_year: int) -> tuple[int,int,str]:
    """
    Prefer generation/Mk sections over infobox windows.
    """
    def is_infobox(h: str) -> bool:
        return h.lower().startswith("infobox")

    covering_head = [(s,e,h) for (s,e,h) in windows
                     if not is_infobox(h) and (s - 1) <= user_year <= (e + 1)]
    covering_any  = [(s,e,h) for (s,e,h) in windows
                     if (s - 1) <= user_year <= (e + 1)]

    if covering_head:
        covering_head.sort(key=lambda t: ((t[1]-t[0]), -t[0]))
        return covering_head[0]

    if covering_any:
        covering_any.sort(key=lambda t: ((t[1]-t[0]), -t[0]))
        return covering_any[0]

    # Nearest-by-gap fallback; still prefer headings
    def dist(s,e):
        if user_year < s: return s - user_year
        if user_year > e: return user_year - e
        return 0
    heads = [(s,e,h) for (s,e,h) in windows if not is_infobox(h)]
    pool = heads or windows
    pool.sort(key=lambda t: (dist(t[0], t[1]), -t[0]))
    return pool[0]

# ------------------ Public API ----------------------
def find_generation_windows(model: str) -> tuple[Optional[str], List[tuple[int,int,str]], Dict]:
    pageid, title = _search_wikipedia_page(model)
    if not pageid:
        return None, [], {"basis":"wikipedia_search","status":"no_results","query":model}

    html_primary = _parse_page_html(pageid)
    soup_primary = BeautifulSoup(html_primary, "html.parser")

    windows: List[tuple[int,int,str]] = []
    # Prefer generation sections; append infobox later
    windows.extend(_extract_generation_sections(soup_primary))
    windows.extend(_extract_infobox_years(soup_primary))

    # If first page looks like a per-generation page, merge likely main page
    extra_title = None
    extra_added = 0
    if _looks_like_generation_title(title):
        base = _sanitize_query(model)
        pid2, title2 = _search_wikipedia_page(base)
        if pid2 and title2 and title2 != title and not _looks_like_generation_title(title2):
            html_extra = _parse_page_html(pid2)
            soup_extra = BeautifulSoup(html_extra, "html.parser")
            w0 = len(windows)
            windows.extend(_extract_generation_sections(soup_extra))
            windows.extend(_extract_infobox_years(soup_extra))
            extra_title = title2
            extra_added = len(windows) - w0

    windows = _merge_and_dedupe(windows)

    diag = {
        "basis": "wikipedia",
        "title": title,
        "extra_title": extra_title,
        "extra_added": extra_added,
        "windows_found": [{"start": s, "end": e, "label": h} for (s,e,h) in windows],
    }
    return title, windows, diag

def detect_via_wikipedia(model: str, user_year: int) -> tuple[Optional[str], Optional[tuple[int,int]], Dict]:
    title, wins, diag0 = find_generation_windows(model)
    if not wins:
        return None, None, {**diag0, "status":"no_windows"}

    s, e, heading = _select_window_for_year(wins, user_year)
    gen_label = _label_from_heading(heading) or "GEN"
    start, end = int(s), int(e)

    diag = {
        **diag0,
        "picked": {"start": start, "end": end, "heading": heading, "label": gen_label},
        "note": f"Active selection for {user_year} (generation sections preferred).",
    }
    return gen_label, (start, end), diag

# ------------------ CLI debug -----------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python scripts/wikipedia_gen.py \"Volkswagen Golf\" 2022")
        sys.exit(1)
    model = sys.argv[1]
    year = int(sys.argv[2])
    label, window, diag = detect_via_wikipedia(model, year)
    print(json.dumps({
        "model": model,
        "year": year,
        "label": label,
        "window": window,
        "diag": diag
    }, indent=2, ensure_ascii=False))

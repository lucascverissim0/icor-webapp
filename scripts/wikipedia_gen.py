#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wikipedia_gen.py
----------------
High-accuracy generation window detector for car models using Wikipedia.

Public API:
    detect_via_wikipedia(model: str, user_year: int) -> tuple[str|None, tuple[int,int]|None, dict]

Returns:
    gen_label: str | None        e.g., "Mk8" (best-effort), or "GEN" if unknown
    window:    (start, end) | None  inclusive window (end is numeric; 'present' resolved to a year)
    diag:      dict              diagnostics for logging

Behavior:
- Prefers the model's main Wikipedia page (sanitizes query).
- Parses both infobox and generation headings.
- If the first hit is a per-generation page, also fetches the main page and merges.
- Chooses the window **covering user_year**; if none covers it, chooses the nearest.
- Handles "present/current/ongoing" ends by returning a large sentinel (caller may refine).
- Robust to minor formatting differences (em/en dashes, hyphens).

Requires: requests, bs4 (BeautifulSoup4)
"""

from __future__ import annotations
import re
import html
import json
from typing import Optional, Tuple, List, Dict

import requests
from bs4 import BeautifulSoup

# ------------------ Config -------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"
SEARCH_TIMEOUT = 20
DEADLINE_YEAR = 2035  # callers may clip/resolve further

# IMPORTANT: Wikipedia requires a descriptive User-Agent.
WIKI_HEADERS = {
    "User-Agent": "ICOR-GenWindow/1.0 (+contact: your-email@your-domain.com)"
}

# ------------------ Regexes ------------------------
# Year ranges: "1998–2005", "1998-2005", "1998—2005", "1998–present/current/ongoing"
YEAR_RANGE_RE = re.compile(
    r"(?P<start>\b\d{4}\b)\s*[–—-]\s*(?P<end>\b\d{4}\b|present|current|ongoing|to\s+present)",
    re.IGNORECASE,
)

# In parentheses like "(E10; 1966–1970)" or "(1966–1970)"
PARENS_RANGE_RE = re.compile(
    r"\((?:[^()]*?;?\s*)?(?P<start>\d{4})\s*[–—-]\s*(?P<end>\d{4}|present|current|ongoing|to\s+present)\)",
    re.IGNORECASE,
)

GEN_TOKEN_RE = re.compile(
    r'\b(mk\s*\d+|mark\s*\d+|gen(?:eration)?\s*\w+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b',
    re.IGNORECASE,
)
YEAR_TOKEN_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')

# ------------------ Helpers ------------------------
def _normalize_dash(s: str) -> str:
    # normalize em dash and hyphen to en dash for consistent parsing
    return s.replace("—", "–").replace("-", "–")

def _clean_text(s: str) -> str:
    return " ".join((s or "").split())

def _to_int_safe(x: str) -> Optional[int]:
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
    # Heuristic for pages with years/chassis codes in parentheses
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
    """
    Prefer a main model page (non per-generation). If not found, fall back to top hit.
    """
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
    # Prefer a result that doesn't look like a per-generation page
    for res in results:
        title = res.get("title", "")
        if title and not _looks_like_generation_title(title):
            return res.get("pageid"), html.unescape(title)
    # Else take the top result
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
    """
    Look into infobox rows for 'Production', 'Model years'.
    Return list of (start, end, label) windows.
    """
    windows: List[tuple[int, int, str]] = []

    # Find the infobox (class names vary a bit on Wikipedia)
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
    if not infobox:
        return windows

    def parse_value_text(text: str) -> List[tuple[int, int]]:
        text = _normalize_dash(html.unescape(text or ""))
        wins: List[tuple[int, int]] = []
        # Try parentheses pattern first
        for m in PARENS_RANGE_RE.finditer(text):
            start = _to_int_safe(m.group("start"))
            end_s = m.group("end")
            if not start:
                continue
            if _is_open_end(end_s):
                end = DEADLINE_YEAR
            else:
                end = _to_int_safe(end_s) or DEADLINE_YEAR
            if 1900 <= start <= 2100 and 1900 <= end <= 2100 and start <= end:
                wins.append((start, end))
        # Then top-level ranges
        for m in YEAR_RANGE_RE.finditer(text):
            start = _to_int_safe(m.group("start"))
            end_s = m.group("end")
            if not start:
                continue
            if _is_open_end(end_s):
                end = DEADLINE_YEAR
            else:
                end = _to_int_safe(end_s) or DEADLINE_YEAR
            if 1900 <= start <= 2100 and 1900 <= end <= 2100 and start <= end:
                wins.append((start, end))
        return wins

    # Scan all rows
    for row in infobox.find_all("tr"):
        th = row.find("th")
        if not th:
            continue
        label = _clean_text(th.get_text(" ", strip=True))
        label_lower = label.lower()
        if "production" in label_lower or "model years" in label_lower:
            td = row.find("td")
            if not td:
                continue
            val_text = _clean_text(td.get_text(" ", strip=True))
            for (s, e) in parse_value_text(val_text):
                windows.append((s, e, f"{label}"))

    return windows

def _extract_generation_headings(soup: BeautifulSoup) -> List[tuple[int, int, str]]:
    """
    Find headings that denote generations, e.g.:
        <h2>First generation (1996–2001)</h2>
        <h3>Volkswagen Golf Mk8 (2019–present)</h3>
    Return list of (start, end, heading_text).
    """
    windows: List[tuple[int, int, str]] = []
    for tag in soup.find_all(["h2", "h3", "h4"]):
        heading = tag.get_text(" ", strip=True)
        h_norm = _normalize_dash(heading)
        lower = h_norm.lower()
        if ("generation" in lower) or re.search(r"\bmk\s*\d+\b", lower):
            # Try parentheses first
            m = PARENS_RANGE_RE.search(h_norm) or YEAR_RANGE_RE.search(h_norm)
            if not m:
                continue
            start = _to_int_safe(m.group("start"))
            end_s = m.group("end")
            if not start:
                continue
            if _is_open_end(end_s):
                end = DEADLINE_YEAR
            else:
                end = _to_int_safe(end_s) or DEADLINE_YEAR
            if 1900 <= start <= 2100 and 1900 <= end <= 2100 and start <= end:
                windows.append((start, end, _clean_text(heading)))
    return windows

def _label_from_heading(heading: str) -> Optional[str]:
    """
    Try to synthesize a human-friendly generation label (e.g., 'Mk8').
    """
    m = re.search(r'\b(?:mk|mark|gen(?:eration)?)\s*([ivx\d]{1,3})\b', heading, flags=re.IGNORECASE)
    if m:
        token = m.group(1)
        try:
            # Roman numerals or digits — just echo uppercased
            return f"Mk{token.upper()}"
        except Exception:
            pass
    # Fallback: First/Second/Third...
    ord_map = {
        "first":"Mk1","second":"Mk2","third":"Mk3","fourth":"Mk4","fifth":"Mk5",
        "sixth":"Mk6","seventh":"Mk7","eighth":"Mk8","ninth":"Mk9","tenth":"Mk10"
    }
    for k,v in ord_map.items():
        if re.search(rf'\b{k}\b', heading, re.IGNORECASE):
            return v
    return None

# ------------------ Main logic ----------------------
def _merge_and_dedupe(windows: List[tuple[int,int,str]]) -> List[tuple[int,int,str]]:
    """
    Deduplicate by (start,end,heading) text.
    """
    seen = set()
    out: List[tuple[int,int,str]] = []
    for s,e,h in windows:
        key = (s,e,_clean_text(h))
        if key in seen:
            continue
        seen.add(key)
        out.append((s,e,_clean_text(h)))
    # Sort by launch year
    out.sort(key=lambda t: (t[0], t[1]))
    return out

def _select_window_for_year(windows: List[tuple[int,int,str]], user_year: int) -> tuple[int,int,str]:
    """
    Pick the window covering user_year; else nearest by gap (ties → latest start).
    """
    covering = [(s,e,h) for (s,e,h) in windows if (s - 1) <= user_year <= (e + 1)]
    if covering:
        covering.sort(key=lambda t: ((t[1]-t[0]), -t[0]))  # narrowest first, then latest start
        return covering[0]
    # No covering window; choose nearest
    def dist(s,e):
        if user_year < s: return s - user_year
        if user_year > e: return user_year - e
        return 0
    windows_sorted = sorted(windows, key=lambda t: (dist(t[0], t[1]), -t[0]))
    return windows_sorted[0]

def find_generation_windows(model: str) -> tuple[Optional[str], List[tuple[int,int,str]], Dict]:
    """
    Fetch Wikipedia page(s) for model and extract generation windows.
    Returns:
        title:   page title chosen
        windows: list of (start, end, heading/label) tuples
        diag:    diagnostics (pages used, which sections hit, etc.)
    """
    pageid, title = _search_wikipedia_page(model)
    if not pageid:
        return None, [], {"basis":"wikipedia_search","status":"no_results","query":model}

    html_primary = _parse_page_html(pageid)
    soup_primary = BeautifulSoup(html_primary, "html.parser")

    windows: List[tuple[int,int,str]] = []
    # Infobox first (high signal)
    windows.extend([(s,e,f"Infobox {lbl.lower()}") for (s,e,lbl) in _extract_infobox_years(soup_primary)])
    # Headings (per-generation sections)
    windows.extend(_extract_generation_headings(soup_primary))

    # If this looks like a per-generation page, also fetch a likely main page and merge
    extra_title = None
    extra_added = 0
    if _looks_like_generation_title(title):
        base = _sanitize_query(model)
        pid2, title2 = _search_wikipedia_page(base)
        if pid2 and title2 and title2 != title and not _looks_like_generation_title(title2):
            html_extra = _parse_page_html(pid2)
            soup_extra = BeautifulSoup(html_extra, "html.parser")
            w0 = len(windows)
            windows.extend([(s,e,f"Infobox {lbl.lower()}") for (s,e,lbl) in _extract_infobox_years(soup_extra)])
            windows.extend(_extract_generation_headings(soup_extra))
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
    """
    Public entry point.
    - Finds & merges generation windows from Wikipedia.
    - Selects the window covering 'user_year' (or nearest).
    - Returns a best-effort generation label and the numeric window.
    """
    title, wins, diag0 = find_generation_windows(model)
    if not wins:
        return None, None, {**diag0, "status":"no_windows"}

    # Select the relevant window
    s, e, heading = _select_window_for_year(wins, user_year)

    # Generate a friendly label
    gen_label = _label_from_heading(heading) or "GEN"

    # Ensure inclusive numeric end; present/current represented as DEADLINE_YEAR here
    start, end = int(s), int(e)

    diag = {
        **diag0,
        "picked": {"start": start, "end": end, "heading": heading, "label": gen_label},
        "note": f"Active selection for {user_year}",
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

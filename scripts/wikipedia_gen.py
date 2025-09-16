#!/usr/bin/env python3
"""
Reusable Wikipedia generation-window detector.

Purpose
- Parse Wikipedia to find generation windows for a car model.
- Pick the window most relevant to a 'view_year'.
- Return a friendly generation label when possible (e.g., Mk7).

Intended to be imported by scripts/script2.py, not used as a CLI.
"""

from typing import List, Tuple, Dict, Any, Optional
import re, requests
from bs4 import BeautifulSoup

WIKI_API = "https://en.wikipedia.org/w/api.php"
DEADLINE_YEAR = 2035

# Year range regex: 1998–2005 / 1998-2005 / 1998—2005 / 1998–present …
YEAR_RANGE_RE = re.compile(
    r"(?P<start>\b\d{4}\b)\s*[–—-]\s*(?P<end>\b\d{4}\b|present|current|ongoing|to\s+present)",
    re.IGNORECASE,
)
# Headings like “First generation (E10; 1966–1970)”
PARENS_RANGE_RE = re.compile(
    r"\((?:[^()]*?;?\s*)?(?P<start>\d{4})\s*[–—-]\s*(?P<end>\d{4}|present|current|ongoing)\)",
    re.IGNORECASE,
)

# Prefer the *main model* page over single-generation pages
GEN_TOKENS_RE = re.compile(r'\b(mk\s*\d+|mark\s*\d+|gen(?:eration)?\s*\w+)\b', re.I)
YEAR_TOKEN_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')

def _sanitize_query(q: str) -> str:
    s = GEN_TOKENS_RE.sub('', q)
    s = YEAR_TOKEN_RE.sub('', s)
    s = re.sub(r'\s+', ' ', s).strip(' -–—')
    return s

def _looks_like_generation_page(title: str) -> bool:
    t = title.lower()
    if re.search(r'\bmk\s*\d+\b', t): return True
    if 'generation' in t: return True
    if re.search(r'\([^)0-9]*\d{4}', t): return True  # titles with years in parens
    return False

def _normalize_dash(s: str) -> str:
    return s.replace("—", "–").replace("-", "–")

def _clean(s: str) -> str:
    return " ".join((s or "").split())

def _extract_years_from_heading(heading: str) -> Tuple[Optional[str], Optional[str]]:
    m = PARENS_RANGE_RE.search(heading) or YEAR_RANGE_RE.search(heading)
    if not m: return None, None
    start = m.group("start")
    end = m.group("end").lower()
    if end.startswith("to"):
        end = "present"
    return start, end

def _search_wikipedia_page(query: str) -> Tuple[Optional[int], Optional[str]]:
    """Prefer main model pages; fall back to top hit."""
    base_q = _sanitize_query(query)
    params = {
        "action": "query",
        "list": "search",
        "format": "json",
        "srlimit": 5,
        "srsearch": f'intitle:"{base_q}" {base_q}',
        "srnamespace": 0,
    }
    r = requests.get(WIKI_API, params=params, timeout=20)
    r.raise_for_status()
    results = r.json().get("query", {}).get("search", [])
    if not results:
        return None, None
    for res in results:
        title = res.get("title", "")
        if title and not _looks_like_generation_page(title):
            return res["pageid"], title
    top = results[0]
    return top["pageid"], top["title"]

def _fetch_page_html(pageid: int) -> str:
    params = {
        "action": "parse",
        "pageid": pageid,
        "prop": "text",
        "format": "json",
        "disableeditsection": 1,
    }
    r = requests.get(WIKI_API, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("parse", {}).get("text", {}).get("*", "") or ""

def _parse_generations_from_html(html_content: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html_content, "html.parser")
    gens = []

    # Prefer explicit generation headings, but allow plain year-range headings too.
    for tag in soup.find_all(["h2", "h3", "h4"]):
        heading = tag.get_text(" ", strip=True)
        lower = heading.lower()
        if ("generation" in lower) or re.search(r"\bmk\s*\d+\b", lower) or YEAR_RANGE_RE.search(heading):
            gens.append(heading)

    out = []
    seen = set()
    for heading in gens:
        heading_norm = _normalize_dash(heading)
        start, end = _extract_years_from_heading(heading_norm)
        if start or end:
            key = (start or "", end or "", heading_norm)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "generation_heading": _clean(heading_norm),
                "launch_year": start,
                "end_year": (None if not end else ("present" if end.lower() in {"present","current","ongoing"} else end)),
            })

    def sort_key(item):
        try:
            return int(item["launch_year"]) if item["launch_year"] else 99999
        except Exception:
            return 99999

    out.sort(key=sort_key)
    return out

def find_generation_windows(model: str) -> Tuple[str, List[Tuple[int,int,str]]]:
    """
    Return (page_title, windows), where windows is a list of (start, end, heading).
    Open-ended 'present' becomes DEADLINE_YEAR; we still expect caller to "resolve" it.
    """
    pageid, title = _search_wikipedia_page(model)
    if not pageid:
        return "", []
    html = _fetch_page_html(pageid)
    gens = _parse_generations_from_html(html)
    windows: List[Tuple[int,int,str]] = []
    for g in gens:
        try:
            s = int(g["launch_year"]) if g["launch_year"] else None
        except Exception:
            s = None
        end_raw = (g.get("end_year") or "").lower()
        if end_raw in {"present","current","ongoing"} or not end_raw:
            e = DEADLINE_YEAR
        else:
            try:
                e = int(end_raw)
            except Exception:
                e = DEADLINE_YEAR
        if s:
            windows.append((s, e, g["generation_heading"]))
    return (title or ""), windows

def pick_window_for_year(windows: List[Tuple[int,int,str]], view_year: int) -> Optional[Tuple[int,int,str]]:
    """Pick the window that covers view_year (or the nearest one)."""
    if not windows:
        return None
    covering = [(s,e,h) for (s,e,h) in windows if (s-1) <= view_year <= (e+1)]
    if covering:
        covering.sort(key=lambda t: ((t[1]-t[0]), -t[0]))  # narrower, then newer
        return covering[0]
    def dist(s,e):
        if view_year < s: return s - view_year
        if view_year > e: return view_year - e
        return 0
    windows.sort(key=lambda t: (dist(t[0], t[1]), -t[0]))
    return windows[0]

def detect_via_wikipedia(model: str, view_year: int) -> Tuple[Optional[str], Optional[Tuple[int,int]], Dict[str,Any]]:
    """
    High-accuracy detector used by script2 FIRST.
    Returns (gen_label, (start,end), diagnostics).
    """
    title, wins = find_generation_windows(model)
    if not wins:
        return None, None, {"basis":"wikipedia_module","status":"no_windows","title":title}
    pick = pick_window_for_year(wins, view_year)
    if not pick:
        return None, None, {"basis":"wikipedia_module","status":"no_pick","title":title}
    s, e, heading = pick
    m = re.search(r'\b(?:mk|mark|gen(?:eration)?)\s*([ivx\d]{1,3})\b', heading, flags=re.I)
    label = f"Mk{m.group(1).upper()}" if m else "GEN"
    return label, (s, e), {"basis":"wikipedia_module","title":title,"heading":heading}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia generation window finder with JSON CLI.

Usage (CLI, JSON out):
    python -m scripts.wikipedia_gen --year 2022 --json "Volkswagen Golf"

What it prints with --json:
{
  "title": "Volkswagen Golf",
  "windows": [{"label":"Volkswagen Golf Mk8","start":2019,"end":"present","source":"infobox"}, ...],
  "active": [{"label":"Volkswagen Golf Mk8","start":2019,"end":"present","source":"infobox"}]
}
"""

import sys, re, html, time, argparse, json, urllib.parse
from typing import List, Dict, Any, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

# -------- HTTP session (UA + retries) ----------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "icor-wiki-gen/3.0 (+support@example.com)"})
retries = Retry(total=4, backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"])
SESSION.mount("https://", HTTPAdapter(max_retries=retries))
SESSION.mount("http://",  HTTPAdapter(max_retries=retries))
REQUEST_DELAY_SEC = 0.2

def _get(url: str, params: dict) -> requests.Response:
    time.sleep(REQUEST_DELAY_SEC)
    r = SESSION.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r

PRESENT_WORDS = {"present","current","ongoing","présent","actuel","en cours",
                 "presente","attuale","oggi","atual","em curso","aktuell","heute","bis heute"}

PARENS_RANGE_FLEX_RE = re.compile(
    r"\((?:[^()]*?;?\s*)?(?P<start>\d{4})\s*[–—-]\s*(?P<end>\d{4}|"
    r"(?:present|current|ongoing|"
    r"présent|actuel|en cours|"
    r"presente|attuale|oggi|"
    r"atual|em curso|"
    r"aktuell|heute|bis heute"
    r"))?\)", re.IGNORECASE
)
YEAR_RANGE_FLEX_RE = re.compile(
    r"(?P<start>\b\d{4}\b)\s*[–—-]\s*(?P<end>\b\d{4}\b|"
    r"(?:present|current|ongoing|"
    r"présent|actuel|en cours|"
    r"presente|attuale|oggi|"
    r"atual|em curso|"
    r"aktuell|heute|bis heute"
    r"))", re.IGNORECASE
)
MK_TEXT_RE = re.compile(r"\b(?:mk|mark)\s*(?:[ivxlcdm]+|\d+)\b", re.IGNORECASE)
GEN_KEYWORDS = ["generation","génération","generación","geração","generazione"]

def _to_int(y):
    try: return int(y)
    except: return None

def _normalize_dash(s: str) -> str:
    return s.replace("—","–").replace("-","–")

def _extract_years_any(text: str):
    t = _normalize_dash(text)
    m = PARENS_RANGE_FLEX_RE.search(t) or YEAR_RANGE_FLEX_RE.search(t)
    if not m:
        m2 = re.search(r"\b(depuis|seit|since)\s+(?P<start>\d{4})\b", t, re.IGNORECASE)
        if m2: return m2.group("start"), "present"
        return None, None
    start, end = m.group("start"), m.group("end")
    if (not end) or (str(end).lower() in PRESENT_WORDS): end = "present"
    return start, end

def _heading_looks_like_generation(h_text: str) -> bool:
    low = h_text.lower()
    return MK_TEXT_RE.search(low) or any(kw in low for kw in GEN_KEYWORDS) or bool(PARENS_RANGE_FLEX_RE.search(low))

def _extract_generation_headings(soup: BeautifulSoup) -> List[str]:
    return [tag.get_text(" ", strip=True)
            for tag in soup.find_all(["h2","h3","h4"])
            if _heading_looks_like_generation(tag.get_text(" ", strip=True))]

def _extract_infobox_years(soup: BeautifulSoup):
    infobox = soup.find("table", class_=lambda c: c and "infobox" in c)
    if not infobox: return None, None
    for row in infobox.find_all("tr"):
        th, td = row.find("th"), row.find("td")
        if not th or not td: continue
        label = th.get_text(" ", strip=True).lower()
        if label in ("production","model years","production years",
                     "années de production","años de producción",
                     "produktionszeitraum","anni di produzione","anos de produção"):
            return _extract_years_any(td.get_text(" ", strip=True))
    return None, None

def _parse_generations_from_html(html_content: str) -> List[Dict[str,Any]]:
    soup = BeautifulSoup(html_content, "html.parser")
    out, seen = [], set()
    for heading in _extract_generation_headings(soup):
        s, e = _extract_years_any(heading)
        if s or e:
            key = (s or "", e or "", heading)
            if key in seen: continue
            seen.add(key)
            out.append({"source":"heading","label":heading, "start":s, "end":e})
    if not out:
        s, e = _extract_infobox_years(soup)
        if s or e:
            out.append({"source":"infobox","label":"Infobox production/model years", "start":s, "end":e})
    def sort_key(item):
        try: return int(item["start"]) if item["start"] else 99999
        except: return 99999
    out.sort(key=sort_key)  # <-- correct keyword usage
    return out

def _discover_generation_subpages_from_links(main_html: str, main_title: str) -> List[str]:
    soup = BeautifulSoup(main_html, "html.parser")
    content_root = soup.find(id="mw-content-text") or soup
    candidates = set()
    base = main_title.split("(")[0].strip().lower()
    for a in content_root.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        if not text: continue
        href = a["href"]
        if not href.startswith("/wiki/"): continue
        if any(p in href for p in ("/wiki/Help:","/wiki/Special:","/wiki/Talk:",
                                   "/wiki/Category:","/wiki/File:","/wiki/Portal:","/wiki/Template:")):
            continue
        if _heading_looks_like_generation(text):
            title = a.get("title") or urllib.parse.unquote(href.split("/wiki/")[-1]).replace("_"," ")
            if base.split()[0] in title.lower() or base in title.lower():
                candidates.add(title)
    return sorted(candidates)

def _discover_generation_subpages_via_search(api, main_title: str) -> List[str]:
    queries = [
        f'intitle:"{main_title}" intitle:Mk',
        f'intitle:"{main_title}" intitle:Mark',
        f'intitle:"{main_title}" generation',
        f'"{main_title}" Mk',
        f'"{main_title}" "première génération"',
        f'"{main_title}" "first generation"',
    ]
    titles = set()
    for q in queries:
        r = _get(api, {"action":"query","list":"search","srsearch":q,"srlimit":20,"format":"json","srnamespace":0})
        for hit in r.json().get("query", {}).get("search", []):
            title = html.unescape(hit["title"])
            if any(tok in title.lower() for tok in (" mk","mk","génération","generation")):
                titles.add(title)
    return sorted(titles)

def _search_wikipedia_page(api: str, query: str) -> Tuple[Optional[int], Optional[str]]:
    r = _get(api, {"action":"query","list":"search","srsearch":query,"srlimit":5,"format":"json","srnamespace":0})
    results = r.json().get("query", {}).get("search", [])
    if not results: return None, None
    ids = [str(res["pageid"]) for res in results]
    titles = [html.unescape(res["title"]) for res in results]
    r2 = _get(api, {"action":"query","pageids":"|".join(ids),"prop":"pageprops","format":"json"})
    pageinfo = r2.json().get("query", {}).get("pages", {})
    for pid, title in zip(ids, titles):
        if "disambiguation" not in pageinfo.get(pid, {}).get("pageprops", {}):
            return int(pid), title
    top = results[0]
    return top["pageid"], html.unescape(top["title"])

def _fetch_html_by_pageid(api: str, pageid: int) -> str:
    r = _get(api, {"action":"parse","pageid":pageid,"prop":"text","format":"json","disableeditsection":1})
    return r.json().get("parse", {}).get("text", {}).get("*", "") or ""

def _get_pageid_by_title(api: str, title: str) -> Optional[int]:
    r = _get(api, {"action":"query","titles":title,"format":"json"})
    pages = r.json().get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if pid != "-1": return int(pid)
    return None

def _infer_end_years_inplace(items: List[Dict[str,Any]]) -> None:
    items.sort(key=lambda g: _to_int(g.get("start")) or 99999)
    for i in range(len(items)-1):
        curr, nxt = items[i], items[i+1]
        s_curr, e_curr, s_next = _to_int(curr.get("start")), _to_int(curr.get("end")), _to_int(nxt.get("start"))
        if not s_curr or not s_next: continue
        needs = (curr.get("end") in (None,"","present")) or (e_curr is not None and e_curr >= s_next)
        if needs:
            curr["end"] = str(max(s_curr, s_next-1))
            curr["end_inferred"] = True

def _covers(year: int, start, end) -> bool:
    s = _to_int(start)
    e = _to_int(end) if (end and end != "present") else 9999
    return bool(s) and s <= year <= e

def detect_via_wikipedia(model: str, year: int, lang: str = "en") -> Tuple[str, Tuple[int,int], Dict[str,Any]]:
    """
    Library entry point.
    Returns: (label, (start,end_int), diagnostics)
    """
    api = f"https://{lang}.wikipedia.org/w/api.php"
    pid, title = _search_wikipedia_page(api, model)
    if not pid:
        raise RuntimeError(f"No Wikipedia page for '{model}' [{lang}]")

    main_html = _fetch_html_by_pageid(api, pid)
    windows = _parse_generations_from_html(main_html)

    # If only broad infobox span, try subpages and merge
    only_infobox = (len(windows) == 1 and windows[0]["source"] == "infobox")
    if (not windows) or only_infobox:
        link_titles = _discover_generation_subpages_from_links(main_html, title)
        search_titles = _discover_generation_subpages_via_search(api, title)
        for t in sorted(set(link_titles) | set(search_titles)):
            spid = _get_pageid_by_title(api, t)
            if not spid: continue
            html2 = _fetch_html_by_pageid(api, spid)
            for g in _parse_generations_from_html(html2):
                g["label"] = g.get("label") or t
                g["source"] = g.get("source") or "subpage"
                windows.append(g)

    # dedupe
    seen, dedup = set(), []
    for g in windows:
        key = (g.get("start") or "", g.get("end") or "", g.get("label") or "", g.get("source") or "")
        if key in seen: continue
        seen.add(key); dedup.append(g)

    dedup.sort(key=lambda g: _to_int(g.get("start")) or 99999)
    _infer_end_years_inplace(dedup)

    active = [g for g in dedup if _covers(year, g.get("start"), g.get("end"))]
    if not active:
        raise RuntimeError(f"No generation covers {year} for '{model}' (found {len(dedup)} windows).")

    # choose the narrowest covering window; tiebreak = latest start
    active.sort(key=lambda g: ((_to_int(g.get("end")) or 9999) - (_to_int(g.get("start")) or 0), -(_to_int(g.get("start")) or 0)))
    pick = active[0]
    start_i = _to_int(pick["start"]) or year
    end_i = (_to_int(pick["end"]) if (pick["end"] and pick["end"] != "present") else 9999)

    label = pick.get("label") or "GEN"
    diag = {"title": title, "basis": "wikipedia_gen_json", "picked": pick, "all_windows": dedup}
    return label, (start_i, end_i), diag

# ------------- CLI -------------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON only")
    return ap.parse_args()

def main():
    args = _parse_args()
    try:
        label, (s, e), diag = detect_via_wikipedia(args.model, args.year, lang=args.lang)
        # Build JSON payload
        all_w = []
        for w in diag.get("all_windows", []):
            all_w.append({
                "label": w.get("label"),
                "start": _to_int(w.get("start")),
                "end": (w.get("end") if w.get("end") == "present" else _to_int(w.get("end"))),
                "source": w.get("source")
            })
        active = [w for w in all_w if _covers(args.year, w["start"], w["end"])]
        payload = {"title": diag.get("title"), "windows": all_w, "active": active}

        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(f"Title: {payload['title']}")
            print("\nGenerations (normalized):")
            for w in all_w:
                e = w["end"] if isinstance(w["end"], str) else (w["end"] or "?")
                print(f" - {w['label']} -> {w['start']}–{e}    [{w['source']}]")
            print(f"\nActive in {args.year}:")
            if not active:
                print(" - none")
            else:
                for w in active:
                    e = w["end"] if isinstance(w["end"], str) else (w["end"] or "?")
                    print(f" - {w['label']} -> {w['start']}–{e}")
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()

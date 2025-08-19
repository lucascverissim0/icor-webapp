#!/usr/bin/env python3
"""
ICOR passenger-car data pipeline v7 (EU + World + Combined)

Fixes & improvements vs v6:
  • EU/World file matching is now strict:
      - EU only loads Top100_YYYY.txt
      - World only loads Top100_World_YYYY.txt
    (Prevents World rows from leaking into EU.)
  • Generation normalization:
      - Treats placeholders ("None", "Dominant", "Unknown", "-", etc.) as missing.
      - Cleans mojibake/stray suffixes like " / India)" or trailing ")"
      - Defaults blanks to "I".
  • Year sheets saved as "2019 EU" / "2019 World".
  • Fleet/Repl pivots for EU & World with integer ceilings.
  • Strategic Opportunities in 3 sheets: ICOR_SO_EU, ICOR_SO_World, ICOR_SO_All (Combined).
  • BEV counterpart lookup cached in bev_cache.json.

Outputs:
  • passenger_car_data.xlsx
  • bev_scores.json
  • bev_cache.json
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
import streamlit as st

# ───────────── API key from Streamlit Secrets ─────────────
try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # fallback for local dev

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not set; BEV counterpart lookups will likely fail.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ─────────────────────────── Constants ─────────────────────────
DECAY_RATE = 0.0556
REPL_RATE_MEAN = 0.021
YEARS_TO_PROJECT = range(2016, 2036)   # projection horizon
SELECTED_YEAR = 2030
MODEL = "gpt-4o-mini"
CACHE_FILE = Path("bev_cache.json")

# ───────────────────────── helper functions ────────────────────
def parse_icor_models(text: str) -> set[str]:
    models = set()
    for ln in text.strip().splitlines():
        if not ln.strip():
            continue
        mpart = ln.split("\t", 1)[0]
        for m in re.split(r"[+/&]", mpart):
            models.add(m.strip().title())
    return models

def ask_openai_bev(model: str, brand: str) -> bool:
    if not client:
        return False
    sys_msg = "You are an automotive product expert. Answer with exactly 'yes' or 'no'."
    user_msg = (
        f"Does the current {brand} {model} with an internal-combustion engine "
        "have a battery-electric counterpart (same vehicle type) sold in Europe? "
        "Respond only with yes or no."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            temperature=0,
            max_tokens=1,
        )
        ans = (resp.choices[0].message.content or "").strip().lower()
        return ans.startswith("y")
    except Exception as e:
        print(f"⚠️  OpenAI error for {brand} {model}: {e} – defaulting to 'no'")
        return False

def has_bev_counterpart_cached(model: str) -> bool:
    cache: Dict[str, bool] = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text())
        except Exception:
            cache = {}

    if model in cache:
        return cache[model]

    parts = model.split()
    brand, base = (parts[0], " ".join(parts[1:])) if len(parts) >= 2 else (model, model)
    answer = ask_openai_bev(base, brand)
    cache[model] = answer
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass
    time.sleep(0.25)  # be polite
    return answer

def pct_rank(series: pd.Series) -> pd.Series:
    return (series.rank(pct=True) * 20).clip(0, 20).round(0)

# ───────── Generation normalization helpers ─────────
def roman_from_int(n: int) -> str:
    if not (1 <= n <= 20):
        return str(n)
    pairs = [
        (10, "X"), (9, "IX"), (8, "VIII"), (7, "VII"), (6, "VI"),
        (5, "V"), (4, "IV"), (3, "III"), (2, "II"), (1, "I")
    ]
    out = []
    q, r = divmod(n, 10)
    out.append("X" * q)
    n = r
    for val, sym in pairs:
        while n >= val:
            out.append(sym)
            n -= val
    return "".join(out)

def int_from_roman(s: str) -> int | None:
    s = s.upper()
    roman_vals = {"I": 1, "V": 5, "X": 10}
    if not re.fullmatch(r"(X{0,2})(IX|IV|V?I{0,3})", s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        v = roman_vals[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total if 1 <= total <= 20 else None

PLACEHOLDERS = {"", "none", "null", "nan", "n/a", "na", "unknown", "-", "—", "--", "dominant"}

def clean_generation(val: str) -> str:
    """
    Fix mojibake, normalize dashes, drop year-only parentheses.
    Treat placeholders like 'None', 'Dominant', '-' as missing.
    If cleaned result is empty → return 'I'.
    """
    if pd.isna(val):
        return "I"
    s = str(val).strip()
    if s.lower() in PLACEHOLDERS:
        return "I"

    s = s.replace("â€“", "–")
    s = re.sub(r"[–—−-]", "-", s)
    # drop year-only parentheses and trailing local-market tails like " / India)"
    s = re.sub(r"\s*\([^)]*(?:19|20)\d{2}[^)]*\)", "", s)
    s = re.sub(r"\s*/\s*[A-Za-z\s]*\)$", "", s)
    s = re.sub(r"[\);]+$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "I"

def normalize_generation_label(val: str) -> str:
    """
    mk2 / mk 2 / mark ii / gen 2 / generation iii / 2 / ii  -> Roman numeral (II/III/…)
    Platform codes like B8/B9 etc. are preserved.
    If blank after normalization → 'I'.
    """
    if pd.isna(val):
        return "I"
    s = str(val).strip()
    if not s:
        return "I"

    # If the entire token is just a number or a simple Roman numeral
    if re.fullmatch(r"\d{1,2}", s):
        return roman_from_int(int(s))
    ri = int_from_roman(s)
    if ri is not None:
        return roman_from_int(ri)

    low = s.lower()
    m = re.search(r"\b(?:mk|mark|gen|generation)\s*([ivx]+|\d{1,2})\b", low, flags=re.I)
    if m:
        tok = m.group(1)
        if tok.isdigit():
            return roman_from_int(int(tok))
        ri = int_from_roman(tok)
        return roman_from_int(ri) if ri else (s or "I")

    m = re.fullmatch(r"(?:mk|mark)([ivx]+|\d{1,2})", low, flags=re.I)
    if m:
        tok = m.group(1)
        if tok.isdigit():
            return roman_from_int(int(tok))
        ri = int_from_roman(tok)
        return roman_from_int(ri) if ri else (s or "I")

    return s or "I"

# ─────────────────────────── Load ICOR list ──────────────────────
icor_path = Path("icor_supported_models.txt")
icor_raw_text = icor_path.read_text(encoding="utf-8") if icor_path.exists() else ""
icor_supported = parse_icor_models(icor_raw_text) if icor_raw_text else set()

# ─────────────────────────── Load Top100 EU & World ──────────────
def _load_region(pattern: str, units_colname: str) -> Dict[int, pd.DataFrame]:
    """
    Returns {year: DataFrame} for a given glob pattern.
    Ensures 'model', 'generation', 'units_sold', 'Year' exist.
    """
    out: Dict[int, pd.DataFrame] = {}
    for txt in Path(".").glob(pattern):
        name = txt.name

        # Extra guard: if the caller didn't explicitly request World, avoid ingesting it accidentally
        if "_World_" in name and "World_" not in pattern:
            continue

        m = re.search(r"(\d{4})", txt.stem)
        if not m:
            continue
        yr = int(m.group(1))
        try:
            df = pd.DataFrame(json.load(txt.open(encoding="utf-8")))
        except Exception as e:
            print(f"⚠️  Could not read {txt}: {e}")
            continue

        # model tidy
        df["model"] = (
            df["model"]
            .astype(str)
            .str.title()
            .str.replace(r"[-/]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # generation tidy -> "I" if missing/placeholder
        df["generation"] = (
            df.get("generation", pd.Series(["I"] * len(df)))
            .apply(clean_generation)
            .apply(normalize_generation_label)
        )
        df["generation"] = df["generation"].astype(str).str.strip()
        df.loc[df["generation"].str.lower().isin(PLACEHOLDERS), "generation"] = "I"

        # units
        if "units_sold" not in df.columns:
            df["units_sold"] = df.get("projected_units_2025", 0)
        df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0).astype(int)

        df["Year"] = yr
        out[yr] = df[["model", "generation", "units_sold", "Year"]].rename(
            columns={"model": "model", "generation": "generation", "units_sold": units_colname}
        )
    return out

# Strict patterns: EU vs World
eu_by_year = _load_region("Top100_[0-9][0-9][0-9][0-9].txt", "units_eu")
world_by_year = _load_region("Top100_World_[0-9][0-9][0-9][0-9].txt", "units_world")

if not eu_by_year and not world_by_year:
    raise SystemExit("❌ No Top100 files found (neither EU nor World).")

eu_all = pd.concat(eu_by_year.values(), ignore_index=True) if eu_by_year else pd.DataFrame(columns=["model","generation","units_eu","Year"])
world_all = pd.concat(world_by_year.values(), ignore_index=True) if world_by_year else pd.DataFrame(columns=["model","generation","units_world","Year"])

# ─────────────────────────── helper to build pivots ──────────────
def build_region_outputs(all_sales: pd.DataFrame, units_col: str
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: (fleet_piv, repl_piv, opportunities_df)
    """
    # projections
    proj_rows: List[Dict] = []
    for yr in YEARS_TO_PROJECT:
        for _, r in all_sales.iterrows():
            age = yr - int(r["Year"])
            base_units = int(r[units_col])
            if age < 0:
                fleet_size = 0.0
            else:
                fleet_size = float(base_units) * (1 - DECAY_RATE) ** age
            proj_rows.append(
                dict(Year=yr, Model=r["model"], Generation=r["generation"], FleetSize=fleet_size)
            )

    fleet = (
        pd.DataFrame(proj_rows)
        .groupby(["Year", "Model", "Generation"], as_index=False)["FleetSize"]
        .sum()
    )

    # Fleet pivot (ceil integers)
    fleet_piv = (
        fleet.pivot_table(index=["Model", "Generation"], columns="Year", values="FleetSize")
        .fillna(0)
        .reset_index()
    )
    year_cols = [c for c in fleet_piv.columns if isinstance(c, (int, np.integer)) or re.fullmatch(r"\d{4}", str(c))]
    fleet_piv[year_cols] = np.ceil(fleet_piv[year_cols].to_numpy()).astype(int)

    # Replacement pivot
    repl = fleet.copy()
    repl["Repl"] = (repl["FleetSize"] * REPL_RATE_MEAN).round(0)
    repl_piv = (
        repl.pivot_table(index=["Model", "Generation"], columns="Year", values="Repl")
        .fillna(0)
        .reset_index()
    )

    # Appearances in Top100 for this region
    appearances = (
        all_sales.groupby(["model", "generation"]).size().reset_index(name="YearsInTop100")
    )

    # Build region opportunities
    opp = (
        fleet_piv.merge(repl_piv, on=["Model", "Generation"], suffixes=("_fleet", "_repl"))
        .merge(appearances.rename(columns={"model": "Model", "generation": "Generation"}),
               on=["Model", "Generation"], how="left")
    )

    opp["FutureFleet"] = opp[f"{SELECTED_YEAR}_fleet"]
    opp["FutureRepl"]  = opp[f"{SELECTED_YEAR}_repl"]
    opp["YearsInTop100"] = opp["YearsInTop100"].fillna(0)

    return fleet_piv, repl_piv, opp

# ───────────────────────── BEV lookup across union ───────────────
all_models_for_bev = set(eu_all["model"].tolist()) | set(world_all["model"].tolist())
bev_lookup = {m: has_bev_counterpart_cached(m) for m in sorted(all_models_for_bev)}

# ─────────────────────────── Write Excel ─────────────────────────
with pd.ExcelWriter("passenger_car_data.xlsx", engine="openpyxl") as xls:
    # Per-year raw sheets
    for y, df in sorted(eu_by_year.items()):
        df_out = df.rename(columns={"units_eu": "units_sold"})
        df_out.to_excel(xls, sheet_name=f"{y} EU", index=False)
    for y, df in sorted(world_by_year.items()):
        df_out = df.rename(columns={"units_world": "units_sold"})
        df_out.to_excel(xls, sheet_name=f"{y} World", index=False)

    # EU region pivots & opps
    if not eu_all.empty:
        eu_fleet_piv, eu_repl_piv, eu_opp = build_region_outputs(eu_all, "units_eu")
        eu_fleet_piv.to_excel(xls, sheet_name="Fleet_By_Model_Year_EU", index=False)
        eu_repl_piv.to_excel(xls, sheet_name="Windshield_Repl_By_Year_EU", index=False)
    else:
        eu_fleet_piv = pd.DataFrame(columns=["Model", "Generation"])
        eu_repl_piv = pd.DataFrame(columns=["Model", "Generation"])
        eu_opp = pd.DataFrame(columns=["Model", "Generation", "FutureFleet", "FutureRepl", "YearsInTop100"])

    # World region pivots & opps
    if not world_all.empty:
        w_fleet_piv, w_repl_piv, w_opp = build_region_outputs(world_all, "units_world")
        w_fleet_piv.to_excel(xls, sheet_name="Fleet_By_Model_Year_World", index=False)
        w_repl_piv.to_excel(xls, sheet_name="Windshield_Repl_By_Year_World", index=False)
    else:
        w_fleet_piv = pd.DataFrame(columns=["Model", "Generation"])
        w_repl_piv = pd.DataFrame(columns=["Model", "Generation"])
        w_opp = pd.DataFrame(columns=["Model", "Generation", "FutureFleet", "FutureRepl", "YearsInTop100"])

    # ICOR & BEV & GenCount additions (function)
    def finalize_opps(df_base: pd.DataFrame, all_sales_for_gen_count: pd.DataFrame) -> pd.DataFrame:
        df = df_base.copy()
        df["ICOR_Supported"] = df["Model"].isin(icor_supported)
        df["FleetScore"] = pct_rank(df["FutureFleet"])
        df["ReplScore"]  = pct_rank(df["FutureRepl"])
        df["YearsScore"] = pct_rank(df["YearsInTop100"])
        df["ICORScore"]  = df["ICOR_Supported"].map({True: 20, False: 10})

        df["BEVScore"] = df["Model"].map(lambda m: 10 if bev_lookup.get(m, False) else 20)

        gen_count = (
            all_sales_for_gen_count.groupby("model")["generation"].nunique().reset_index(name="GenCount")
        )
        df = df.merge(gen_count.rename(columns={"model": "Model"}), on="Model", how="left")
        df["GenScore"] = pct_rank(df["GenCount"].fillna(1))

        df["Score"] = df[
            ["FleetScore", "ReplScore", "YearsScore", "ICORScore", "BEVScore", "GenScore"]
        ].sum(axis=1)

        def label(s: float) -> str:
            return (
                "⭐⭐⭐ High Priority" if s >= 80 else
                "⭐⭐ Medium Priority" if s >= 60 else
                "⭐ Low Priority" if s >= 40 else
                "Not Recommended"
            )
        df["Recommendation"] = df["Score"].apply(label)

        cols = [
            "Model", "Generation",
            "FutureFleet", "FutureRepl", "YearsInTop100",
            "FleetScore", "ReplScore", "YearsScore", "ICORScore", "BEVScore", "GenScore",
            "Score", "Recommendation",
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols].sort_values("Score", ascending=False)

    # EU only
    eu_opps_final = finalize_opps(eu_opp, eu_all if not eu_all.empty else pd.DataFrame(columns=["model","generation"]))
    eu_opps_final.to_excel(xls, sheet_name="ICOR_SO_EU", index=False)

    # World only
    w_opps_final = finalize_opps(w_opp, world_all if not world_all.empty else pd.DataFrame(columns=["model","generation"]))
    w_opps_final.to_excel(xls, sheet_name="ICOR_SO_World", index=False)

    # Combined: prefer World values; fallback to EU if World missing
    comb = pd.merge(
        w_opp[["Model", "Generation", "FutureFleet", "FutureRepl", "YearsInTop100"]],
        eu_opp[["Model", "Generation", "FutureFleet", "FutureRepl", "YearsInTop100"]],
        on=["Model", "Generation"], how="outer", suffixes=("_World", "_EU")
    )

    def _pick(a, b):  # prefer world (a), else eu (b)
        return a if pd.notna(a) else (b if pd.notna(b) else 0)

    comb["FutureFleet"]   = [ _pick(a, b) for a, b in zip(comb.get("FutureFleet_World"), comb.get("FutureFleet_EU")) ]
    comb["FutureRepl"]    = [ _pick(a, b) for a, b in zip(comb.get("FutureRepl_World"), comb.get("FutureRepl_EU")) ]
    comb["YearsInTop100"] = [ max(a or 0, b or 0) for a, b in zip(comb.get("YearsInTop100_World"), comb.get("YearsInTop100_EU")) ]

    comb_base = comb[["Model","Generation","FutureFleet","FutureRepl","YearsInTop100"]].copy()
    all_for_gen_count = pd.concat(
        [eu_all.rename(columns={"units_eu":"units"}), world_all.rename(columns={"units_world":"units"})],
        ignore_index=True
    ) if not (eu_all.empty and world_all.empty) else pd.DataFrame(columns=["model","generation"])

    comb_final = finalize_opps(comb_base, all_for_gen_count)
    comb_final.to_excel(xls, sheet_name="ICOR_SO_All", index=False)

# ─────────────────────────── JSON export ────────────────────────
bev_pairs = [{"Model": m, "BEVScore": 10 if bev_lookup.get(m, False) else 20}
             for m in sorted(all_models_for_bev)]
Path("bev_scores.json").write_text(json.dumps(bev_pairs, indent=2, ensure_ascii=False))

print("✅ passenger_car_data.xlsx, bev_scores.json and bev_cache.json updated.")

# ui/pages/02_Model_Researcher.py
import os
import glob
import json
import subprocess
import time
import pandas as pd
import streamlit as st
import posthog
import sys

# Optional import; only used if credentials are present in secrets
try:
    import streamlit_authenticator as stauth  # noqa: F401
    HAS_AUTH_LIB = True
except Exception:
    HAS_AUTH_LIB = False

# === CONFIG ===
PREFERRED_SCRIPT2 = "script2.py"  # preferred filename in /scripts

# === PATHS (go up TWO levels from ui/pages/ to the repo root) ===
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")
SCRIPTS_DIR = os.path.join(ROOT, "scripts")

# --- Logo path (robust to where the page runs from)
def find_logo_path(root: str) -> str | None:
    candidates = [
        os.path.join(root, "ui", "assets", "icor-logo.png"),
        os.path.join(root, "assets", "icor-logo.png"),
        os.path.join(root, "icor-logo.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

LOGO_PATH = find_logo_path(ROOT)

# IMPORTANT: Do NOT call st.set_page_config() here (keep it in app.py only)

# === DARK THEME (visuals only) ===
st.markdown(
    """
    <style>
      html, body, .block-container { background-color: #0E1117 !important; color: #E6E6E6 !important; }
      .stDataFrame, .stMarkdown { color: #E6E6E6 !important; }
      .badges { display:flex; gap:10px; flex-wrap:wrap; margin: 6px 0 14px 0; }
      .badge {
        background:#161A22; border:1px solid #30363d; border-radius: 999px;
        padding:6px 12px; font-size:13px; color:#E6E6E6;
      }
      .badge-strong { border-color:#3b82f6; }
      .badge-warn { border-color:#f59e0b; }
      .badge-soft { border-color:#52525b; }
      .k { color:#A1A1AA; }
    </style>
    """,
    unsafe_allow_html=True,
)

# === AUTHENTICATION ===
def ensure_auth():
    """Use streamlit_authenticator if credentials+cookie exist; else rely on session set by app.py."""
    creds = st.secrets.get("credentials")
    cookie = st.secrets.get("cookie")

    if creds and cookie and HAS_AUTH_LIB:
        try:
            authenticator = stauth.Authenticate(
                credentials=creds,
                cookie_name=cookie["name"],
                key=cookie["key"],
                cookie_expiry_days=cookie["expiry_days"],
            )
        except Exception as e:
            st.error(f"Authentication configuration error: {e}")
            st.stop()

        name, auth_status, username = authenticator.login("Login", "main")

        if auth_status is False:
            st.error("Invalid username or password.")
            st.stop()
        elif auth_status is None:
            st.info("Please log in to continue.")
            st.stop()

        st.session_state["user_id"] = username
        st.session_state["user_name"] = name

        with st.sidebar:
            authenticator.logout("Logout", "sidebar")
        return True

    # Fallback: trust session created by app.py's custom login
    if "user_id" in st.session_state:
        return True

    st.info("Please open the **app** page and log in first.")
    st.stop()

ensure_auth()

# === POSTHOG (tracking) ===
try:
    posthog.project_api_key = st.secrets.posthog.api_key
    posthog.host = st.secrets.posthog.host
except Exception:
    pass

def track(event: str, props: dict | None = None):
    try:
        uid = st.session_state.get("user_id", "anon")
        posthog.capture(uid, event, properties=props or {})
    except Exception:
        pass

track("page_view_model_researcher", {"page": "02_Generation_Estimator"})

# --- Header: logo + title + slogan
c1, c2 = st.columns([1, 6], vertical_alignment="center")
with c1:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_column_width=True)
with c2:
    st.title("ICOR – Model Researcher")
    st.caption("automatically perfect")

# === HELPERS ===
def localize_bools(df: pd.DataFrame, prefer_cols=None, true_txt="VRAI", false_txt="FAUX") -> pd.DataFrame:
    d = df.copy()
    candidates = list(prefer_cols) if prefer_cols else list(d.columns)
    for c in candidates:
        if c not in d.columns:
            continue
        s = d[c]
        if s.dtype == "bool":
            d[c] = s.map({True: true_txt, False: false_txt})
            continue
        if s.dtype == "object":
            vals = set(str(x).strip().lower() for x in s.dropna().unique())
            if vals.issubset({"true", "false", "vrai", "faux", "1", "0"}):
                d[c] = s.map(
                    {
                        True: true_txt, False: false_txt,
                        "True": true_txt, "False": false_txt,
                        "true": true_txt, "false": false_txt,
                        "VRAI": true_txt, "FAUX": false_txt,
                        "vrai": true_txt, "faux": false_txt,
                        "1": true_txt, "0": false_txt,
                    }
                ).fillna(s)
    return d

def find_script2() -> str | None:
    candidates = [
        os.path.join(SCRIPTS_DIR, PREFERRED_SCRIPT2),
        os.path.join(SCRIPTS_DIR, "car_sales_estimator_gen_autodetect_alias_icor.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    hits = sorted(glob.glob(os.path.join(SCRIPTS_DIR, "car_sales_estimator*.py")))
    return hits[0] if hits else None

def load_latest_output():
    matches = sorted(glob.glob(os.path.join(DATA_DIR, "sales_estimates_*.xlsx")))
    if not matches:
        return None
    latest = max(matches, key=os.path.getmtime)
    return latest

def parse_seed_badges(xlsx_path: str):
    """
    Parse Summary + Seeds_Constraints to produce UI badges and metadata,
    including generation launch year.
    """
    confidence = None
    eu_badge = {"style": "badge-soft", "text": "Europe seed: none"}
    w_badge  = {"style": "badge-soft", "text": "World seed: none"}
    plaus_badge = None
    launch_year = None
    gen_end = None
    basis = None

    # 1) Confidence + plausibility + launch year/basis from Summary
    try:
        summary = pd.read_excel(xlsx_path, sheet_name="Summary")
        if len(summary):
            if "Confidence" in summary.columns:
                confidence = str(summary.loc[0, "Confidence"])
            if "Generation_Window_Start" in summary.columns:
                ly = summary.loc[0, "Generation_Window_Start"]
                launch_year = int(ly) if pd.notnull(ly) else None
            if "Generation_Window_End" in summary.columns:
                ge = summary.loc[0, "Generation_Window_End"]
                gen_end = int(ge) if pd.notnull(ge) else None
            if "Generation_Window_Basis" in summary.columns:
                basis = str(summary.loc[0, "Generation_Window_Basis"] or "").strip()

            if "Plausibility_Flag" in summary.columns:
                flag = bool(summary.loc[0, "Plausibility_Flag"])
                reason = str(summary.loc[0, "Plausibility_Reason"] or "").strip()
                plaus_badge = (
                    {"style": "badge-warn", "text": f"Plausibility: check · {reason[:110]}"}
                    if flag else {"style": "badge", "text": "Plausibility: OK"}
                )
    except Exception:
        pass

    # 2) Seeds / constraints from Seeds_Constraints
    seed_json, cons_json = None, None
    try:
        sc = pd.read_excel(xlsx_path, sheet_name="Seeds_Constraints")
        for _, row in sc.iterrows():
            kind = str(row.get("Seed_or_Constraint","")).strip().lower()
            raw = row.get("JSON","")
            if isinstance(raw, str) and raw.strip():
                if kind == "seed":
                    seed_json = json.loads(raw)
                elif kind == "constraints":
                    cons_json = json.loads(raw)
    except Exception:
        pass

    if isinstance(cons_json, dict) and "europe" in cons_json:
        ce = cons_json.get("europe", {})
        if "exact" in ce and ce["exact"]:
            year, val = next(iter(ce["exact"].items()))
            src = (seed_json or {}).get("europe", {}).get("source", "seed")
            eu_badge = {"style":"badge-strong", "text": f"Europe {year}: exact {val:,}  ·  {src}"}
        elif "range" in ce and ce["range"] and seed_json:
            year, pair = next(iter(ce["range"].items()))
            lo, hi = pair
            eu_badge = {"style":"badge-warn", "text": f"Europe {year}: range {lo:,}–{hi:,}  ·  EU-share prior"}
    elif isinstance(seed_json, dict) and seed_json.get("europe"):
        e = seed_json["europe"]
        eu_badge = {"style":"badge-strong", "text": f"Europe {seed_json.get('year')}: {int(e.get('value',0)):,}  ·  {e.get('source','seed')}"}

    if isinstance(cons_json, dict) and "world" in cons_json:
        cw = cons_json.get("world", {})
        if "exact" in cw and cw["exact"]:
            year, val = next(iter(cw["exact"].items()))
            src = (seed_json or {}).get("world", {}).get("source", "seed")
            w_badge = {"style":"badge-strong", "text": f"World {year}: exact {val:,}  ·  {src}"}
        elif "range" in cw and cw["range"]:
            year, pair = next(iter(cw["range"].items()))
            lo, hi = pair
            w_badge = {"style":"badge-warn", "text": f"World {year}: range {lo:,}–{hi:,}  ·  EU→World prior"}
    elif isinstance(seed_json, dict) and seed_json.get("world"):
        w = seed_json["world"]
        w_badge = {"style":"badge-strong", "text": f"World {seed_json.get('year')}: {int(w.get('value',0)):,}  ·  {w.get('source','seed')}"}

    return confidence, eu_badge, w_badge, plaus_badge, launch_year, gen_end, basis

def header_badges(confidence: str | None, eu_badge: dict, w_badge: dict,
                  plaus_badge: dict | None = None,
                  launch_year: int | None = None,
                  basis: str | None = None):
    conf_txt = f"Confidence: <strong>{confidence}</strong>" if confidence else "Confidence: <span class='k'>n/a</span>"
    plaus_html = f"<div class='badge {plaus_badge['style']}'>{plaus_badge['text']}</div>" if plaus_badge else ""
    launch_html = f"<div class='badge badge-strong'>Launch: <strong>{launch_year}</strong></div>" if launch_year else ""
    basis_html = f"<div class='badge'>Basis: <span class='k'>{basis}</span></div>" if basis else ""
    st.markdown(
        f"""
        <div class="badges">
          <div class="badge">{conf_txt}</div>
          {launch_html}
          <div class="badge {eu_badge['style']}">{eu_badge['text']}</div>
          <div class="badge {w_badge['style']}">{w_badge['text']}</div>
          {plaus_html}
          {basis_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _safe_get(dict_like, dotted, default=None):
    try:
        cur = dict_like
        for part in dotted.split("."):
            cur = cur[part]
        return cur
    except Exception:
        return default

def run_script2_with_env(input_text: str):
    """
    Launch Script 2 as a child process, piping stdin with the 3 lines:
    <model>\\n<generation or blank>\\n<start_year>\\n
    """
    script_path = find_script2()
    if not script_path:
        files = ", ".join(sorted(os.listdir(SCRIPTS_DIR))) if os.path.exists(SCRIPTS_DIR) else "(missing)"
        return 127, f"[ERROR] Script not found in {SCRIPTS_DIR}. Files here: {files}"

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = _safe_get(st.secrets, "openai.api_key", env.get("OPENAI_API_KEY", ""))
    env["SERPAPI_KEY"]    = _safe_get(st.secrets, "serpapi.api_key", env.get("SERPAPI_KEY", ""))

    # 👇 use the same interpreter Streamlit is running
    proc = subprocess.Popen(
        [sys.executable, script_path],
        cwd=DATA_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    try:
        out, _ = proc.communicate(input_text, timeout=240)  # 4-minute guard
        code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        out = "[ERROR] Script 2 timed out."
        code = 124
    return code, out

def extract_saved_paths_from_stdout(out: str) -> tuple[str | None, str | None]:
    """Return (csv_path, xlsx_path) if the child printed 'Saved CSV:' / 'Saved Excel:' lines."""
    csv_path = None
    xlsx_path = None
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("Saved CSV:"):
            csv_path = line.split("Saved CSV:", 1)[1].strip()
        elif line.startswith("Saved Excel:"):
            xlsx_path = line.split("Saved Excel:", 1)[1].strip()
    return csv_path, xlsx_path

# === UI (inputs + action) ===
with st.form("gen_estimator_form", clear_on_submit=False):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        model = st.text_input("Car model", placeholder="Ford Fiesta")
    with col2:
        generation = st.text_input("Generation (optional)", placeholder="Mk6")
    with col3:
        start_year = st.number_input("Start year", min_value=1990, max_value=2035, value=2013, step=1)

    submitted = st.form_submit_button("Run Generation Estimator", type="primary")

if submitted:
    if not model:
        st.error("Please enter a car model.")
    else:
        input_text = f"{model}\n{generation}\n{start_year}\n"
        with st.status("Running Script 2…", expanded=False):
            code, out = run_script2_with_env(input_text)

        # Surface common setup issues
        if "OPENAI_API_KEY not set" in (out or ""):
            st.warning("OPENAI_API_KEY isn’t configured for Script 2. Set it in `st.secrets` or env.")
        if "SERPAPI_KEY not set" in (out or ""):
            st.info("SERPAPI_KEY isn’t set; web seeding will be disabled (local-only).")

        if code != 0:
            tail = "\n".join((out or "").splitlines()[-25:])
            st.error("Script 2 failed. See brief diagnostic below.")
            if tail.strip():
                with st.expander("Diagnostic (last lines)"):
                    st.code(tail)
            track("run_script2", {
                "model": model,
                "generation": generation or "",
                "start_year": int(start_year),
                "status": "error",
                "timestamp": int(time.time()),
            })
        else:
            # Prefer exact path from stdout
            csv_path, xlsx_from_stdout = extract_saved_paths_from_stdout(out)
            
            if not xlsx_from_stdout:
                st.error("The estimator finished but didn’t save an Excel file. "
                         "Most likely the OpenAI step failed or returned invalid JSON. "
                         "See the Run log below.")
                # Still show the run log for debugging
                tail = "\n".join((out or "").splitlines()[-50:])
                with st.expander("Run log", expanded=False):
                    st.code(tail)
            else:
                xlsx = xlsx_from_stdout if os.path.exists(xlsx_from_stdout) else load_latest_output()
            
                # Always show last lines for debugging
                tail = "\n".join((out or "").splitlines()[-25:])
                with st.expander("Run log", expanded=False):
                    st.code(tail)
            
                if not xlsx or not os.path.exists(xlsx):
                    st.warning("No output Excel found. The estimator ran but didn’t produce a file (or the path wasn’t accessible).")
                else:
                    confidence, eu_b, w_b, plaus_b, launch_year, gen_end, basis = parse_seed_badges(xlsx)
                    header_badges(confidence, eu_b, w_b, plaus_b, launch_year=launch_year, basis=basis)

                try:
                    est = pd.read_excel(xlsx, sheet_name="Estimates")
                    est = localize_bools(est, prefer_cols=["ICOR_Supported", "Gen_Active"])
                    st.dataframe(est, use_container_width=True)

                    st.download_button(
                        "Download estimates Excel",
                        data=open(xlsx, "rb").read(),
                        file_name=os.path.basename(xlsx),
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Could not read Estimates sheet: {e}")

                track("run_script2", {
                    "model": model,
                    "generation": generation or "",
                    "start_year": int(start_year),
                    "status": "success",
                    "timestamp": int(time.time()),
                })

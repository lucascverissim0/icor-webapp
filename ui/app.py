# app.py
import os
import re
import glob
import time
import json
import subprocess
import pandas as pd
import streamlit as st
import posthog
import sys  # for sys.executable

# =================== PAGE META + THEME ===================
st.set_page_config(page_title="ICOR – Strategic Opportunities", layout="wide")
print("[CHECKPOINT] Page config set")

# ICOR styling for cards & tables
st.markdown("""
<style>
  html, body, .block-container { background-color:#0E1117 !important; color:#E6E6E6 !important; }
  .stDataFrame { border-radius:12px; overflow:hidden; }

  /* Make buttons look like our cards */
  div.stButton > button {
    width: 100%;
    text-align: left;
    background: #161A22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 16px 18px;
    line-height: 1.25;
    transition: transform .08s ease-in-out, border-color .08s;
    white-space: pre-wrap;    /* allow newlines */
    font-size: 16px;
    color: #E6E6E6;
  }
  div.stButton > button:hover { transform: translateY(-2px); border-color:#2AA7C9; }
</style>
""", unsafe_allow_html=True)

# =================== ONE-SHOT RERUN GUARDS ===================
# We set a flag before st.rerun() and pop it at the next render to avoid loops.
def _consume_flag(flag_key: str) -> bool:
    """Pop and return True if a one-shot flag was present."""
    return bool(st.session_state.pop(flag_key, False))

# Consume any one-shot flags set just before a rerun
_just_logged_in  = _consume_flag("_just_logged_in")
_just_logged_out = _consume_flag("_just_logged_out")
_just_ran_script1 = _consume_flag("_just_ran_script1")

# =================== PATHS ===================
HERE = os.path.abspath(os.path.dirname(__file__))

def find_project_root(start: str) -> str:
    """
    Walk upward from 'start' until we find a directory that contains
    the expected project folders (data/ and scripts/). Fallback to 'start'.
    """
    cur = os.path.abspath(start)
    while True:
        has_data = os.path.isdir(os.path.join(cur, "data"))
        has_scripts = os.path.isdir(os.path.join(cur, "scripts"))
        if has_data and has_scripts:
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:  # reached filesystem root
            return start
        cur = parent

PROJECT_ROOT = find_project_root(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
EXCEL_PATH = os.path.join(DATA_DIR, "passenger_car_data.xlsx")
SCRIPT1_FILENAME = "script1.py"

def find_logo_path() -> str | None:
    candidates = [
        os.path.join(PROJECT_ROOT, "ui", "assets", "icor-logo.png"),
        os.path.join(PROJECT_ROOT, "assets", "icor-logo.png"),
        os.path.join(PROJECT_ROOT, "icor-logo.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

LOGO_PATH = find_logo_path()
print(f"[CHECKPOINT] Paths | ROOT={PROJECT_ROOT} | DATA={DATA_DIR} | SCRIPTS={SCRIPTS_DIR} | LOGO={LOGO_PATH if LOGO_PATH else 'None'}")

# =================== OPTIONAL: POSTHOG ===================
def _safe_get(dict_like, dotted, default=None):
    try:
        cur = dict_like
        for part in dotted.split("."):
            cur = cur[part]
        return cur
    except Exception:
        return default

PH_KEY = _safe_get(st.secrets, "posthog.api_key")
PH_HOST = _safe_get(st.secrets, "posthog.host", "https://app.posthog.com")
if PH_KEY:
    posthog.project_api_key = PH_KEY
    posthog.host = PH_HOST
    print("[CHECKPOINT] PostHog configured")
else:
    print("[CHECKPOINT] PostHog not configured")

def track(event: str, props: dict | None = None):
    if not PH_KEY:
        return
    uid = st.session_state.get("user_id", "anon")
    try:
        posthog.capture(uid, event, properties=props or {})
    except Exception:
        pass

# =================== AUTH (CUSTOM LOGIN) ===================
USERS = st.secrets.get("users", {})
print(f"[CHECKPOINT] Users loaded: {list(USERS.keys()) if hasattr(USERS,'keys') else 'none'}")

def _verify_password(input_password: str, stored_password: str) -> bool:
    if not isinstance(stored_password, str):
        return False
    if stored_password.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            import bcrypt
        except Exception:
            st.error("bcrypt not installed but hashed password found. Add 'bcrypt' to requirements.txt or use plaintext in secrets.")
            return False
        try:
            return bcrypt.checkpw(input_password.encode("utf-8"), stored_password.encode("utf-8"))
        except Exception:
            return False
    return input_password == stored_password

def login_form():
    if LOGO_PATH:
        st.image(LOGO_PATH, width=160)
    st.title("Strategic Opportunities")
    st.caption("Please log in to continue.")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        if not username or not password:
            st.error("Please enter both username and password.")
            return False
        user_rec = USERS.get(username)
        if not user_rec:
            st.error("Invalid username or password.")
            track("login_failed", {"reason": "unknown_user"})
            return False
        name = user_rec.get("name", username)
        stored_pw = user_rec.get("password", "")
        if _verify_password(password, stored_pw):
            st.session_state["user_id"] = username
            st.session_state["user_name"] = name
            track("login_success", {"user": username})
            # One-shot rerun guard
            st.session_state["_just_logged_in"] = True
            st.rerun()
            return True
        else:
            st.error("Invalid username or password.")
            track("login_failed", {"reason": "bad_password"})
            return False
    return False

# Gate: require login
if "user_id" not in st.session_state:
    login_form()
    st.stop()

# =================== HEADER (logo + title + logout) ===================
col_logo, col_title, col_logout = st.columns([1, 5, 1])
with col_logo:
    if LOGO_PATH:
        st.image(LOGO_PATH, use_column_width=True)
with col_title:
    st.title("Strategic Opportunities")
    st.caption("ICOR – automatically perfect")
with col_logout:
    if st.button("Logout"):
        track("logout", {"user": st.session_state.get("user_id")})
        for k in ("user_id", "user_name"):
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["_just_logged_out"] = True  # one-shot guard
        st.rerun()

# =================== HELPERS ===================

def _timeout_communicate(proc: subprocess.Popen, timeout_sec: int = 420):
    """communicate() with timeout and friendly capture."""
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
        return out, proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        return "[ERROR] Script timed out.", 124

def run_script1():
    """Run backend Script 1 in DATA_DIR as a child process; capture combined stdout/stderr."""
    path = os.path.join(SCRIPTS_DIR, SCRIPT1_FILENAME)
    if not os.path.exists(path):
        return 127, f"[ERROR] Script not found: {path}"
    print("[CHECKPOINT] Running Script 1…")
    proc = subprocess.Popen(
        [sys.executable, path],
        cwd=DATA_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, code = _timeout_communicate(proc, timeout_sec=420)
    print("[CHECKPOINT] Script 1 finished with code", code)
    return code, out

@st.cache_data(show_spinner=False)
def _excel_sheet_names(xlsx_path: str) -> list[str]:
    """Cached list of sheet names (closes file handle)."""
    if not os.path.exists(xlsx_path):
        return []
    try:
        with pd.ExcelFile(xlsx_path) as xf:
            return xf.sheet_names
    except Exception:
        return []

def sheet_exists(name: str) -> bool:
    return name in _excel_sheet_names(EXCEL_PATH)

def read_sheet(name: str) -> pd.DataFrame:
    return pd.read_excel(EXCEL_PATH, sheet_name=name)

def detect_year_sheets(region: str) -> list[str]:
    names = _excel_sheet_names(EXCEL_PATH)
    pat = re.compile(r"^(20\d{2})\s+(EU|World)$")
    return sorted([s for s in names if pat.match(s) and s.endswith(region)])

def nice_table(df: pd.DataFrame):
    st.dataframe(df, use_container_width=True, height=550)

def first_available_icor() -> str | None:
    for s in ("ICOR_SO_All", "ICOR_SO_EU", "ICOR_SO_World"):
        if sheet_exists(s):
            return s
    return None

# =================== SIDEBAR ===================
with st.sidebar:
    st.header("Strategic Opportunities")
    st.caption(f"Logged in as **{st.session_state.get('user_name','')}**")
    st.markdown("---")
    if st.button("Run backend (Script 1)"):
        started_at = int(time.time())
        with st.status("Running backend…", expanded=False):
            code, log = run_script1()
        st.session_state["_script1_log"] = log
        outcome = "success" if code == 0 else "error"
        track("run_script1", {"status": outcome})
        if code != 0:
            st.error("Backend failed. See log below.")
        else:
            st.success("Backend finished. Reloading…")
            # One-shot guard to avoid rerun loops
            st.session_state["_just_ran_script1"] = True
            st.rerun()

    if os.path.exists(EXCEL_PATH):
        try:
            with open(EXCEL_PATH, "rb") as fh:
                st.download_button(
                    "Download workbook",
                    data=fh.read(),
                    file_name="passenger_car_data.xlsx",
                    use_container_width=True,
                )
        except Exception:
            pass

log = st.session_state.get("_script1_log")
if log:
    with st.expander("Backend run log", expanded=False):
        st.code(log)

# =================== LANDING: STRATEGIC TABLE ===================
if not os.path.exists(EXCEL_PATH):
    st.warning("No workbook found. Click **Run backend (Script 1)** in the sidebar.")
else:
    st.subheader("Strategic Opportunities")
    view = st.radio("View", ["Combined", "EU", "World"], horizontal=True)
    sheet_map = {"Combined": "ICOR_SO_All", "EU": "ICOR_SO_EU", "World": "ICOR_SO_World"}
    chosen_sheet = sheet_map[view]
    fallback = first_available_icor()
    if not sheet_exists(chosen_sheet):
        st.caption(f"`{chosen_sheet}` not found. Showing `{fallback}` instead.")
        chosen_sheet = fallback
    if not chosen_sheet:
        st.error("No ICOR strategic sheets found in workbook.")
    else:
        try:
            df_icor = read_sheet(chosen_sheet)
            nice_table(df_icor)
            track("view_icor_sheet", {"sheet": chosen_sheet})
        except Exception as e:
            st.error(f"Could not read `{chosen_sheet}`: {e}")

# =================== NAV CARDS (styled buttons) ===================
st.markdown(" ")
st.markdown("### Explore more")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚗  Fleet by Model / Year\n\nView EU or World", key="card_fleet"):
        st.session_state["section"] = "fleet"

with col2:
    if st.button("🛠️  Windshield Replacements\n\nView EU or World", key="card_repl"):
        st.session_state["section"] = "repl"

with col3:
    if st.button("📜  Historical Sales\n\nPick EU/World & Year", key="card_hist"):
        st.session_state["section"] = "history"

section = st.session_state.get("section")

# =================== SECTION: FLEET ===================
if section == "fleet":
    st.markdown("---")
    st.subheader("Fleet by Model / Year")
    region = st.radio("Region", ["EU", "World"], horizontal=True)
    sheet_name = f"Fleet_By_Model_Year_{region}"
    if not sheet_exists(sheet_name):
        st.warning(f"Sheet `{sheet_name}` not found.")
    else:
        try:
            nice_table(read_sheet(sheet_name))
            track("view_fleet", {"region": region})
        except Exception as e:
            st.error(f"Could not read `{sheet_name}`: {e}")

# =================== SECTION: REPLACEMENTS ===================
elif section == "repl":
    st.markdown("---")
    st.subheader("Windshield Replacements by Year")
    region = st.radio("Region", ["EU", "World"], horizontal=True)
    sheet_name = f"Windshield_Repl_By_Year_{region}"
    if not sheet_exists(sheet_name):
        st.warning(f"Sheet `{sheet_name}` not found.")
    else:
        try:
            nice_table(read_sheet(sheet_name))
            track("view_repl", {"region": region})
        except Exception as e:
            st.error(f"Could not read `{sheet_name}`: {e}")

# =================== SECTION: HISTORY ===================
elif section == "history":
    st.markdown("---")
    st.subheader("Historical Sales (Top 100)")
    region = st.radio("Region", ["EU", "World"], horizontal=True, key="hist_region")
    years = detect_year_sheets(region)
    if not years:
        st.warning(f"No per-year sheets for {region}.")
    else:
        year_options = sorted({int(s.split()[0]) for s in years})
        year = st.selectbox("Year", year_options, index=len(year_options)-1)
        sheet_name = f"{year} {region}"
        try:
            nice_table(read_sheet(sheet_name))
            track("view_history", {"region": region, "year": year})
        except Exception as e:
            st.error(f"Could not read `{sheet_name}`: {e}")

# =================== OPTIONAL: DEBUG HEARTBEAT ===================
# Uncomment to see if the script is truly rerendering too often.
# st.caption(f"Render #{st.session_state.setdefault('_renders', 0)}")
# st.session_state['_renders'] += 1

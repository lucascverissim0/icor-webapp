import os
import glob
import re
import subprocess
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import posthog
import time

# ────────────────────────── PATHS ──────────────────────────
HERE = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
SCRIPTS_DIR = os.path.abspath(os.path.join(HERE, "..", "scripts"))
EXCEL_PATH = os.path.join(DATA_DIR, "passenger_car_data.xlsx")
SCRIPT1_FILENAME = "script1.py"   # backend pipeline

print("[CHECKPOINT] Page config set")
st.set_page_config(page_title="ICOR – Decisions made simple", layout="wide")

# ───────────────────────── AUTH ─────────────────────────────
import copy

cfg = copy.deepcopy(st.secrets)  # 🔹 make a deep copy so authenticator can mutate safely

authenticator = stauth.Authenticate(
    credentials=cfg["credentials"],
    cookie_name=cfg["cookie"]["name"],
    cookie_key=cfg["cookie"]["key"],
    cookie_expiry_days=cfg["cookie"]["expiry_days"],
)

name, auth_status, username = authenticator.login("Login", "main")

if auth_status is False:
    st.error("Invalid username or password")
    st.stop()
elif auth_status is None:
    st.info("Please log in to continue")
    st.stop()

st.session_state["user_id"] = username
st.session_state["user_name"] = name

with st.sidebar:
    authenticator.logout("Logout", "sidebar")

# ─────────────────────── POSTHOG SETUP ───────────────────────
print("[CHECKPOINT] Posthog setup")
posthog.project_api_key = st.secrets.posthog.api_key
posthog.host = st.secrets.posthog.host

def track(event: str, props: dict | None = None):
    uid = st.session_state.get("user_id", "anon")
    posthog.capture(uid, event, properties=props or {})

# ─────────────────────── PAGE META/STYLE ───────────────────
st.markdown(
    """
    <style>
      html, body, .block-container { background-color: #0E1117 !important; color: #E6E6E6 !important; }
      .stDataFrame, .stMarkdown, .css-1v0mbdj, .css-10trblm { color: #E6E6E6 !important; }
      .card {
        background: #161A22; border: 1px solid #30363d; border-radius: 16px;
        padding: 18px; text-align: center; transition: transform 0.08s ease-in-out; cursor: pointer;
      }
      .card:hover { transform: translateY(-2px); border-color:#3b82f6; }
      .card-emoji { font-size: 34px; line-height: 1; }
      .card-title { font-size: 16px; margin-top: 8px; color:#E6E6E6; }
      .subtle { color:#A1A1AA; font-size: 13px; }
      .section-title { font-weight:700; margin-top: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ICOR – Decisions made simple")

# ───────────────────────── HELPERS ──────────────────────────
def run_script1():
    path = os.path.join(SCRIPTS_DIR, SCRIPT1_FILENAME)
    if not os.path.exists(path):
        return 127, f"[ERROR] Script not found: {path}"
    proc = subprocess.Popen(
        ["python", path],
        cwd=DATA_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = proc.communicate()
    return proc.returncode, out

def sheet_exists(name: str) -> bool:
    if not os.path.exists(EXCEL_PATH):
        return False
    try:
        return name in pd.ExcelFile(EXCEL_PATH).sheet_names
    except Exception:
        return False

def read_sheet(name: str) -> pd.DataFrame:
    return pd.read_excel(EXCEL_PATH, sheet_name=name)

def detect_year_sheets(region: str) -> list[str]:
    if not os.path.exists(EXCEL_PATH):
        return []
    xf = pd.ExcelFile(EXCEL_PATH)
    pat = re.compile(r"^(20\d{2})\s+(EU|World)$")
    return sorted([s for s in xf.sheet_names if pat.match(s) and s.endswith(region)])

def nice_table(df: pd.DataFrame):
    st.dataframe(df, use_container_width=True, height=550)

def first_available_icor() -> str | None:
    for s in ("ICOR_SO_All", "ICOR_SO_EU", "ICOR_SO_World"):
        if sheet_exists(s):
            return s
    return None

# ───────────────────────── SIDEBAR ──────────────────────────
with st.sidebar:
    st.header("Strategic Opportunities")
    st.caption("Default landing view.")
    st.markdown("---")
    if st.button("Run backend (Script 1)"):
        with st.status("Running backend…", expanded=False):
            code, log = run_script1()
        st.session_state["_script1_log"] = log
        if code != 0:
            st.error("Backend failed. See log below.")
            track("run_script1", {"status": "error", "timestamp": int(time.time())})
        else:
            st.success("Backend finished. Reloading…")
            track("run_script1", {"status": "success", "timestamp": int(time.time())})
            st.experimental_rerun()

    if os.path.exists(EXCEL_PATH):
        st.download_button(
            "Download workbook",
            data=open(EXCEL_PATH, "rb").read(),
            file_name="passenger_car_data.xlsx",
            use_container_width=True,
        )

log = st.session_state.get("_script1_log")
if log:
    with st.expander("Backend run log", expanded=False):
        st.code(log)

# ───────────────────── LANDING: STRATEGIC TABLE ──────────────────
if not os.path.exists(EXCEL_PATH):
    st.warning("No workbook found. Click **Run backend (Script 1)** in the sidebar.")
else:
    # Toggle right above the table (Combined / EU / World)
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
        df_icor = read_sheet(chosen_sheet)
        nice_table(df_icor)

# ───────────────── NAV CARDS (FLEET / REPL / HISTORY) ────────────
st.markdown(" ")
st.markdown("### Explore more")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button(" ", key="card_fleet", help="Fleet by Model/Year", use_container_width=True):
        st.session_state["section"] = "fleet"
    st.markdown('<div class="card"><div class="card-emoji">🚗</div>'
                '<div class="card-title">Fleet by Model / Year</div>'
                '<div class="subtle">View EU or World</div></div>', unsafe_allow_html=True)

with col2:
    if st.button(" ", key="card_repl", help="Windshield replacements", use_container_width=True):
        st.session_state["section"] = "repl"
    st.markdown('<div class="card"><div class="card-emoji">🛠️</div>'
                '<div class="card-title">Windshield Replacements</div>'
                '<div class="subtle">View EU or World</div></div>', unsafe_allow_html=True)

with col3:
    if st.button(" ", key="card_hist", help="Historical sales", use_container_width=True):
        st.session_state["section"] = "history"
    st.markdown('<div class="card"><div class="card-emoji">📜</div>'
                '<div class="card-title">Historical Sales</div>'
                '<div class="subtle">Pick EU/World & Year</div></div>', unsafe_allow_html=True)

section = st.session_state.get("section")

# ───────────────────── SECTION: FLEET BY MODEL/YEAR ──────────────
if section == "fleet":
    st.markdown("---")
    st.subheader("Fleet by Model / Year")
    region = st.radio("Region", ["EU", "World"], horizontal=True)
    sheet_name = f"Fleet_By_Model_Year_{region}"
    if not sheet_exists(sheet_name):
        st.warning(f"Sheet `{sheet_name}` not found.")
    else:
        nice_table(read_sheet(sheet_name))

# ───────────────────── SECTION: WINDSHIELD REPLACEMENTS ──────────
elif section == "repl":
    st.markdown("---")
    st.subheader("Windshield Replacements by Year")
    region = st.radio("Region", ["EU", "World"], horizontal=True)
    sheet_name = f"Windshield_Repl_By_Year_{region}"
    if not sheet_exists(sheet_name):
        st.warning(f"Sheet `{sheet_name}` not found.")
    else:
        nice_table(read_sheet(sheet_name))

# ───────────────────── SECTION: HISTORICAL SALES ─────────────────
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
        nice_table(read_sheet(sheet_name))

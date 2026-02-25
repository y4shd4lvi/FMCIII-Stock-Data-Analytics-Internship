"""
myProjectFile — NSE Stock Analysis Dashboard
Main entry point. Run with: streamlit run app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="NSE Analytica",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0a0e17;
    --surface:   #111827;
    --surface2:  #1a2235;
    --border:    #1e2d45;
    --accent:    #00d4aa;
    --accent2:   #f59e0b;
    --accent3:   #3b82f6;
    --danger:    #ef4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Bebas Neue', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* ── App shell ── */
.stApp { background: var(--bg); color: var(--text); font-family: var(--font-body); }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: var(--font-body) !important; }

/* ── Sidebar radio nav ── */
[data-testid="stSidebar"] .stRadio label {
    display: block;
    padding: 0.6rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--muted);
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: var(--surface2); color: var(--text); }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0a0e17 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.03em;
}
.stButton > button:hover {
    background: #00bfa0 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,212,170,0.25) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 6px !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(0,212,170,0.08) !important;
}

/* ── Text inputs / selects / date pickers ── */
.stTextInput input, .stSelectbox select,
[data-testid="stDateInput"] input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.88rem !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--font-mono) !important; }

/* ── Status / alerts ── */
.stAlert { border-radius: 8px !important; font-family: var(--font-body) !important; }
[data-testid="stSuccess"]  { border-left: 3px solid var(--accent)  !important; background: rgba(0,212,170,0.07) !important; }
[data-testid="stInfo"]     { border-left: 3px solid var(--accent3) !important; background: rgba(59,130,246,0.07) !important; }
[data-testid="stWarning"]  { border-left: 3px solid var(--accent2) !important; background: rgba(245,158,11,0.07) !important; }
[data-testid="stError"]    { border-left: 3px solid var(--danger)  !important; background: rgba(239,68,68,0.07) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border) !important; gap: 0; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Custom component classes ── */
.page-header {
    font-family: var(--font-head);
    font-size: 2.8rem;
    letter-spacing: 0.06em;
    line-height: 1;
    color: var(--text);
    margin-bottom: 0.2rem;
}
.page-sub {
    font-family: var(--font-body);
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 1.5rem;
    letter-spacing: 0.02em;
}
.section-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}
.card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.tag {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    background: rgba(0,212,170,0.12);
    color: var(--accent);
    letter-spacing: 0.05em;
    margin-right: 0.3rem;
}
.log-box {
    background: #070c14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: #7dd3c4;
    max-height: 280px;
    overflow-y: auto;
    line-height: 1.7;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem 0;'>
        <div style='font-family:"Bebas Neue",sans-serif; font-size:1.8rem;
                    letter-spacing:0.1em; color:#00d4aa; line-height:1;'>
            NSE<br>ANALYTICA
        </div>
        <div style='font-family:"DM Mono",monospace; font-size:0.65rem;
                    color:#64748b; letter-spacing:0.12em; margin-top:0.3rem;'>
            NIFTY 50 · MARKET INTELLIGENCE
        </div>
    </div>
    <hr style='margin-bottom:1rem;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "🏠  Overview",
            "📥  NSE Downloader",
            "📊  Fundamental Analysis",
            "📦  Volume Analysis",
            "📈  Trend Analysis",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"DM Mono",monospace; font-size:0.65rem;
                color:#2d3f55; padding: 0 0.5rem; line-height:1.8;'>
        DATA  →  data/raw/<br>
        PROCESSED  →  data/processed/<br>
        REPORTS  →  data/reports/
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────────────────────
if   "Overview"       in page: from pages.p0_overview     import render; render()
elif "NSE Downloader" in page: from pages.p1_downloader   import render; render()
elif "Fundamental"    in page: from pages.p2_fundamental  import render; render()
elif "Volume"         in page: from pages.p3_volume       import render; render()
elif "Trend"          in page: from pages.p4_trend        import render; render()

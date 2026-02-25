"""
Overview page — project summary and quick status check.
"""

import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def _file_status(path: Path) -> tuple[str, str]:
    """Return (icon, size_str) for a file."""
    if path.exists():
        size = path.stat().st_size
        if size > 1_000_000:
            return "✅", f"{size/1_000_000:.1f} MB"
        elif size > 1_000:
            return "✅", f"{size/1_000:.1f} KB"
        else:
            return "✅", f"{size} B"
    return "⬜", "not found"


def render():
    st.markdown('<div class="page-header">OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Project status · file inventory · quick navigation</div>',
        unsafe_allow_html=True,
    )

    # ── Key file status ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Data Files</div>', unsafe_allow_html=True)

    files = {
        "bhavcopy_master.csv":                      BASE_DIR / "data" / "raw"       / "bhavcopy_master.csv",
        "Top_50_Companies_Data.csv":         BASE_DIR / "data" / "raw"       / "Top_50_Companies_Data.csv",
        "strong_fundamental_companies.csv":  BASE_DIR / "data" / "processed" / "strong_fundamental_companies.csv",
        "nifty50_volume_analysis_report.csv":BASE_DIR / "data" / "processed" / "nifty50_volume_analysis_report.csv",
    }

    cols = st.columns(4)
    for col, (name, path) in zip(cols, files.items()):
        icon, size = _file_status(path)
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:1rem;">
                <div style="font-size:1.6rem; margin-bottom:0.4rem;">{icon}</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                            color:#64748b; word-break:break-all;">{name}</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.75rem;
                            color:#00d4aa; margin-top:0.3rem;">{size}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Reports status ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Generated Reports</div>', unsafe_allow_html=True)

    report_dirs = {
        "Fundamental Charts":  BASE_DIR / "data" / "reports" / "fundamental",
        "Volume Charts":       BASE_DIR / "data" / "reports" / "volume",
        "Trend Reports":       BASE_DIR / "data" / "reports" / "trend" / "reports",
        "Trend Charts":        BASE_DIR / "data" / "reports" / "trend" / "charts",
    }

    cols2 = st.columns(4)
    for col, (label, dpath) in zip(cols2, report_dirs.items()):
        if dpath.exists():
            count = len(list(dpath.iterdir()))
            icon, detail = ("📁", f"{count} file(s)") if count else ("📂", "empty")
        else:
            icon, detail = "⬜", "not created"
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:1rem;">
                <div style="font-size:1.6rem; margin-bottom:0.4rem;">{icon}</div>
                <div style="font-family:'DM Sans',sans-serif; font-size:0.78rem;
                            color:#e2e8f0;">{label}</div>
                <div style="font-family:'DM Mono',monospace; font-size:0.72rem;
                            color:#64748b; margin-top:0.3rem;">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Pipeline guide ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Recommended Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("01", "NSE Downloader",      "Download + preprocess bhavcopy.csv",           "#00d4aa"),
        ("02", "Fundamental Analysis","Score & filter top NIFTY 50 companies",        "#3b82f6"),
        ("03", "Volume Analysis",     "OBV, A/D line, breakout detection",            "#f59e0b"),
        ("04", "Trend Analysis",      "MA crossovers, HH/HL structure per stock",     "#a78bfa"),
    ]

    step_cols = st.columns(4)
    for col, (num, title, desc, color) in zip(step_cols, steps):
        with col:
            st.markdown(f"""
            <div class="card" style="border-left: 3px solid {color};">
                <div style="font-family:'Bebas Neue',sans-serif; font-size:2rem;
                            color:{color}; opacity:0.4; line-height:1;">{num}</div>
                <div style="font-family:'DM Sans',sans-serif; font-weight:600;
                            font-size:0.88rem; color:#e2e8f0; margin-top:0.2rem;">{title}</div>
                <div style="font-family:'DM Sans',sans-serif; font-size:0.78rem;
                            color:#64748b; margin-top:0.3rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

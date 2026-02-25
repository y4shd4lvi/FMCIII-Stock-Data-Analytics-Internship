"""
Volume Analysis page — run volume analyzer, view report table, download results.
"""

import streamlit as st
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.volume_analysis.volume import Nifty50VolumeAnalyzer

INPUT_CSV  = BASE_DIR / "data" / "raw"       / "bhavcopy_master.csv"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "nifty50_volume_analysis_report.csv"

# Colour maps for recommendation badges
REC_COLORS = {
    "STRONG_BUY":  ("#00d4aa", "#0a3d35"),
    "BUY":         ("#3b82f6", "#0d1f3c"),
    "HOLD":        ("#f59e0b", "#3d2d06"),
    "SELL":        ("#f97316", "#3d1a06"),
    "STRONG_SELL": ("#ef4444", "#3d0a0a"),
}


def _rec_badge(rec: str) -> str:
    fg, bg = REC_COLORS.get(rec, ("#64748b", "#1a2235"))
    return (
        f'<span style="background:{bg}; color:{fg}; border:1px solid {fg}; '
        f'padding:2px 8px; border-radius:4px; font-family:\'DM Mono\',monospace; '
        f'font-size:0.72rem; letter-spacing:0.05em;">{rec}</span>'
    )


def render():
    st.markdown('<div class="page-header">VOLUME ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">OBV · A/D line · breakout detection · accumulation signals</div>',
        unsafe_allow_html=True,
    )

    # ── Pre-flight ────────────────────────────────────────────────────
    if not INPUT_CSV.exists():
        st.error("❌ `bhavcopy.csv` not found in `data/raw/`.  \nRun **NSE Downloader** first.")
        return

    # ── Run control ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">Run Analysis</div>', unsafe_allow_html=True)

    if st.button("▶  Run Volume Analysis", use_container_width=False):
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("Analyzing volume patterns for all NIFTY 50 stocks…"):
            try:
                analyzer  = Nifty50VolumeAnalyzer(INPUT_CSV)
                results   = analyzer.analyze_all_stocks()
                analyzer.generate_report(results, OUTPUT_CSV)
                st.success(f"✅ Volume analysis complete — {len(results)} stocks analyzed.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────
    if OUTPUT_CSV.exists():
        import pandas as pd
        df = pd.read_csv(OUTPUT_CSV)

        # Summary metrics
        st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Stocks",      len(df))
        m2.metric("Strong Buy",  len(df[df["RECOMMENDATION"] == "STRONG_BUY"]))
        m3.metric("Buy",         len(df[df["RECOMMENDATION"] == "BUY"]))
        m4.metric("Strong Sell", len(df[df["RECOMMENDATION"] == "STRONG_SELL"]))
        m5.metric("High Risk",   len(df[df["RISK_LEVEL"] == "HIGH"]))

        st.markdown("<br>", unsafe_allow_html=True)

        # Filters
        st.markdown('<div class="section-label">Filter</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        rec_filter  = fc1.multiselect("Recommendation", sorted(df["RECOMMENDATION"].unique()),  default=list(df["RECOMMENDATION"].unique()))
        risk_filter = fc2.multiselect("Risk Level",     sorted(df["RISK_LEVEL"].unique()),       default=list(df["RISK_LEVEL"].unique()))
        trend_filter= fc3.multiselect("Trend",          sorted(df["TREND"].unique()),            default=list(df["TREND"].unique()))

        filtered = df[
            df["RECOMMENDATION"].isin(rec_filter) &
            df["RISK_LEVEL"].isin(risk_filter) &
            df["TREND"].isin(trend_filter)
        ]

        st.markdown(f"<div class='section-label'>{len(filtered)} stocks matched</div>", unsafe_allow_html=True)

        # Display columns
        show_cols = [c for c in [
            "SECURITY", "CLOSE_PRICE", "PRICE_CHANGE_%", "RECOMMENDATION",
            "RISK_LEVEL", "TREND", "VOL_RATIO_20D", "VOL_SPIKE",
            "PROFIT_PROBABILITY", "BREAKOUT_STATUS", "ACCUMULATION",
        ] if c in filtered.columns]

        st.dataframe(filtered[show_cols], use_container_width=True, height=450)

        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "⬇  Download Full Report CSV",
            data=OUTPUT_CSV.read_bytes(),
            file_name="nifty50_volume_analysis_report.csv",
            mime="text/csv",
        )

        # Top opportunities detail
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Top 5 Strong Buy — Detail</div>', unsafe_allow_html=True)

        top5 = df[df["RECOMMENDATION"] == "STRONG_BUY"].head(5)
        if len(top5) == 0:
            st.info("No STRONG_BUY signals at this time.")
        else:
            for _, row in top5.iterrows():
                with st.expander(f"{row['SECURITY']}  —  ₹{row['CLOSE_PRICE']:.2f}"):
                    d1, d2, d3 = st.columns(3)
                    d1.markdown(f"**Recommendation** {_rec_badge(row['RECOMMENDATION'])}", unsafe_allow_html=True)
                    d2.metric("Profit Probability", row.get("PROFIT_PROBABILITY","—"))
                    d3.metric("Volume Ratio (20D)", f"{row.get('VOL_RATIO_20D',0):.2f}x")

                    e1, e2, e3 = st.columns(3)
                    e1.metric("Risk Level",      row.get("RISK_LEVEL","—"))
                    e2.metric("Trend",           row.get("TREND","—"))
                    e3.metric("Breakout Status", row.get("BREAKOUT_STATUS","—"))

                    st.markdown(
                        f"**Accumulation:** {row.get('ACCUMULATION','—')} &nbsp;|&nbsp; "
                        f"**OBV Trend:** {row.get('OBV_TREND','—')} &nbsp;|&nbsp; "
                        f"**A/D Trend:** {row.get('AD_TREND','—')}",
                        unsafe_allow_html=True,
                    )

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:2rem; border-style:dashed;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">📦</div>
            <div style="color:#64748b; font-size:0.88rem;">
                No results yet — click <b style="color:#e2e8f0;">Run Volume Analysis</b> to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)

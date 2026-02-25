"""
Trend Analysis page — run per-stock trend analysis, browse reports & charts.
"""

import streamlit as st
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.trend_analysis.trend import Nifty50TrendAnalyzer

INPUT_CSV    = BASE_DIR / "data" / "raw"     / "bhavcopy_master.csv"
REPORTS_DIR  = BASE_DIR / "data" / "reports" / "trend" / "reports"
CHARTS_DIR   = BASE_DIR / "data" / "reports" / "trend" / "charts"
SUMMARY_CSV  = BASE_DIR / "data" / "reports" / "trend" / "NIFTY50_SUMMARY.csv"
SUMMARY_TXT  = BASE_DIR / "data" / "reports" / "trend" / "NIFTY50_SUMMARY.txt"

SCORE_COLORS = {
    "🟢 STRONG UPTREND":           "#00d4aa",
    "🟡 WEAK/DEVELOPING UPTREND":  "#f59e0b",
    "⚪ SIDEWAYS / RANGE-BOUND":   "#94a3b8",
    "🟠 WEAK DOWNTREND":           "#f97316",
    "🔴 STRONG DOWNTREND":         "#ef4444",
}


def render():
    st.markdown('<div class="page-header">TREND ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">MA crossovers · HH/HL structure · golden/death cross · per-stock reports</div>',
        unsafe_allow_html=True,
    )

    # ── Pre-flight ────────────────────────────────────────────────────
    if not INPUT_CSV.exists():
        st.error("❌ `bhavcopy_master.csv` not found in `data/raw/`.  \nRun **NSE Downloader** first.")
        return

    # ── Run control ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">Run Analysis</div>', unsafe_allow_html=True)
    st.info("⏱️  Trend analysis generates 50 individual reports + charts. This may take 2–5 minutes.")

    if st.button("▶  Run Trend Analysis (All NIFTY 50)", use_container_width=False):
        with st.spinner("Analyzing trends for all NIFTY 50 stocks… please wait."):
            try:
                analyzer = Nifty50TrendAnalyzer(INPUT_CSV)
                summary  = analyzer.analyze_all_nifty50()
                st.success(f"✅ Trend analysis complete — {len(summary)} stocks processed.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────
    if SUMMARY_CSV.exists():
        import pandas as pd
        summary_df = pd.read_csv(SUMMARY_CSV)

        # Summary metrics
        st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Stocks Analyzed", len(summary_df))
        strong_up   = len(summary_df[summary_df["Classification"].str.contains("STRONG UPTREND",   na=False)])
        strong_down = len(summary_df[summary_df["Classification"].str.contains("STRONG DOWNTREND", na=False)])
        sideways    = len(summary_df[summary_df["Classification"].str.contains("SIDEWAYS",         na=False)])
        m2.metric("Strong Uptrend",   strong_up)
        m3.metric("Sideways",         sideways)
        m4.metric("Strong Downtrend", strong_down)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs — Summary table | Browse Reports | Browse Charts | Summary text
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋  Summary Table",
            "📄  Stock Reports",
            "📊  Stock Charts",
            "📝  Full Summary",
        ])

        # ── Tab 1: Summary table ──────────────────────────────────────
        with tab1:
            class_filter = st.multiselect(
                "Filter by Classification",
                sorted(summary_df["Classification"].unique()),
                default=list(summary_df["Classification"].unique()),
            )
            filtered = summary_df[summary_df["Classification"].isin(class_filter)]
            st.dataframe(filtered, use_container_width=True, height=440)

            st.download_button(
                "⬇  Download Summary CSV",
                data=SUMMARY_CSV.read_bytes(),
                file_name="NIFTY50_SUMMARY.csv",
                mime="text/csv",
            )

        # ── Tab 2: Per-stock text reports ─────────────────────────────
        with tab2:
            report_files = sorted(REPORTS_DIR.glob("*.txt")) if REPORTS_DIR.exists() else []

            if not report_files:
                st.info("No text reports found — run analysis first.")
            else:
                # Build a display name → file map
                stock_names = {f.stem.replace("_report", "").replace("_", " "): f for f in report_files}

                # Sort by score using summary CSV
                score_map = dict(zip(
                    summary_df["Security"].str.replace(" ", "_").str.replace(".", "").str.replace("&", "AND"),
                    summary_df["Score"],
                )) if "Security" in summary_df.columns else {}

                col_search, col_sort = st.columns([2, 1])
                search = col_search.text_input("Search stock", placeholder="e.g. RELIANCE")
                sort_by_score = col_sort.checkbox("Sort by score", value=True)

                filtered_names = {
                    k: v for k, v in stock_names.items()
                    if search.upper() in k.upper()
                }

                if sort_by_score and "Score" in summary_df.columns:
                    def _score(name):
                        key = name.replace(" ", "_").upper()
                        return score_map.get(key, 0)
                    filtered_names = dict(
                        sorted(filtered_names.items(), key=lambda x: _score(x[0]), reverse=True)
                    )

                if not filtered_names:
                    st.warning("No reports match your search.")
                else:
                    chosen_name = st.selectbox("Select Stock", list(filtered_names.keys()))
                    chosen_file = filtered_names[chosen_name]

                    report_text = chosen_file.read_text(encoding="utf-8")

                    st.markdown(
                        f'<div class="log-box" style="max-height:500px; color:#c8e6df;">'
                        f'{report_text}</div>',
                        unsafe_allow_html=True,
                    )

                    st.download_button(
                        f"⬇  Download {chosen_name} Report",
                        data=report_text.encode("utf-8"),
                        file_name=chosen_file.name,
                        mime="text/plain",
                    )

        # ── Tab 3: Per-stock charts ───────────────────────────────────
        with tab3:
            chart_files = sorted(CHARTS_DIR.glob("*.png")) if CHARTS_DIR.exists() else []

            if not chart_files:
                st.info("No charts found — run analysis first.")
            else:
                chart_names = {f.stem.replace("_chart", "").replace("_", " "): f for f in chart_files}

                search_c = st.text_input("Search stock ", placeholder="e.g. TCS", key="chart_search")
                filtered_charts = {k: v for k, v in chart_names.items() if search_c.upper() in k.upper()}

                if not filtered_charts:
                    st.warning("No charts match your search.")
                else:
                    chosen_chart_name = st.selectbox("Select Stock ", list(filtered_charts.keys()))
                    chosen_chart_file = filtered_charts[chosen_chart_name]

                    st.image(str(chosen_chart_file), use_column_width=True)

                    st.download_button(
                        f"⬇  Download {chosen_chart_name} Chart",
                        data=chosen_chart_file.read_bytes(),
                        file_name=chosen_chart_file.name,
                        mime="image/png",
                    )

        # ── Tab 4: Full summary text ──────────────────────────────────
        with tab4:
            if SUMMARY_TXT.exists():
                summary_text = SUMMARY_TXT.read_text(encoding="utf-8")
                st.markdown(
                    f'<div class="log-box" style="max-height:600px; color:#c8e6df;">'
                    f'{summary_text}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "⬇  Download Summary Text",
                    data=summary_text.encode("utf-8"),
                    file_name="NIFTY50_SUMMARY.txt",
                    mime="text/plain",
                )
            else:
                st.info("Summary text not found — run analysis first.")

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:2rem; border-style:dashed;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">📈</div>
            <div style="color:#64748b; font-size:0.88rem;">
                No results yet — click <b style="color:#e2e8f0;">Run Trend Analysis</b> to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)

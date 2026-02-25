"""
Fundamental Analysis page — run scorer, view charts, download results.
"""

import streamlit as st
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.fundamental_analysis.fundamental import FundamentalAnalyzer
from src.fundamental_analysis.advanced    import AdvancedAnalyzer

INPUT_CSV   = BASE_DIR / "data" / "raw"       / "Top_50_Companies_Data.csv"
OUTPUT_CSV  = BASE_DIR / "data" / "processed" / "strong_fundamental_companies.csv"
REPORTS_DIR = BASE_DIR / "data" / "reports"   / "fundamental"

CHART_MAP = {
    "Main Analysis":        REPORTS_DIR / "fundamental_analysis_charts.png",
    "Correlation Heatmap":  REPORTS_DIR / "correlation_heatmap.png",
    "Risk / Return":        REPORTS_DIR / "risk_return_analysis.png",
    "Sector Comparison":    REPORTS_DIR / "sector_comparison.png",
    "Valuation Analysis":   REPORTS_DIR / "valuation_analysis.png",
    "Growth Momentum":      REPORTS_DIR / "growth_momentum_analysis.png",
}


def render():
    st.markdown('<div class="page-header">FUNDAMENTAL ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Score & filter NIFTY 50 companies · advanced multi-factor analysis</div>',
        unsafe_allow_html=True,
    )

    # ── Pre-flight check ──────────────────────────────────────────────
    if not INPUT_CSV.exists():
        st.error(f"❌ Input file not found: `{INPUT_CSV}`  \nPlease place `Top_50_Companies_Data.csv` in `data/raw/`.")
        return

    # ── Run controls ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Run Analysis</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    run_basic    = col_a.button("▶  Run Basic Analysis",    use_container_width=True)
    run_advanced = col_b.button("▶  Run Advanced Analysis", use_container_width=True)

    if run_basic:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("Running fundamental analysis…"):
            try:
                analyzer = FundamentalAnalyzer(INPUT_CSV)
                analyzer.calculate_fundamental_score()
                analyzer.get_summary_statistics()
                analyzer.get_top_performers()
                analyzer.generate_detailed_report()
                analyzer.create_visualizations(REPORTS_DIR / "fundamental_analysis_charts.png")
                analyzer.export_results(OUTPUT_CSV)
                st.success("✅ Basic analysis complete — results saved.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

    if run_advanced:
        if not OUTPUT_CSV.exists():
            st.warning("⚠️  Run Basic Analysis first to generate the processed CSV.")
        else:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            with st.spinner("Running advanced analysis…"):
                try:
                    import pandas as pd
                    df = pd.read_csv(OUTPUT_CSV)
                    adv = AdvancedAnalyzer(df)
                    adv.correlation_analysis(REPORTS_DIR / "correlation_heatmap.png")
                    adv.risk_return_analysis(REPORTS_DIR / "risk_return_analysis.png")
                    adv.sector_comparative_analysis(REPORTS_DIR / "sector_comparison.png")
                    adv.valuation_analysis(REPORTS_DIR / "valuation_analysis.png")
                    adv.growth_momentum_analysis(REPORTS_DIR / "growth_momentum_analysis.png")
                    st.success("✅ Advanced analysis complete — charts saved.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results section (only if outputs exist) ───────────────────────
    if OUTPUT_CSV.exists():
        import pandas as pd
        df = pd.read_csv(OUTPUT_CSV)

        # Metrics row
        st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Strong Companies", len(df))
        m2.metric("Avg ROCE",  f"{df['ROCE %'].mean():.1f}%"      if 'ROCE %'          in df.columns else "—")
        m3.metric("Avg P/E",   f"{df['P/E'].mean():.1f}"           if 'P/E'             in df.columns else "—")
        m4.metric("Avg Div Yield", f"{df['Div Yld %'].mean():.2f}%" if 'Div Yld %'      in df.columns else "—")

        st.markdown("<br>", unsafe_allow_html=True)

        # Data table
        tab1, tab2 = st.tabs(["📋  Data Table", "📊  Charts"])

        with tab1:
            st.markdown('<div class="section-label">Filtered Companies</div>', unsafe_allow_html=True)
            display_cols = [c for c in ["Name","CMP Rs.","P/E","ROCE %","Qtr Profit Var %",
                                         "Qtr Sales Var %","Div Yld %","Fundamental_Score"]
                            if c in df.columns]
            st.dataframe(
                df[display_cols].style.background_gradient(
                    subset=["Fundamental_Score"] if "Fundamental_Score" in display_cols else [],
                    cmap="YlGn",
                ),
                use_container_width=True,
                height=420,
            )
            st.download_button(
                "⬇  Download CSV",
                data=OUTPUT_CSV.read_bytes(),
                file_name="strong_fundamental_companies.csv",
                mime="text/csv",
            )

        with tab2:
            available = {k: v for k, v in CHART_MAP.items() if v.exists()}
            if not available:
                st.info("No charts yet — run an analysis above.")
            else:
                chosen = st.selectbox("Select Chart", list(available.keys()))
                chart_path = available[chosen]
                st.image(str(chart_path), use_column_width=True)
                st.download_button(
                    f"⬇  Download {chosen}",
                    data=chart_path.read_bytes(),
                    file_name=chart_path.name,
                    mime="image/png",
                )

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:2rem; border-style:dashed;">
            <div style="font-size:2rem; margin-bottom:0.5rem;">📊</div>
            <div style="color:#64748b; font-size:0.88rem;">
                No results yet — click <b style="color:#e2e8f0;">Run Basic Analysis</b> to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)

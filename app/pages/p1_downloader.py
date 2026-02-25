"""
NSE Downloader page — download bhavcopy + run preprocessor.
"""

import streamlit as st
from datetime import date, timedelta
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.nse.downloader  import NSEBhavcopyDownloader
from src.nse.preprocess  import BhavcopyPreprocessor


def render():
    st.markdown('<div class="page-header">NSE DOWNLOADER</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Download bhavcopy data from NSE · preprocess into master CSV</div>',
        unsafe_allow_html=True,
    )

    bhavcopy_path = BASE_DIR / "data" / "raw" / "bhavcopy_master.csv"

    # ── Current master status ─────────────────────────────────────────
    st.markdown('<div class="section-label">Master File Status</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if bhavcopy_path.exists():
        import pandas as pd
        try:
            df_info = pd.read_csv(bhavcopy_path, nrows=0)
            full    = pd.read_csv(bhavcopy_path)
            rows    = len(full)
            dates   = pd.to_datetime(full["TRADE_DATE"], errors="coerce")
            min_d   = dates.min().strftime("%d %b %Y") if not dates.isna().all() else "—"
            max_d   = dates.max().strftime("%d %b %Y") if not dates.isna().all() else "—"
        except Exception:
            rows, min_d, max_d = "?", "?", "?"
        c1.metric("Status",     "✅ Found")
        c2.metric("Total Rows", f"{rows:,}" if isinstance(rows, int) else rows)
        c3.metric("Date Range", f"{min_d} → {max_d}")
    else:
        c1.metric("Status",     "⬜ Not Found")
        c2.metric("Total Rows", "—")
        c3.metric("Date Range", "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download form ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Download Range</div>', unsafe_allow_html=True)

    with st.form("download_form"):
        col1, col2 = st.columns(2)
        with col1:
            start_dt = st.date_input(
                "Start Date",
                value=date.today() - timedelta(days=7),
                max_value=date.today(),
            )
        with col2:
            end_dt = st.date_input(
                "End Date",
                value=date.today(),
                max_value=date.today(),
            )

        delete_after = st.checkbox(
            "Delete daily files after appending to master",
            value=True,
            help="Keeps data/raw/ clean — daily CSVs are removed once merged.",
        )
        run_preprocess = st.checkbox(
            "Run preprocessor after download",
            value=True,
            help="Converts dates, numerics, and categoricals in bhavcopy.csv",
        )

        submitted = st.form_submit_button("🚀  Run Download + Append")

    if submitted:
        start_str = start_dt.strftime("%d-%b-%Y")
        end_str   = end_dt.strftime("%d-%b-%Y")

        log_lines: list[str] = []
        log_placeholder = st.empty()

        def log_cb(msg: str):
            log_lines.append(msg)
            log_placeholder.markdown(
                '<div class="log-box">' + "\n".join(log_lines) + "</div>",
                unsafe_allow_html=True,
            )

        # Step 1 — Download
        with st.spinner("Downloading bhavcopy files…"):
            try:
                downloader = NSEBhavcopyDownloader(log_callback=log_cb)
                master_path = downloader.run(
                    start_date=start_str,
                    end_date=end_str,
                    delete_after_append=delete_after,
                )
                st.success(f"✅ Master CSV updated → `{master_path.name}`")
            except Exception as e:
                st.error(f"❌ Download failed: {e}")
                return

        # Step 2 — Preprocess
        if run_preprocess:
            with st.spinner("Preprocessing bhavcopy.csv…"):
                try:
                    pre = BhavcopyPreprocessor(log_callback=log_cb)
                    pre.process()
                    pre.save()
                    st.success("✅ Preprocessing complete")
                except Exception as e:
                    st.error(f"❌ Preprocessing failed: {e}")

        # Download button for master CSV
        if bhavcopy_path.exists():
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
            st.download_button(
                label="⬇  Download bhavcopy.csv",
                data=bhavcopy_path.read_bytes(),
                file_name="bhavcopy.csv",
                mime="text/csv",
            )

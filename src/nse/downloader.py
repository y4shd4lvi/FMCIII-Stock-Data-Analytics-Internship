import requests
import pandas as pd
import zipfile
import io
import os
from datetime import datetime, timedelta


class NSEBhavcopyDownloader:
    REPORT_URL = "https://www.nseindia.com/api/reports"
    HOME_URL = "https://www.nseindia.com"

    def __init__(
        self,
        daily_dir="bhavcopies",
        final_dir="final_bhavcopy",
        timeout=15
    ):
        self.daily_dir = daily_dir
        self.final_dir = final_dir
        self.timeout = timeout

        os.makedirs(self.daily_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)

        self.session = requests.Session()
        self._setup_session()

    # ---------------- SESSION SETUP (same as original) ----------------
    def _setup_session(self):
        self.session.get(
            self.HOME_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )

    # ---------------- REQUEST PARAMS (same as original) ----------------
    def _headers(self):
        return {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com/all-reports",
            "X-Requested-With": "XMLHttpRequest"
        }

    def _params(self, date_str):
        return {
            "archives": '[{"name":"CM - Bhavcopy (PR.zip)","type":"archives","category":"capital-market","section":"equities"}]',
            "date": date_str,
            "type": "equities",
            "mode": "single"
        }

    # ---------------- ZIP EXTRACTION (IDENTICAL LOGIC) ----------------
    def _extract_pr_csv(self, zip_bytes: bytes) -> pd.DataFrame | None:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            files = z.namelist()

            valid_files = [
                f for f in files
                if f.lower().startswith(("pr", "cm"))
                and f.lower().endswith(".csv")
            ]

            if not valid_files:
                return None

            with z.open(valid_files[0]) as f:
                return pd.read_csv(
                    f,
                    engine="python",
                    on_bad_lines="skip"
                )

    # ---------------- DOWNLOAD DAILY RANGE ----------------
    def download_range(self, start_date: str, end_date: str):
        start = datetime.strptime(start_date, "%d-%b-%Y")
        end = datetime.strptime(end_date, "%d-%b-%Y")

        current = start

        while current <= end:
            date_str = current.strftime("%d-%b-%Y")
            print(f"\n📅 Processing {date_str}")

            try:
                resp = self.session.get(
                    self.REPORT_URL,
                    headers=self._headers(),
                    params=self._params(date_str),
                    timeout=self.timeout
                )

                resp.raise_for_status()

                df = self._extract_pr_csv(resp.content)

                if df is None:
                    print("⚠️ No bhavcopy (holiday / no trading)")
                    current += timedelta(days=1)
                    continue

                out_file = os.path.join(
                    self.daily_dir, f"bhavcopy_{date_str}.csv"
                )

                df.to_csv(out_file, index=False)
                print(f"✅ Saved {out_file} | Rows: {df.shape[0]}")

            except Exception as e:
                print(f"❌ Error {date_str}: {e}")

            current += timedelta(days=1)

    # ---------------- INCREMENTAL APPEND ----------------
    def incremental_append(
        self,
        output_filename="bhavcopy_master.csv"
    ) -> str:

        output_path = os.path.join(self.final_dir, output_filename)

        if os.path.exists(output_path):
            master_df = pd.read_csv(
                output_path,
                engine="python",
                on_bad_lines="skip"
            )
            existing_dates = set(master_df["TRADE_DATE"].astype(str))
            print(f"📘 Existing dates: {len(existing_dates)}")
        else:
            master_df = None
            existing_dates = set()
            print("📘 Creating new master file")

        new_dfs = []

        for file in sorted(os.listdir(self.daily_dir)):
            if not file.lower().endswith(".csv"):
                continue

            try:
                date_str = file.replace("bhavcopy_", "").replace(".csv", "")
                trade_date = datetime.strptime(date_str, "%d-%b-%Y")
                trade_date_str = trade_date.strftime("%Y-%m-%d")
            except Exception:
                continue

            if trade_date_str in existing_dates:
                continue

            file_path = os.path.join(self.daily_dir, file)
            df = pd.read_csv(
                file_path,
                engine="python",
                on_bad_lines="skip"
            )

            df["TRADE_DATE"] = trade_date_str
            new_dfs.append(df)

        if not new_dfs:
            print("✅ No new data to append")
            return output_path

        new_data = pd.concat(new_dfs, ignore_index=True)

        if master_df is not None:
            final_df = pd.concat(
                [master_df, new_data], ignore_index=True
            )
        else:
            final_df = new_data

        final_df.sort_values("TRADE_DATE", inplace=True)
        final_df.to_csv(output_path, index=False)

        print(f"✅ Appended rows: {new_data.shape[0]}")
        return output_path

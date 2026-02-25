import pandas as pd
from pathlib import Path


class BhavcopyPreprocessor:
    """
    Loads and preprocesses the master bhavcopy CSV.

    Streamlit-friendly:
      - All output goes through self.log(msg) — override or pass a
        callback to redirect messages to st.write / st.status etc.
      - process() returns the cleaned DataFrame so the UI can use it
        directly without touching the filesystem.
      - save() persists the result back to data/raw/bhavcopy.csv
        (overwrites in place — same file, cleaned).
    """

    def __init__(self, file_path=None, log_callback=None):
        """
        Parameters
        ----------
        file_path : str or Path, optional
            Path to the bhavcopy CSV.
            Defaults to data/raw/bhavcopy.csv relative to project root.
        log_callback : callable, optional
            Function that accepts a single string message.
            Defaults to print(). Pass st.write for Streamlit.
        """
        # Resolve project root (myProjectFile/)
        # File is at: myProjectFile/src/nse/preprocess.py
        # So parents[2] = myProjectFile/
        BASE_DIR = Path(__file__).resolve().parents[2]

        self.default_path = BASE_DIR / "data" / "raw" / "bhavcopy.csv"
        self.file_path    = Path(file_path) if file_path else self.default_path

        # Streamlit-friendly logging
        self._log = log_callback if callable(log_callback) else print

        self.df = None

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def log(self, msg: str):
        """Send a status message to the configured log callback."""
        self._log(msg)

    # ------------------------------------------------------------------
    # Pipeline steps  (each returns self for chaining)
    # ------------------------------------------------------------------

    def load_data(self):
        """Load the CSV into self.df."""
        self.df = pd.read_csv(self.file_path)
        self.log(f"✓ Loaded {self.file_path.name} — {len(self.df):,} rows, {self.df.shape[1]} columns")
        return self

    def convert_date(self):
        """Parse TRADE_DATE to datetime."""
        self.df['TRADE_DATE'] = pd.to_datetime(self.df['TRADE_DATE'])
        self.log("✓ TRADE_DATE converted to datetime")
        return self

    def convert_price_columns(self):
        """Coerce price columns to numeric."""
        price_cols = [
            'PREV_CL_PR', 'OPEN_PRICE', 'HIGH_PRICE',
            'LOW_PRICE',  'CLOSE_PRICE', 'HI_52_WK', 'LO_52_WK'
        ]
        self.df[price_cols] = self.df[price_cols].apply(
            pd.to_numeric, errors='coerce'
        )
        self.log(f"✓ Price columns converted: {price_cols}")
        return self

    def convert_trade_columns(self):
        """Coerce trade quantity / value columns to numeric."""
        trade_cols = ['NET_TRDQTY', 'NET_TRDVAL', 'TRADES']
        self.df[trade_cols] = self.df[trade_cols].apply(
            pd.to_numeric, errors='coerce'
        )
        self.log(f"✓ Trade columns converted: {trade_cols}")
        return self

    def convert_categorical_columns(self):
        """Cast low-cardinality string columns to category dtype."""
        cat_cols = ['MKT', 'SECURITY', 'IND_SEC', 'CORP_IND']
        for col in cat_cols:
            self.df[col] = self.df[col].astype('category')
        self.log(f"✓ Categorical columns set: {cat_cols}")
        return self

    def convert_date_parts(self):
        """Extract year / month / day helper columns from TRADE_DATE."""
        self.df['year']  = self.df['TRADE_DATE'].dt.year
        self.df['month'] = self.df['TRADE_DATE'].dt.month
        self.df['day']   = self.df['TRADE_DATE'].dt.day
        self.log("✓ Date part columns added: year, month, day")
        return self

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, output_path=None) -> Path:
        """
        Persist the cleaned DataFrame to disk.

        Parameters
        ----------
        output_path : str or Path, optional
            Defaults to the same file that was loaded (overwrites in place).

        Returns
        -------
        Path where the file was saved.
        """
        if self.df is None:
            raise RuntimeError("No data to save — run process() first.")

        dest = Path(output_path) if output_path else self.file_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(dest, index=False)
        self.log(f"✅ Preprocessed data saved → {dest}")
        return dest

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process(self) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline and return the cleaned DataFrame.
        Call save() afterwards if you want to persist it.
        """
        self.log("🔄 Starting BhavcopyPreprocessor pipeline...")
        return (
            self.load_data()
                .convert_date()
                .convert_price_columns()
                .convert_trade_columns()
                .convert_categorical_columns()
                .convert_date_parts()
                .df
        )


# ------------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    preprocessor = BhavcopyPreprocessor()
    df = preprocessor.process()
    preprocessor.save()

    print(f"\n✅ Preprocessing complete.")
    print(f"   Rows     : {len(df):,}")
    print(f"   Columns  : {list(df.columns)}")
    print(f"   Date range: {df['TRADE_DATE'].min()} → {df['TRADE_DATE'].max()}")
    
    
    
    
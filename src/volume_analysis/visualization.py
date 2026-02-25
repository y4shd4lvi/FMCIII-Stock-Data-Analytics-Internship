import pandas as pd
import sys
from pathlib import Path


class VolumeResultsVisualizer:
    """
    Console Visualization for Nifty 50 Volume Analysis Results.
    Reads the processed report CSV and renders a formatted summary.
    """

    def __init__(self, csv_file=None):
        """
        Initialize with an optional CSV path.
        Defaults to data/processed/nifty50_volume_analysis_report.csv
        relative to the project root.
        """
        # Resolve project root (myProjectFile/)
        # File is at: myProjectFile/src/volume_analysis/visualization.py
        # So parents[2] = myProjectFile/
        self.BASE_DIR = Path(__file__).resolve().parents[2]

        self.default_csv = (
            self.BASE_DIR / "data" / "processed" / "nifty50_volume_analysis_report.csv"
        )

        self.csv_file = Path(csv_file) if csv_file else self.default_csv
        self.df = None

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_data(self):
        """Load the analysis CSV into a DataFrame."""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✓ Loaded: {self.csv_file}")
        except FileNotFoundError:
            print(f"Error: File '{self.csv_file}' not found.")
            print("Please run volume.py first to generate the report.")
            return False
        return True

    # ------------------------------------------------------------------
    # Section Renderers
    # ------------------------------------------------------------------

    def _print_overview(self):
        """Print portfolio overview and distribution breakdowns."""
        df = self.df
        print("\n" + "="*120)
        print("NIFTY 50 VOLUME ANALYSIS - DETAILED RESULTS".center(120))
        print("="*120)

        print("\n📊 PORTFOLIO OVERVIEW")
        print("-" * 120)
        print(f"Total Stocks Analyzed: {len(df)}")
        print(f"Analysis Date: {df['DATE'].iloc[0] if len(df) > 0 else 'N/A'}")

        print("\n💡 RECOMMENDATION BREAKDOWN:")
        rec_counts = df['RECOMMENDATION'].value_counts()
        for rec, count in rec_counts.items():
            pct = (count / len(df)) * 100
            bar = "█" * int(pct / 2)
            print(f"  {rec:15s}: {count:3d} stocks ({pct:5.1f}%) {bar}")

        print("\n⚠️  RISK DISTRIBUTION:")
        risk_counts = df['RISK_LEVEL'].value_counts()
        for risk, count in risk_counts.items():
            pct = (count / len(df)) * 100
            bar = "█" * int(pct / 2)
            print(f"  {risk:15s}: {count:3d} stocks ({pct:5.1f}%) {bar}")

        print("\n📈 TREND ANALYSIS:")
        trend_counts = df['TREND'].value_counts()
        for trend, count in trend_counts.items():
            pct = (count / len(df)) * 100
            bar = "█" * int(pct / 2)
            print(f"  {trend:20s}: {count:3d} stocks ({pct:5.1f}%) {bar}")

    def _print_strong_buys(self):
        """Print top STRONG BUY opportunities."""
        df = self.df
        print("\n" + "="*120)
        print("🔥 TOP 10 STRONG BUY OPPORTUNITIES".center(120))
        print("="*120)

        strong_buys = df[df['RECOMMENDATION'] == 'STRONG_BUY'].head(10)
        if len(strong_buys) > 0:
            print(f"\n{'Rank':<6}{'Stock':<35}{'Price':<12}{'Change%':<10}{'Risk':<15}{'Profit%':<12}{'Volume':<10}")
            print("-" * 120)

            for idx, (i, row) in enumerate(strong_buys.iterrows(), 1):
                stock_name = row['SECURITY'][:32] + "..." if len(row['SECURITY']) > 32 else row['SECURITY']
                change_icon = "🟢" if row['PRICE_CHANGE_%'] > 0 else "🔴"
                print(f"{idx:<6}{stock_name:<35}₹{row['CLOSE_PRICE']:<10.2f}{change_icon} {row['PRICE_CHANGE_%']:<8.2f}{'  '}{row['RISK_LEVEL']:<13}  {row['PROFIT_PROBABILITY']:<10}  {row['VOL_RATIO_20D']:<8.2f}x")

            print("\n" + "-" * 120)
            print("DETAILS OF TOP 3 OPPORTUNITIES:")
            print("-" * 120)

            for idx, (i, row) in enumerate(strong_buys.head(3).iterrows(), 1):
                print(f"\n#{idx}. {row['SECURITY']}")
                print(f"   Current Price: ₹{row['CLOSE_PRICE']:.2f} | Daily Change: {row['PRICE_CHANGE_%']:+.2f}%")
                print(f"   52-Week Range: ₹{row['52W_LOW']:.2f} - ₹{row['52W_HIGH']:.2f} | Position: {row['POSITION_IN_52W']}")
                print(f"   Risk Assessment: {row['RISK_LEVEL']} ({row['RISK_SCORE']}) | Profit Probability: {row['PROFIT_PROBABILITY']}")
                print(f"   Market Trend: {row['TREND']} | Accumulation: {row['ACCUMULATION']}")
                print(f"   Volume Analysis: {row['VOL_RATIO_20D']:.2f}x average | Spike: {row['VOL_SPIKE']} | Trend: {row['VOL_TREND']}")
                print(f"   Technical Signals: OBV {row['OBV_TREND']} | A/D Line {row['AD_TREND']} | Breakout: {row['BREAKOUT_STATUS']}")
        else:
            print("\n   No STRONG_BUY recommendations at this time.")

    def _print_buys(self):
        """Print BUY recommendations."""
        df = self.df
        print("\n" + "="*120)
        print("✅ BUY RECOMMENDATIONS (Accumulation Candidates)".center(120))
        print("="*120)

        buys = df[df['RECOMMENDATION'] == 'BUY'].head(10)
        if len(buys) > 0:
            print(f"\n{'Stock':<35}{'Price':<12}{'Change%':<10}{'Risk':<15}{'Profit%':<12}{'Accumulation':<25}")
            print("-" * 120)
            for i, row in buys.iterrows():
                stock_name = row['SECURITY'][:32] + "..." if len(row['SECURITY']) > 32 else row['SECURITY']
                change_icon = "🟢" if row['PRICE_CHANGE_%'] > 0 else "🔴"
                print(f"{stock_name:<35}₹{row['CLOSE_PRICE']:<10.2f}{change_icon} {row['PRICE_CHANGE_%']:<8.2f}  {row['RISK_LEVEL']:<13}  {row['PROFIT_PROBABILITY']:<10}  {row['ACCUMULATION']:<25}")
        else:
            print("\n   No BUY recommendations at this time.")

    def _print_holds(self):
        """Print HOLD positions."""
        df = self.df
        print("\n" + "="*120)
        print("⏸️  HOLD POSITIONS (Wait & Watch)".center(120))
        print("="*120)

        holds = df[df['RECOMMENDATION'] == 'HOLD'].head(10)
        if len(holds) > 0:
            print(f"\n{'Stock':<35}{'Price':<12}{'Change%':<10}{'Risk':<15}{'Trend':<20}")
            print("-" * 120)
            for i, row in holds.iterrows():
                stock_name = row['SECURITY'][:32] + "..." if len(row['SECURITY']) > 32 else row['SECURITY']
                change_icon = "🟢" if row['PRICE_CHANGE_%'] > 0 else "🔴"
                print(f"{stock_name:<35}₹{row['CLOSE_PRICE']:<10.2f}{change_icon} {row['PRICE_CHANGE_%']:<8.2f}  {row['RISK_LEVEL']:<13}  {row['TREND']:<20}")
        else:
            print("\n   No HOLD recommendations at this time.")

    def _print_sells(self):
        """Print SELL recommendations."""
        df = self.df
        print("\n" + "="*120)
        print("⚠️  SELL RECOMMENDATIONS (Exit Signals)".center(120))
        print("="*120)

        sells = df[df['RECOMMENDATION'] == 'SELL'].head(10)
        if len(sells) > 0:
            print(f"\n{'Stock':<35}{'Price':<12}{'Change%':<10}{'Risk':<15}{'Profit%':<12}{'Distribution':<25}")
            print("-" * 120)
            for i, row in sells.iterrows():
                stock_name = row['SECURITY'][:32] + "..." if len(row['SECURITY']) > 32 else row['SECURITY']
                change_icon = "🟢" if row['PRICE_CHANGE_%'] > 0 else "🔴"
                print(f"{stock_name:<35}₹{row['CLOSE_PRICE']:<10.2f}{change_icon} {row['PRICE_CHANGE_%']:<8.2f}  {row['RISK_LEVEL']:<13}  {row['PROFIT_PROBABILITY']:<10}  {row['ACCUMULATION']:<25}")
        else:
            print("\n   No SELL recommendations at this time.")

    def _print_strong_sells(self):
        """Print STRONG SELL - high risk stocks."""
        df = self.df
        print("\n" + "="*120)
        print("🚨 STRONG SELL - HIGH RISK STOCKS TO AVOID".center(120))
        print("="*120)

        strong_sells = df[df['RECOMMENDATION'] == 'STRONG_SELL'].head(10)
        if len(strong_sells) > 0:
            print(f"\n{'Stock':<35}{'Price':<12}{'Change%':<10}{'Risk':<15}{'Trend':<20}{'Breakout':<20}")
            print("-" * 120)
            for i, row in strong_sells.iterrows():
                stock_name = row['SECURITY'][:32] + "..." if len(row['SECURITY']) > 32 else row['SECURITY']
                change_icon = "🟢" if row['PRICE_CHANGE_%'] > 0 else "🔴"
                print(f"{stock_name:<35}₹{row['CLOSE_PRICE']:<10.2f}{change_icon} {row['PRICE_CHANGE_%']:<8.2f}  {row['RISK_LEVEL']:<13}  {row['TREND']:<18}  {row['BREAKOUT_STATUS']:<20}")
        else:
            print("\n   No STRONG_SELL recommendations at this time.")

    def _print_volume_leaders(self):
        """Print top volume activity stocks."""
        df = self.df
        print("\n" + "="*120)
        print("📊 TOP VOLUME ACTIVITY (High Trading Interest)".center(120))
        print("="*120)

        top_volume = df.nlargest(10, 'VOL_RATIO_20D')
        print(f"\n{'Stock':<35}{'Volume Ratio':<15}{'Spike':<10}{'Trend':<15}{'Recommendation':<20}")
        print("-" * 120)

        for i, row in top_volume.iterrows():
            stock_name = row['SECURITY'][:32] + "..." if len(row['SECURITY']) > 32 else row['SECURITY']
            spike_icon = "🔥" if row['VOL_SPIKE'] == 'YES' else "  "
            print(f"{stock_name:<35}{row['VOL_RATIO_20D']:<13.2f}x {spike_icon} {row['VOL_SPIKE']:<8}  {row['VOL_TREND']:<13}  {row['RECOMMENDATION']:<20}")

    def _print_high_profit_probability(self):
        """Print highest profit probability stocks."""
        df = self.df
        print("\n" + "="*120)
        print("🎯 HIGHEST PROFIT PROBABILITY STOCKS".center(120))
        print("="*120)

        df['PROFIT_PCT'] = df['PROFIT_PROBABILITY'].str.rstrip('%').astype(float)
        high_prob = df.nlargest(10, 'PROFIT_PCT')

        print(f"\n{'Stock':<35}{'Probability':<15}{'Risk':<15}{'Trend':<20}{'Recommendation':<20}")
        print("-" * 120)

        for i, row in high_prob.iterrows():
            stock_name = row['SECURITY'][:32] + "..." if len(row['SECURITY']) > 32 else row['SECURITY']
            print(f"{stock_name:<35}{row['PROFIT_PROBABILITY']:<15}{row['RISK_LEVEL']:<15}{row['TREND']:<20}{row['RECOMMENDATION']:<20}")

    def _print_market_insights(self):
        """Print key market insights summary."""
        df = self.df
        print("\n" + "="*120)
        print("💡 KEY MARKET INSIGHTS".center(120))
        print("="*120)

        bullish_stocks      = len(df[df['TREND'].str.contains('BULLISH', na=False)])
        bearish_stocks      = len(df[df['TREND'].str.contains('BEARISH', na=False)])
        accumulation_stocks = len(df[df['ACCUMULATION'].str.contains('ACCUMULATION', na=False)])
        distribution_stocks = len(df[df['ACCUMULATION'].str.contains('DISTRIBUTION', na=False)])

        print(f"\n1. Market Sentiment:")
        print(f"   • Bullish Stocks: {bullish_stocks} ({bullish_stocks/len(df)*100:.1f}%)")
        print(f"   • Bearish Stocks: {bearish_stocks} ({bearish_stocks/len(df)*100:.1f}%)")

        print(f"\n2. Smart Money Flow:")
        print(f"   • Accumulation Phase: {accumulation_stocks} stocks ({accumulation_stocks/len(df)*100:.1f}%)")
        print(f"   • Distribution Phase: {distribution_stocks} stocks ({distribution_stocks/len(df)*100:.1f}%)")

        print(f"\n3. Risk Profile:")
        high_risk = len(df[df['RISK_LEVEL'] == 'HIGH'])
        med_risk  = len(df[df['RISK_LEVEL'] == 'MEDIUM'])
        low_risk  = len(df[df['RISK_LEVEL'] == 'LOW'])
        print(f"   • High Risk: {high_risk} stocks (avoid or small positions)")
        print(f"   • Medium Risk: {med_risk} stocks (normal position sizing)")
        print(f"   • Low Risk: {low_risk} stocks (can increase allocation)")

        print(f"\n4. Trading Opportunities:")
        strong_buys_count = len(df[df['RECOMMENDATION'] == 'STRONG_BUY'])
        buys_count        = len(df[df['RECOMMENDATION'] == 'BUY'])
        print(f"   • Immediate Action: {strong_buys_count} STRONG_BUY stocks")
        print(f"   • Accumulation Candidates: {buys_count} BUY stocks")
        print(f"   • Total Opportunities: {strong_buys_count + buys_count} stocks for portfolio building")

        print("\n" + "="*120)
        print("\n✅ Analysis complete! Use this data to make informed trading decisions.")
        print("⚠️  Remember: Always combine technical analysis with fundamental research.")
        print(f"📄 Full report saved in: {self.csv_file}")
        print("\n" + "="*120 + "\n")

    # ------------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------------

    def visualize(self):
        """Run the full visualization pipeline."""
        if not self.load_data():
            return

        self._print_overview()
        self._print_strong_buys()
        self._print_buys()
        self._print_holds()
        self._print_sells()
        self._print_strong_sells()
        self._print_volume_leaders()
        self._print_high_profit_probability()
        self._print_market_insights()


# ------------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Accept an optional CSV path as a command-line argument;
    # otherwise fall back to the project-relative default.
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None

    visualizer = VolumeResultsVisualizer(csv_file)
    visualizer.visualize()
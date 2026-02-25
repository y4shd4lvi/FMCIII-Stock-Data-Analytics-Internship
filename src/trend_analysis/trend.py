import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional charts
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class Nifty50TrendAnalyzer:
    """
    Professional Technical Analysis System for Nifty 50 Stocks
    Implements institutional-grade trend analysis with moving averages
    """
    
    def __init__(self, csv_path):
        """Initialize the analyzer with bhav copy data"""
        self.df = pd.read_csv(csv_path)
        self.nifty50_securities = [
            "ADANI ENTERPRISES LIMITED", "ADANI PORT & SEZ LTD", "APOLLO HOSPITALS ENTER. L",
            "ASIAN PAINTS LIMITED", "AXIS BANK LIMITED", "BAJAJ AUTO LIMITED",
            "BAJAJ FINSERV LTD.", "BAJAJ FINANCE LIMITED", "BHARAT ELECTRONICS LTD",
            "BHARTI AIRTEL LIMITED", "CIPLA LTD", "COAL INDIA LTD",
            "DR. REDDY S LABORATORIES", "EICHER MOTORS LTD", "ETERNAL LIMITED",
            "GRASIM INDUSTRIES LTD", "HCL TECHNOLOGIES LTD", "HDFC BANK LTD",
            "HDFC LIFE INS CO LTD", "HINDALCO INDUSTRIES LTD", "HINDUSTAN UNILEVER LTD.",
            "ICICI BANK LTD.", "INTERGLOBE AVIATION LTD", "INFOSYS LIMITED", "ITC LTD",
            "JIO FIN SERVICES LTD", "JSW STEEL LIMITED", "KOTAK MAHINDRA BANK LTD",
            "LARSEN & TOUBRO LTD.", "MAHINDRA & MAHINDRA LTD", "MARUTI SUZUKI INDIA LTD.",
            "MAX HEALTHCARE INS LTD", "NESTLE INDIA LIMITED", "NTPC LTD",
            "OIL AND NATURAL GAS CORP.", "POWER GRID CORP. LTD.",
            "RELIANCE INDUSTRIES LTD", "SBI LIFE INSURANCE CO LTD", "STATE BANK OF INDIA",
            "SHRIRAM FINANCE LIMITED", "SUN PHARMACEUTICAL IND L",
            "TATA CONSUMER PRODUCT LTD", "TATA STEEL LIMITED",
            "TATA CONSULTANCY SERV LT", "TECH MAHINDRA LIMITED",
            "TITAN COMPANY LIMITED", "TATA MOTORS PASS VEH LTD", "TRENT LTD",
            "ULTRATECH CEMENT LIMITED", "WIPRO LTD"
        ]
        self.prepare_data()
        
    def prepare_data(self):
        # Convert date
        self.df['TRADE_DATE'] = pd.to_datetime(self.df['TRADE_DATE'], errors='coerce')

        # 🔥 FORCE NUMERIC COLUMNS (THIS FIXES YOUR ERROR)
        price_cols = ['CLOSE_PRICE', 'HIGH_PRICE', 'LOW_PRICE']
        for col in price_cols:
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.replace(',', '')      # remove thousand separators
                .str.strip()
            )
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Volume
        self.df['NET_TRDQTY'] = pd.to_numeric(self.df['NET_TRDQTY'], errors='coerce')

        # Drop rows where prices failed conversion
        self.df = self.df.dropna(subset=['CLOSE_PRICE', 'HIGH_PRICE', 'LOW_PRICE'])

        # Filter Nifty 50
        self.df = self.df[self.df['SECURITY'].isin(self.nifty50_securities)].copy()

        # Sort
        self.df = self.df.sort_values(['SECURITY', 'TRADE_DATE']).reset_index(drop=True)

        print(f"✓ Data loaded: {len(self.df)} records")
        print(f"✓ Date range: {self.df['TRADE_DATE'].min()} to {self.df['TRADE_DATE'].max()}")
        print(f"✓ Nifty 50 stocks found: {self.df['SECURITY'].nunique()}")
    
    def calculate_moving_averages(self, stock_data):
        """Calculate 50-day and 200-day moving averages"""
        stock_data = stock_data.copy()
        stock_data['MA_50'] = stock_data['CLOSE_PRICE'].rolling(window=50, min_periods=1).mean()
        stock_data['MA_200'] = stock_data['CLOSE_PRICE'].rolling(window=200, min_periods=1).mean()
        return stock_data
    
    def identify_swing_points(self, stock_data, window=5):
        """Identify swing highs and swing lows"""
        stock_data = stock_data.copy()
        
        # Swing Highs: Local maxima
        stock_data['swing_high'] = stock_data['HIGH_PRICE'].rolling(window=window*2+1, center=True).apply(
            lambda x: x[window] == x.max() if len(x) == window*2+1 else False, raw=True
        )
        
        # Swing Lows: Local minima
        stock_data['swing_low'] = stock_data['LOW_PRICE'].rolling(window=window*2+1, center=True).apply(
            lambda x: x[window] == x.min() if len(x) == window*2+1 else False, raw=True
        )
        
        return stock_data
    
    def analyze_market_structure(self, stock_data):
        """Analyze market structure - HH/HL or LH/LL pattern"""
        stock_data = self.identify_swing_points(stock_data)
        
        # Get recent swing points (last 10)
        swing_highs = stock_data[stock_data['swing_high'] == 1]['HIGH_PRICE'].tail(10)
        swing_lows = stock_data[stock_data['swing_low'] == 1]['LOW_PRICE'].tail(10)
        
        structure = "Sideways / Range-bound"
        structure_detail = ""
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Check for Higher Highs and Higher Lows (Uptrend)
            hh = swing_highs.iloc[-1] > swing_highs.iloc[-2] if len(swing_highs) >= 2 else False
            hl = swing_lows.iloc[-1] > swing_lows.iloc[-2] if len(swing_lows) >= 2 else False
            
            # Check for Lower Highs and Lower Lows (Downtrend)
            lh = swing_highs.iloc[-1] < swing_highs.iloc[-2] if len(swing_highs) >= 2 else False
            ll = swing_lows.iloc[-1] < swing_lows.iloc[-2] if len(swing_lows) >= 2 else False
            
            if hh and hl:
                structure = "Uptrend (HH/HL)"
                structure_detail = f"Higher High: {swing_highs.iloc[-1]:.2f}, Higher Low: {swing_lows.iloc[-1]:.2f}"
            elif lh and ll:
                structure = "Downtrend (LH/LL)"
                structure_detail = f"Lower High: {swing_highs.iloc[-1]:.2f}, Lower Low: {swing_lows.iloc[-1]:.2f}"
            else:
                structure = "Sideways / Range-bound"
                structure_detail = f"Range: {swing_lows.min():.2f} - {swing_highs.max():.2f}"
        
        return structure, structure_detail, stock_data
    
    def analyze_ma_200(self, stock_data):
        """Analyze 200-day MA position and slope"""
        current_price = stock_data['CLOSE_PRICE'].iloc[-1]
        ma_200 = stock_data['MA_200'].iloc[-1]
        
        # Position relative to MA
        position = "Above" if current_price > ma_200 else "Below"
        distance_pct = ((current_price - ma_200) / ma_200) * 100
        
        # MA Slope (comparing last value to 20 days ago)
        if len(stock_data) >= 20:
            ma_200_old = stock_data['MA_200'].iloc[-20]
            slope = "Upward" if ma_200 > ma_200_old else "Downward" if ma_200 < ma_200_old else "Flat"
        else:
            slope = "Insufficient data"
        
        # Institutional positioning
        if position == "Above" and slope == "Upward":
            positioning = "Bullish - Strong institutional support"
        elif position == "Above" and slope == "Downward":
            positioning = "Neutral - Price above declining MA (caution)"
        elif position == "Below" and slope == "Upward":
            positioning = "Neutral - Price below rising MA (potential recovery)"
        else:
            positioning = "Bearish - Weak institutional support"
        
        return {
            'position': position,
            'distance_pct': distance_pct,
            'slope': slope,
            'positioning': positioning,
            'ma_200_value': ma_200
        }
    
    def detect_golden_death_cross(self, stock_data):
        """Detect Golden Cross and Death Cross events"""
        stock_data = stock_data.copy()
        
        # Create crossover signals
        stock_data['MA_50_above_MA_200'] = stock_data['MA_50'] > stock_data['MA_200']
        stock_data['cross_signal'] = stock_data['MA_50_above_MA_200'].diff()
        
        # Find most recent Golden Cross (50 MA crosses above 200 MA)
        golden_crosses = stock_data[stock_data['cross_signal'] == 1]
        golden_cross = None
        if len(golden_crosses) > 0:
            last_gc = golden_crosses.iloc[-1]
            golden_cross = {
                'date': last_gc['TRADE_DATE'],
                'days_ago': (stock_data['TRADE_DATE'].iloc[-1] - last_gc['TRADE_DATE']).days,
                'price_at_cross': last_gc['CLOSE_PRICE'],
                'current_price': stock_data['CLOSE_PRICE'].iloc[-1],
                'confirmed': stock_data['CLOSE_PRICE'].iloc[-1] > last_gc['CLOSE_PRICE']
            }
        
        # Find most recent Death Cross (50 MA crosses below 200 MA)
        death_crosses = stock_data[stock_data['cross_signal'] == -1]
        death_cross = None
        if len(death_crosses) > 0:
            last_dc = death_crosses.iloc[-1]
            death_cross = {
                'date': last_dc['TRADE_DATE'],
                'days_ago': (stock_data['TRADE_DATE'].iloc[-1] - last_dc['TRADE_DATE']).days,
                'price_at_cross': last_dc['CLOSE_PRICE'],
                'current_price': stock_data['CLOSE_PRICE'].iloc[-1],
                'confirmed': stock_data['CLOSE_PRICE'].iloc[-1] < last_dc['CLOSE_PRICE']
            }
        
        return golden_cross, death_cross
    
    def classify_trend_strength(self, stock_data, structure, ma_200_analysis, golden_cross, death_cross):
        """Classify overall trend strength"""
        current_price = stock_data['CLOSE_PRICE'].iloc[-1]
        ma_50 = stock_data['MA_50'].iloc[-1]
        ma_200 = stock_data['MA_200'].iloc[-1]
        
        score = 0
        
        # Market Structure (3 points)
        if "Uptrend (HH/HL)" in structure:
            score += 3
        elif "Downtrend (LH/LL)" in structure:
            score -= 3
        
        # MA 200 Position (2 points)
        if ma_200_analysis['position'] == "Above":
            score += 2
        else:
            score -= 2
        
        # MA 200 Slope (2 points)
        if ma_200_analysis['slope'] == "Upward":
            score += 2
        elif ma_200_analysis['slope'] == "Downward":
            score -= 2
        
        # Recent Crossover (3 points)
        if golden_cross and golden_cross['days_ago'] < 60:
            score += 3
        if death_cross and death_cross['days_ago'] < 60:
            score -= 3
        
        # Price relative to MAs (2 points)
        if current_price > ma_50 > ma_200:
            score += 2
        elif current_price < ma_50 < ma_200:
            score -= 2
        
        # Final Classification
        if score >= 7:
            classification = "🟢 STRONG UPTREND"
        elif score >= 3:
            classification = "🟡 WEAK/DEVELOPING UPTREND"
        elif score >= -2:
            classification = "⚪ SIDEWAYS / RANGE-BOUND"
        elif score >= -6:
            classification = "🟠 WEAK DOWNTREND"
        else:
            classification = "🔴 STRONG DOWNTREND"
        
        return classification, score
    
    def analyze_stock(self, security_name):
        """Complete analysis for a single stock"""
        stock_data = self.df[self.df['SECURITY'] == security_name].copy()
        
        if len(stock_data) < 50:
            return None, f"Insufficient data for {security_name}"
        
        # Calculate moving averages
        stock_data = self.calculate_moving_averages(stock_data)
        
        # 1. Market Structure Analysis
        structure, structure_detail, stock_data = self.analyze_market_structure(stock_data)
        
        # 2. 200-Day MA Analysis
        ma_200_analysis = self.analyze_ma_200(stock_data)
        
        # 3. Golden/Death Cross Detection
        golden_cross, death_cross = self.detect_golden_death_cross(stock_data)
        
        # 4. Trend Strength Classification
        classification, score = self.classify_trend_strength(
            stock_data, structure, ma_200_analysis, golden_cross, death_cross
        )
        
        analysis_report = {
            'security': security_name,
            'current_price': stock_data['CLOSE_PRICE'].iloc[-1],
            'structure': structure,
            'structure_detail': structure_detail,
            'ma_200_analysis': ma_200_analysis,
            'golden_cross': golden_cross,
            'death_cross': death_cross,
            'classification': classification,
            'score': score,
            'data': stock_data
        }
        
        return analysis_report, None
    
    def plot_stock_analysis(self, analysis_report, save_path=None):
        """Create professional chart for stock analysis"""
        stock_data = analysis_report['data']
        security = analysis_report['security']
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot 1: Price and Moving Averages
        ax1.plot(stock_data['TRADE_DATE'], stock_data['CLOSE_PRICE'], 
                label='Close Price', color='black', linewidth=2, alpha=0.8)
        ax1.plot(stock_data['TRADE_DATE'], stock_data['MA_50'], 
                label='50-Day MA', color='blue', linewidth=1.5, linestyle='--')
        ax1.plot(stock_data['TRADE_DATE'], stock_data['MA_200'], 
                label='200-Day MA', color='red', linewidth=2, linestyle='--')
        
        # Mark swing highs and lows
        swing_highs = stock_data[stock_data['swing_high'] == 1]
        swing_lows = stock_data[stock_data['swing_low'] == 1]
        
        ax1.scatter(swing_highs['TRADE_DATE'], swing_highs['HIGH_PRICE'], 
                   color='green', marker='^', s=100, label='Swing High', zorder=5)
        ax1.scatter(swing_lows['TRADE_DATE'], swing_lows['LOW_PRICE'], 
                   color='red', marker='v', s=100, label='Swing Low', zorder=5)
        
        # Mark Golden/Death Cross
        if analysis_report['golden_cross']:
            gc_date = analysis_report['golden_cross']['date']
            gc_price = analysis_report['golden_cross']['price_at_cross']
            ax1.axvline(gc_date, color='gold', linestyle=':', linewidth=2, alpha=0.7)
            ax1.annotate('Golden Cross', xy=(gc_date, gc_price), 
                        xytext=(10, 20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='gold', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if analysis_report['death_cross']:
            dc_date = analysis_report['death_cross']['date']
            dc_price = analysis_report['death_cross']['price_at_cross']
            ax1.axvline(dc_date, color='purple', linestyle=':', linewidth=2, alpha=0.7)
            ax1.annotate('Death Cross', xy=(dc_date, dc_price), 
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='purple', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax1.set_title(f'{security} - Trend Analysis\n{analysis_report["classification"]} (Score: {analysis_report["score"]})', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (₹)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Volume
        ax2.bar(stock_data['TRADE_DATE'], stock_data['NET_TRDQTY'], 
               color='steelblue', alpha=0.6, label='Volume')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Chart saved: {save_path}")
        
        return fig
    
    def generate_text_report(self, analysis_report):
        """Generate detailed text report for a stock"""
        report = []
        report.append("=" * 80)
        report.append(f"INSTITUTIONAL TREND ANALYSIS REPORT")
        report.append(f"Security: {analysis_report['security']}")
        report.append(f"Current Price: ₹{analysis_report['current_price']:.2f}")
        report.append("=" * 80)
        report.append("")
        
        # 1. Market Structure
        report.append("1. MARKET STRUCTURE (Price Action)")
        report.append("-" * 80)
        report.append(f"   Pattern: {analysis_report['structure']}")
        report.append(f"   Detail: {analysis_report['structure_detail']}")
        report.append("")
        
        # 2. 200-Day MA Analysis
        report.append("2. 200-DAY MOVING AVERAGE (Long-Term Trend)")
        report.append("-" * 80)
        ma_200 = analysis_report['ma_200_analysis']
        report.append(f"   Current Price vs MA(200): {ma_200['position']} by {abs(ma_200['distance_pct']):.2f}%")
        report.append(f"   MA(200) Value: ₹{ma_200['ma_200_value']:.2f}")
        report.append(f"   MA(200) Slope: {ma_200['slope']}")
        report.append(f"   Institutional Positioning: {ma_200['positioning']}")
        report.append("")
        
        # 3. Golden Cross / Death Cross
        report.append("3. GOLDEN CROSS / DEATH CROSS")
        report.append("-" * 80)
        
        if analysis_report['golden_cross']:
            gc = analysis_report['golden_cross']
            report.append(f"   ✓ GOLDEN CROSS Detected")
            report.append(f"     Date: {gc['date'].strftime('%Y-%m-%d')} ({gc['days_ago']} days ago)")
            report.append(f"     Price at Cross: ₹{gc['price_at_cross']:.2f}")
            report.append(f"     Current Price: ₹{gc['current_price']:.2f}")
            report.append(f"     Signal Confirmed: {'YES' if gc['confirmed'] else 'NO (price below cross level)'}")
        else:
            report.append("   ✗ No Recent Golden Cross")
        
        report.append("")
        
        if analysis_report['death_cross']:
            dc = analysis_report['death_cross']
            report.append(f"   ✓ DEATH CROSS Detected")
            report.append(f"     Date: {dc['date'].strftime('%Y-%m-%d')} ({dc['days_ago']} days ago)")
            report.append(f"     Price at Cross: ₹{dc['price_at_cross']:.2f}")
            report.append(f"     Current Price: ₹{dc['current_price']:.2f}")
            report.append(f"     Signal Confirmed: {'YES' if dc['confirmed'] else 'NO (price above cross level)'}")
        else:
            report.append("   ✗ No Recent Death Cross")
        
        report.append("")
        
        # 4. Trend Strength Conclusion
        report.append("4. TREND STRENGTH CONCLUSION")
        report.append("-" * 80)
        report.append(f"   Classification: {analysis_report['classification']}")
        report.append(f"   Composite Score: {analysis_report['score']}/12")
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def analyze_all_nifty50(self):
        """Analyze all Nifty 50 stocks and generate reports"""
        # Resolve project root (myProjectFile/)
        # File is at: myProjectFile/src/trend_analysis/trend.py
        # So parents[2] = myProjectFile/
        BASE_DIR = Path(__file__).resolve().parents[2]

        # Output directories
        reports_dir = BASE_DIR / "data" / "reports" / "trend" / "reports"
        charts_dir  = BASE_DIR / "data" / "reports" / "trend" / "charts"

        reports_dir.mkdir(parents=True, exist_ok=True)
        charts_dir.mkdir(parents=True, exist_ok=True)

        results = []
        failed = []
        
        print(f"\n{'='*80}")
        print(f"ANALYZING NIFTY 50 STOCKS")
        print(f"{'='*80}\n")
        
        for i, security in enumerate(self.nifty50_securities, 1):
            print(f"[{i}/{len(self.nifty50_securities)}] Analyzing {security}...")
            
            analysis_report, error = self.analyze_stock(security)
            
            if error:
                print(f"    ✗ {error}")
                failed.append(security)
                continue
            
            # Generate text report
            text_report = self.generate_text_report(analysis_report)
            
            # Safe filename stem
            safe_name = security.replace(' ', '_').replace('.', '').replace('&', 'AND')

            # Save text report → data/reports/trend/reports/
            report_file = reports_dir / f"{safe_name}_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(text_report)
            
            # Generate and save chart → data/reports/trend/charts/
            chart_file = charts_dir / f"{safe_name}_chart.png"
            fig = self.plot_stock_analysis(analysis_report, save_path=chart_file)
            plt.close(fig)
            
            print(f"    ✓ {analysis_report['classification']}")
            
            results.append({
                'Security': security,
                'Classification': analysis_report['classification'],
                'Score': analysis_report['score'],
                'Current_Price': analysis_report['current_price'],
                'Structure': analysis_report['structure'],
                'MA200_Position': analysis_report['ma_200_analysis']['position'],
                'MA200_Slope': analysis_report['ma_200_analysis']['slope']
            })
        
        # Create summary report
        summary_df = pd.DataFrame(results)
        summary_df = summary_df.sort_values('Score', ascending=False)
        
        # Summary CSV → data/reports/trend/
        summary_dir = BASE_DIR / "data" / "reports" / "trend"
        summary_file = summary_dir / "NIFTY50_SUMMARY.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Summary text report → data/reports/trend/
        summary_text = []
        summary_text.append("=" * 80)
        summary_text.append("NIFTY 50 TREND ANALYSIS SUMMARY")
        summary_text.append("=" * 80)
        summary_text.append(f"\nTotal Stocks Analyzed: {len(results)}")
        summary_text.append(f"Failed: {len(failed)}")
        summary_text.append("\n" + "=" * 80)
        summary_text.append("TREND DISTRIBUTION")
        summary_text.append("=" * 80)
        
        for classification in summary_df['Classification'].unique():
            count = len(summary_df[summary_df['Classification'] == classification])
            summary_text.append(f"{classification}: {count} stocks")
        
        summary_text.append("\n" + "=" * 80)
        summary_text.append("TOP 10 STRONGEST UPTRENDS")
        summary_text.append("=" * 80)
        for idx, row in summary_df.head(10).iterrows():
            summary_text.append(f"{row['Security']}: {row['Classification']} (Score: {row['Score']})")
        
        summary_text.append("\n" + "=" * 80)
        summary_text.append("TOP 10 WEAKEST/DOWNTRENDS")
        summary_text.append("=" * 80)
        for idx, row in summary_df.tail(10).iterrows():
            summary_text.append(f"{row['Security']}: {row['Classification']} (Score: {row['Score']})")
        
        summary_text_file = summary_dir / "NIFTY50_SUMMARY.txt"
        with open(summary_text_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_text))
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Text reports saved to : {reports_dir}")
        print(f"✓ Charts saved to       : {charts_dir}")
        print(f"✓ Summary CSV           : {summary_file}")
        print(f"✓ Summary text          : {summary_text_file}")
        
        return summary_df


# Example usage
if __name__ == "__main__":
    # Resolve project root (myProjectFile/)
    BASE_DIR = Path(__file__).resolve().parents[2]

    input_csv = BASE_DIR / "data" / "raw" / "bhavcopy_master.csv"

    analyzer = Nifty50TrendAnalyzer(input_csv)
    summary  = analyzer.analyze_all_nifty50()

    print("\nAnalysis complete! Check data/reports/trend/ for detailed reports and charts.")
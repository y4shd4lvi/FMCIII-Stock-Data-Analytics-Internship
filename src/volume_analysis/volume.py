import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class Nifty50VolumeAnalyzer:
    """
    Comprehensive Volume Analysis for Nifty 50 Stocks
    Analyzes: Price-Volume Relationship, Volume Spikes, Breakouts/Breakdowns,
    Volume Trends, Accumulation/Distribution, and generates Buy/Sell signals
    """
    
    def __init__(self, data_path):
        """Initialize with dataset path"""
        self.df = pd.read_csv(data_path)
        self.nifty50_stocks = [
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
        """Prepare and clean data"""

        # Filter only Nifty 50 stocks
        self.df = self.df[self.df['SECURITY'].isin(self.nifty50_stocks)].copy()

        # -------------------- FIX 1: CLEAN NUMERIC COLUMNS --------------------
        numeric_cols = [
            'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE',
            'PREV_CL_PR', 'NET_TRDQTY', 'HI_52_WK', 'LO_52_WK'
        ]

        for col in numeric_cols:
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.strip()
            )
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # -------------------- FIX 2: CLEAN DATE COLUMN --------------------
        self.df['TRADE_DATE'] = pd.to_datetime(self.df['TRADE_DATE'], errors='coerce')

        # Drop rows where essential values are missing
        self.df = self.df.dropna(subset=['TRADE_DATE', 'CLOSE_PRICE', 'PREV_CL_PR', 'NET_TRDQTY'])

        # Sort by security and date
        self.df = self.df.sort_values(['SECURITY', 'TRADE_DATE'])

        # -------------------- FIX 3: SAFE PRICE CALCULATIONS --------------------
        self.df['PRICE_CHANGE'] = self.df['CLOSE_PRICE'] - self.df['PREV_CL_PR']
        self.df['PRICE_CHANGE_PCT'] = (self.df['PRICE_CHANGE'] / self.df['PREV_CL_PR']) * 100

        print(f"Data loaded: {len(self.df)} records for {self.df['SECURITY'].nunique()} Nifty 50 stocks")
        print(f"Date range: {self.df['TRADE_DATE'].min()} to {self.df['TRADE_DATE'].max()}")
    
    def calculate_volume_metrics(self, stock_df):
        """Calculate volume-based technical indicators"""
        stock_df = stock_df.copy()
        
        # 1. Volume Moving Averages
        stock_df['VOL_MA_5'] = stock_df['NET_TRDQTY'].rolling(window=5).mean()
        stock_df['VOL_MA_20'] = stock_df['NET_TRDQTY'].rolling(window=20).mean()
        stock_df['VOL_MA_50'] = stock_df['NET_TRDQTY'].rolling(window=50).mean()
        
        # 2. Volume Ratio (Current volume vs average)
        stock_df['VOL_RATIO_5'] = stock_df['NET_TRDQTY'] / stock_df['VOL_MA_5']
        stock_df['VOL_RATIO_20'] = stock_df['NET_TRDQTY'] / stock_df['VOL_MA_20']
        
        # 3. Volume Spike Detection (>2x average)
        stock_df['VOL_SPIKE'] = stock_df['VOL_RATIO_20'] > 2.0
        
        # 4. Price-Volume Correlation
        stock_df['PV_PRODUCT'] = stock_df['PRICE_CHANGE_PCT'] * stock_df['VOL_RATIO_20']
        
        # 5. On-Balance Volume (OBV)
        stock_df['OBV'] = 0
        for i in range(1, len(stock_df)):
            if stock_df.iloc[i]['CLOSE_PRICE'] > stock_df.iloc[i]['PREV_CL_PR']:
                stock_df.iloc[i, stock_df.columns.get_loc('OBV')] = stock_df.iloc[i-1]['OBV'] + stock_df.iloc[i]['NET_TRDQTY']
            elif stock_df.iloc[i]['CLOSE_PRICE'] < stock_df.iloc[i]['PREV_CL_PR']:
                stock_df.iloc[i, stock_df.columns.get_loc('OBV')] = stock_df.iloc[i-1]['OBV'] - stock_df.iloc[i]['NET_TRDQTY']
            else:
                stock_df.iloc[i, stock_df.columns.get_loc('OBV')] = stock_df.iloc[i-1]['OBV']
        
        # 6. OBV Moving Average
        stock_df['OBV_MA_10'] = stock_df['OBV'].rolling(window=10).mean()
        
        # 7. Volume Trend (increasing/decreasing)
        stock_df['VOL_TREND'] = np.where(stock_df['VOL_MA_5'] > stock_df['VOL_MA_20'], 'Increasing', 'Decreasing')
        
        # 8. Accumulation/Distribution Line
        stock_df['MF_MULTIPLIER'] = ((stock_df['CLOSE_PRICE'] - stock_df['LOW_PRICE']) - 
                                      (stock_df['HIGH_PRICE'] - stock_df['CLOSE_PRICE'])) / \
                                     (stock_df['HIGH_PRICE'] - stock_df['LOW_PRICE'])
        stock_df['MF_MULTIPLIER'] = stock_df['MF_MULTIPLIER'].fillna(0)
        stock_df['MF_VOLUME'] = stock_df['MF_MULTIPLIER'] * stock_df['NET_TRDQTY']
        stock_df['AD_LINE'] = stock_df['MF_VOLUME'].cumsum()
        stock_df['AD_MA_10'] = stock_df['AD_LINE'].rolling(window=10).mean()
        
        return stock_df
    
    def detect_breakout_breakdown(self, stock_df):
        """Detect breakout/breakdown with volume confirmation"""
        latest = stock_df.iloc[-1]
        prev_20 = stock_df.tail(20)
        
        # Check if price is near 52-week high/low
        near_52w_high = latest['CLOSE_PRICE'] >= (latest['HI_52_WK'] * 0.98)
        near_52w_low = latest['CLOSE_PRICE'] <= (latest['LO_52_WK'] * 1.02)
        
        # Volume confirmation (must be above average)
        volume_confirmed = latest['VOL_RATIO_20'] > 1.2
        
        # Breakout: Price breaking resistance with volume
        if near_52w_high and volume_confirmed and latest['PRICE_CHANGE_PCT'] > 2:
            return 'BREAKOUT_CONFIRMED'
        elif near_52w_high and latest['PRICE_CHANGE_PCT'] > 1:
            return 'BREAKOUT_FORMING'
        
        # Breakdown: Price breaking support with volume
        elif near_52w_low and volume_confirmed and latest['PRICE_CHANGE_PCT'] < -2:
            return 'BREAKDOWN_CONFIRMED'
        elif near_52w_low and latest['PRICE_CHANGE_PCT'] < -1:
            return 'BREAKDOWN_FORMING'
        
        return 'NEUTRAL'
    
    def assess_accumulation_distribution(self, stock_df):
        """Assess if stock is being accumulated or distributed"""
        latest = stock_df.iloc[-1]
        recent = stock_df.tail(10)
        
        # AD Line trending up = Accumulation
        ad_trend = latest['AD_LINE'] > latest['AD_MA_10']
        
        # OBV trending up = Accumulation
        obv_trend = latest['OBV'] > latest['OBV_MA_10']
        
        # Price up + Volume up = Strong Accumulation
        price_up = recent['PRICE_CHANGE_PCT'].mean() > 0
        volume_up = latest['VOL_MA_5'] > latest['VOL_MA_20']
        
        score = 0
        if ad_trend: score += 1
        if obv_trend: score += 1
        if price_up and volume_up: score += 2
        
        if score >= 3:
            return 'STRONG_ACCUMULATION', score
        elif score == 2:
            return 'ACCUMULATION', score
        elif score == 1:
            return 'NEUTRAL', score
        elif score == 0 and not price_up:
            return 'DISTRIBUTION', score
        else:
            return 'STRONG_DISTRIBUTION', score
    
    def calculate_risk_score(self, stock_df):
        """Calculate risk score (0-100, higher = riskier)"""
        latest = stock_df.iloc[-1]
        recent = stock_df.tail(20)
        
        risk_score = 0
        
        # 1. Volatility (price range)
        volatility = recent['PRICE_CHANGE_PCT'].std()
        if volatility > 3: risk_score += 30
        elif volatility > 2: risk_score += 20
        elif volatility > 1: risk_score += 10
        
        # 2. Volume inconsistency
        vol_std = recent['NET_TRDQTY'].std() / recent['NET_TRDQTY'].mean()
        if vol_std > 1: risk_score += 25
        elif vol_std > 0.7: risk_score += 15
        elif vol_std > 0.5: risk_score += 10
        
        # 3. Distance from 52-week range
        price_position = (latest['CLOSE_PRICE'] - latest['LO_52_WK']) / (latest['HI_52_WK'] - latest['LO_52_WK'])
        if price_position > 0.95 or price_position < 0.05:
            risk_score += 20
        elif price_position > 0.90 or price_position < 0.10:
            risk_score += 15
        
        # 4. Recent trend reversal risk
        if latest['PRICE_CHANGE_PCT'] < -3:
            risk_score += 15
        
        # 5. Low volume (liquidity risk)
        if latest['VOL_RATIO_20'] < 0.5:
            risk_score += 10
        
        return min(risk_score, 100)
    
    def predict_trend(self, stock_df):
        """Predict short-term trend based on volume analysis"""
        latest = stock_df.iloc[-1]
        recent = stock_df.tail(10)
        
        trend_signals = []
        
        # 1. OBV trend
        if latest['OBV'] > latest['OBV_MA_10']:
            trend_signals.append('BULLISH')
        else:
            trend_signals.append('BEARISH')
        
        # 2. AD Line trend
        if latest['AD_LINE'] > latest['AD_MA_10']:
            trend_signals.append('BULLISH')
        else:
            trend_signals.append('BEARISH')
        
        # 3. Volume trend with price
        if latest['VOL_MA_5'] > latest['VOL_MA_20'] and recent['PRICE_CHANGE_PCT'].mean() > 0:
            trend_signals.append('BULLISH')
        elif latest['VOL_MA_5'] > latest['VOL_MA_20'] and recent['PRICE_CHANGE_PCT'].mean() < 0:
            trend_signals.append('BEARISH')
        
        # 4. Recent momentum
        if recent.tail(5)['PRICE_CHANGE_PCT'].mean() > 1:
            trend_signals.append('BULLISH')
        elif recent.tail(5)['PRICE_CHANGE_PCT'].mean() < -1:
            trend_signals.append('BEARISH')
        
        bullish_count = trend_signals.count('BULLISH')
        bearish_count = trend_signals.count('BEARISH')
        
        if bullish_count >= 3:
            return 'STRONG_BULLISH', bullish_count
        elif bullish_count > bearish_count:
            return 'BULLISH', bullish_count
        elif bearish_count >= 3:
            return 'STRONG_BEARISH', bearish_count
        elif bearish_count > bullish_count:
            return 'BEARISH', bearish_count
        else:
            return 'NEUTRAL', 2
    
    def calculate_profit_probability(self, stock_df):
        """Calculate probability of profit (0-100%)"""
        latest = stock_df.iloc[-1]
        
        probability = 50  # Base probability
        
        # Get trend and accumulation
        trend, trend_strength = self.predict_trend(stock_df)
        acc_dist, acc_score = self.assess_accumulation_distribution(stock_df)
        
        # Adjust based on trend
        if trend == 'STRONG_BULLISH':
            probability += 25
        elif trend == 'BULLISH':
            probability += 15
        elif trend == 'BEARISH':
            probability -= 15
        elif trend == 'STRONG_BEARISH':
            probability -= 25
        
        # Adjust based on accumulation/distribution
        if acc_dist == 'STRONG_ACCUMULATION':
            probability += 15
        elif acc_dist == 'ACCUMULATION':
            probability += 10
        elif acc_dist == 'DISTRIBUTION':
            probability -= 10
        elif acc_dist == 'STRONG_DISTRIBUTION':
            probability -= 15
        
        # Volume confirmation
        if latest['VOL_RATIO_20'] > 1.5:
            probability += 10
        elif latest['VOL_RATIO_20'] < 0.7:
            probability -= 10
        
        # Breakout/Breakdown
        breakout = self.detect_breakout_breakdown(stock_df)
        if breakout == 'BREAKOUT_CONFIRMED':
            probability += 15
        elif breakout == 'BREAKDOWN_CONFIRMED':
            probability -= 15
        
        return max(min(probability, 95), 5)  # Cap between 5-95%
    
    def generate_recommendation(self, stock_df):
        """Generate BUY/SELL/HOLD recommendation"""
        latest = stock_df.iloc[-1]
        
        risk_score = self.calculate_risk_score(stock_df)
        trend, _ = self.predict_trend(stock_df)
        acc_dist, _ = self.assess_accumulation_distribution(stock_df)
        profit_prob = self.calculate_profit_probability(stock_df)
        breakout = self.detect_breakout_breakdown(stock_df)
        
        # Decision logic
        buy_signals = 0
        sell_signals = 0
        
        # Trend signals
        if trend in ['STRONG_BULLISH', 'BULLISH']:
            buy_signals += 2 if trend == 'STRONG_BULLISH' else 1
        elif trend in ['STRONG_BEARISH', 'BEARISH']:
            sell_signals += 2 if trend == 'STRONG_BEARISH' else 1
        
        # Accumulation/Distribution signals
        if acc_dist in ['STRONG_ACCUMULATION', 'ACCUMULATION']:
            buy_signals += 2 if acc_dist == 'STRONG_ACCUMULATION' else 1
        elif acc_dist in ['STRONG_DISTRIBUTION', 'DISTRIBUTION']:
            sell_signals += 2 if acc_dist == 'STRONG_DISTRIBUTION' else 1
        
        # Breakout signals
        if breakout == 'BREAKOUT_CONFIRMED':
            buy_signals += 2
        elif breakout == 'BREAKDOWN_CONFIRMED':
            sell_signals += 2
        
        # Profit probability
        if profit_prob >= 70:
            buy_signals += 1
        elif profit_prob <= 30:
            sell_signals += 1
        
        # Risk consideration
        if risk_score > 70:
            sell_signals += 1
        
        # Final recommendation
        if buy_signals >= 4 and risk_score < 60:
            return 'STRONG_BUY'
        elif buy_signals >= 2 and sell_signals == 0:
            return 'BUY'
        elif sell_signals >= 4 or risk_score > 80:
            return 'STRONG_SELL'
        elif sell_signals >= 2 and buy_signals == 0:
            return 'SELL'
        else:
            return 'HOLD'
    
    def analyze_stock(self, stock_name):
        """Complete analysis for a single stock"""
        stock_df = self.df[self.df['SECURITY'] == stock_name].copy()
        
        if len(stock_df) < 50:
            return None  # Insufficient data
        
        # Calculate all metrics
        stock_df = self.calculate_volume_metrics(stock_df)
        
        # Get latest data
        latest = stock_df.iloc[-1]
        
        # Perform analyses
        risk_score = self.calculate_risk_score(stock_df)
        trend, trend_strength = self.predict_trend(stock_df)
        acc_dist, acc_score = self.assess_accumulation_distribution(stock_df)
        profit_prob = self.calculate_profit_probability(stock_df)
        breakout = self.detect_breakout_breakdown(stock_df)
        recommendation = self.generate_recommendation(stock_df)
        
        # Compile results
        result = {
            'SECURITY': stock_name,
            'DATE': latest['TRADE_DATE'].strftime('%Y-%m-%d'),
            'CLOSE_PRICE': latest['CLOSE_PRICE'],
            'PRICE_CHANGE_%': round(latest['PRICE_CHANGE_PCT'], 2),
            
            # Volume Analysis
            'VOLUME': int(latest['NET_TRDQTY']),
            'VOL_RATIO_20D': round(latest['VOL_RATIO_20'], 2),
            'VOL_SPIKE': 'YES' if latest['VOL_SPIKE'] else 'NO',
            'VOL_TREND': latest['VOL_TREND'],
            
            # Technical Indicators
            'OBV_TREND': 'UP' if latest['OBV'] > latest['OBV_MA_10'] else 'DOWN',
            'AD_TREND': 'UP' if latest['AD_LINE'] > latest['AD_MA_10'] else 'DOWN',
            
            # Analysis Results
            'RISK_SCORE': f"{risk_score}/100",
            'RISK_LEVEL': 'HIGH' if risk_score > 70 else 'MEDIUM' if risk_score > 40 else 'LOW',
            'TREND': trend,
            'BREAKOUT_STATUS': breakout,
            'ACCUMULATION': acc_dist,
            'PROFIT_PROBABILITY': f"{profit_prob}%",
            
            # Recommendation
            'RECOMMENDATION': recommendation,
            
            # 52-week position
            '52W_HIGH': latest['HI_52_WK'],
            '52W_LOW': latest['LO_52_WK'],
            'POSITION_IN_52W': f"{round((latest['CLOSE_PRICE'] - latest['LO_52_WK']) / (latest['HI_52_WK'] - latest['LO_52_WK']) * 100, 1)}%"
        }
        
        return result
    
    def analyze_all_stocks(self):
        """Analyze all Nifty 50 stocks"""
        results = []
        
        print("\n" + "="*100)
        print("ANALYZING NIFTY 50 STOCKS - VOLUME-BASED ANALYSIS")
        print("="*100)
        
        for stock in self.nifty50_stocks:
            result = self.analyze_stock(stock)
            if result:
                results.append(result)
                print(f"✓ Analyzed: {stock}")
            else:
                print(f"✗ Skipped: {stock} (insufficient data)")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def generate_report(self, results_df, output_path):
        """Generate detailed analysis report"""
        
        # Sort by recommendation priority
        priority_order = {'STRONG_BUY': 1, 'BUY': 2, 'HOLD': 3, 'SELL': 4, 'STRONG_SELL': 5}
        results_df['PRIORITY'] = results_df['RECOMMENDATION'].map(priority_order)
        results_df = results_df.sort_values('PRIORITY')
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Full analysis saved to: {output_path}")
        
        # Generate summary
        print("\n" + "="*100)
        print("VOLUME ANALYSIS SUMMARY - NIFTY 50")
        print("="*100)
        
        print(f"\nTotal Stocks Analyzed: {len(results_df)}")
        print(f"\nRECOMMENDATION BREAKDOWN:")
        print(results_df['RECOMMENDATION'].value_counts().to_string())
        
        print(f"\n\nRISK LEVEL DISTRIBUTION:")
        print(results_df['RISK_LEVEL'].value_counts().to_string())
        
        print(f"\n\nTREND DISTRIBUTION:")
        print(results_df['TREND'].value_counts().to_string())
        
        # Top picks
        print("\n" + "="*100)
        print("TOP 10 STRONG BUY RECOMMENDATIONS")
        print("="*100)
        strong_buys = results_df[results_df['RECOMMENDATION'] == 'STRONG_BUY'].head(10)
        if len(strong_buys) > 0:
            for idx, row in strong_buys.iterrows():
                print(f"\n{row['SECURITY']}")
                print(f"  Price: ₹{row['CLOSE_PRICE']} | Change: {row['PRICE_CHANGE_%']}%")
                print(f"  Risk: {row['RISK_LEVEL']} ({row['RISK_SCORE']}) | Profit Probability: {row['PROFIT_PROBABILITY']}")
                print(f"  Trend: {row['TREND']} | Accumulation: {row['ACCUMULATION']}")
                print(f"  Volume Ratio: {row['VOL_RATIO_20D']}x | Breakout: {row['BREAKOUT_STATUS']}")
        else:
            print("No STRONG_BUY recommendations found")
        
        # High risk stocks to avoid
        print("\n" + "="*100)
        print("HIGH RISK STOCKS TO AVOID (STRONG SELL)")
        print("="*100)
        strong_sells = results_df[results_df['RECOMMENDATION'] == 'STRONG_SELL'].head(10)
        if len(strong_sells) > 0:
            for idx, row in strong_sells.iterrows():
                print(f"\n{row['SECURITY']}")
                print(f"  Price: ₹{row['CLOSE_PRICE']} | Change: {row['PRICE_CHANGE_%']}%")
                print(f"  Risk: {row['RISK_LEVEL']} ({row['RISK_SCORE']}) | Profit Probability: {row['PROFIT_PROBABILITY']}")
                print(f"  Trend: {row['TREND']} | Distribution: {row['ACCUMULATION']}")
        else:
            print("No STRONG_SELL recommendations found")
        
        return results_df


# Main execution
if __name__ == "__main__":
    import sys

    # Resolve project root (myProjectFile/)
    # File is at: myProjectFile/src/volume_analysis/volume.py
    # So parents[2] = myProjectFile/
    BASE_DIR = Path(__file__).resolve().parents[2]

    input_csv   = BASE_DIR / "data" / "raw"       / "bhavcopy_master.csv"
    output_csv  = BASE_DIR / "data" / "processed" / "nifty50_volume_analysis_report.csv"

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = Nifty50VolumeAnalyzer(input_csv)

    # Analyze all stocks
    results = analyzer.analyze_all_stocks()

    # Generate report
    analyzer.generate_report(results, output_csv)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
    print(f"\nDetailed report saved to: {output_csv}")
    print("\nKey Insights:")
    print("- Use STRONG_BUY stocks for new positions")
    print("- Consider BUY stocks for gradual accumulation")
    print("- HOLD stocks for existing positions")
    print("- Exit SELL positions gradually")
    print("- Avoid or short STRONG_SELL stocks")
    print("\nDisclaimer: This is technical analysis based on volume patterns.")
    print("Always conduct your own research and consider fundamental analysis.")
    print("="*100)
"""
Fundamental Analysis and Visualization Script
==============================================
This script analyzes company data to identify stocks with strong fundamentals
and creates comprehensive visualizations.

Author: Analysis Script
Date: February 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


class FundamentalAnalyzer:
    """Class to perform fundamental analysis on company data"""
    
    def __init__(self, csv_file_path):
        """Initialize with CSV file path"""
        self.df = pd.read_csv(csv_file_path)
        self.strong_companies = None
        print(f"✓ Data loaded successfully: {len(self.df)} companies")
        
    def calculate_fundamental_score(self):
        """Calculate fundamental score based on multiple criteria"""
        print("\n" + "="*80)
        print("CALCULATING FUNDAMENTAL SCORES")
        print("="*80)
        
        # Initialize score
        self.df['Fundamental_Score'] = 0
        
        # Scoring criteria with explanations
        criteria = {
            'P/E < 25': (self.df['P/E'] < 25, "Reasonable valuation"),
            'ROCE > 15%': (self.df['ROCE %'] > 15, "Efficient capital utilization"),
            'Positive Profit Growth': (self.df['Qtr Profit Var %'] > 0, "Growing profitability"),
            'Dividend Yield > 1%': (self.df['Div Yld %'] > 1, "Shareholder returns"),
            'Sales Growth > 10%': (self.df['Qtr Sales Var %'] > 10, "Strong revenue expansion")
        }
        
        # Apply each criterion
        for criterion, (condition, description) in criteria.items():
            self.df['Fundamental_Score'] += condition.astype(int)
            count = condition.sum()
            print(f"  • {criterion:25} | {description:30} | {count:2} companies")
        
        # Filter strong companies (score >= 3)
        self.strong_companies = self.df[self.df['Fundamental_Score'] >= 3].copy()
        self.strong_companies = self.strong_companies.sort_values('Fundamental_Score', ascending=False)
        
        print(f"\n✓ Companies with strong fundamentals (score ≥3): {len(self.strong_companies)}")
        return self.strong_companies
    
    def get_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        if self.strong_companies is None:
            self.calculate_fundamental_score()
        
        df = self.strong_companies
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        stats = {
            'Total Companies': len(df),
            'Average Market Cap (Rs. Cr)': f"{df['Mar Cap Rs.Cr.'].mean():,.2f}",
            'Total Market Cap (Rs. Cr)': f"{df['Mar Cap Rs.Cr.'].sum():,.2f}",
            'Average ROCE (%)': f"{df['ROCE %'].mean():.2f}",
            'Average P/E Ratio': f"{df['P/E'].mean():.2f}",
            'Average Dividend Yield (%)': f"{df['Div Yld %'].mean():.2f}",
            'Avg Quarterly Profit Growth (%)': f"{df['Qtr Profit Var %'].mean():.2f}",
            'Avg Quarterly Sales Growth (%)': f"{df['Qtr Sales Var %'].mean():.2f}",
        }
        
        for key, value in stats.items():
            print(f"  {key:35} : {value}")
        
        return stats
    
    def get_top_performers(self):
        """Identify top performing companies across different metrics"""
        if self.strong_companies is None:
            self.calculate_fundamental_score()
        
        df = self.strong_companies
        
        print("\n" + "="*80)
        print("TOP PERFORMERS")
        print("="*80)
        
        # Best overall score
        print("\n★ EXCELLENT FUNDAMENTALS (Score 5):")
        score_5 = df[df['Fundamental_Score'] == 5]
        if len(score_5) > 0:
            for idx, row in score_5.iterrows():
                print(f"  • {row['Name']:25} | P/E: {row['P/E']:6.2f} | ROCE: {row['ROCE %']:6.2f}% | Div Yield: {row['Div Yld %']:5.2f}%")
        else:
            print("  None")
        
        # Best value picks
        print("\n★ BEST VALUE PICKS (P/E < 25 & ROCE > 30%):")
        best_value = df[(df['P/E'] < 25) & (df['ROCE %'] > 30)]
        for idx, row in best_value.head(5).iterrows():
            print(f"  • {row['Name']:25} | P/E: {row['P/E']:6.2f} | ROCE: {row['ROCE %']:6.2f}%")
        
        # High growth companies
        print("\n★ HIGH GROWTH COMPANIES (Profit > 20% & Sales > 15%):")
        high_growth = df[(df['Qtr Profit Var %'] > 20) & (df['Qtr Sales Var %'] > 15)]
        for idx, row in high_growth.head(5).iterrows():
            print(f"  • {row['Name']:25} | Profit: {row['Qtr Profit Var %']:7.2f}% | Sales: {row['Qtr Sales Var %']:7.2f}%")
        
        # High dividend yielders
        print("\n★ HIGH DIVIDEND YIELDERS (Yield > 3%):")
        high_dividend = df[df['Div Yld %'] > 3].nlargest(5, 'Div Yld %')
        for idx, row in high_dividend.iterrows():
            print(f"  • {row['Name']:25} | Div Yield: {row['Div Yld %']:5.2f}%")
        
        # Highest ROCE
        print("\n★ EXCEPTIONAL ROCE (ROCE > 40%):")
        high_roce = df[df['ROCE %'] > 40].nlargest(5, 'ROCE %')
        for idx, row in high_roce.iterrows():
            print(f"  • {row['Name']:25} | ROCE: {row['ROCE %']:6.2f}%")
    
    def create_visualizations(self, save_path):
        """Create comprehensive visualization charts"""
        if self.strong_companies is None:
            self.calculate_fundamental_score()
        
        df = self.strong_companies
        
        print(f"\n{'='*80}")
        print("CREATING VISUALIZATIONS")
        print("="*80)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 28))
        
        # 1. Fundamental Score Distribution
        print("  • Creating Score Distribution chart...")
        ax1 = plt.subplot(5, 2, 1)
        score_counts = df['Fundamental_Score'].value_counts().sort_index()
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4d96ff']
        bars = ax1.bar(score_counts.index, score_counts.values, color=colors, 
                       edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Fundamental Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Companies', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Fundamental Scores', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(score_counts.index)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. Top 10 by ROCE
        print("  • Creating Top ROCE chart...")
        ax2 = plt.subplot(5, 2, 2)
        top_roce = df.nlargest(10, 'ROCE %')
        bars = ax2.barh(range(len(top_roce)), top_roce['ROCE %'], 
                        color='#4CAF50', edgecolor='black')
        ax2.set_yticks(range(len(top_roce)))
        ax2.set_yticklabels(top_roce['Name'], fontsize=10)
        ax2.set_xlabel('ROCE %', fontsize=12, fontweight='bold')
        ax2.set_title('Top 10 Companies by ROCE', fontsize=14, fontweight='bold', pad=20)
        ax2.invert_yaxis()
        for i, v in enumerate(top_roce['ROCE %']):
            ax2.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        # 3. P/E Ratio Distribution
        print("  • Creating P/E Distribution chart...")
        ax3 = plt.subplot(5, 2, 3)
        ax3.hist(df['P/E'], bins=15, color='#3498db', edgecolor='black', alpha=0.7)
        ax3.axvline(df['P/E'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {df["P/E"].mean():.1f}')
        ax3.axvline(df['P/E'].median(), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {df["P/E"].median():.1f}')
        ax3.set_xlabel('P/E Ratio', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('P/E Ratio Distribution', fontsize=14, fontweight='bold', pad=20)
        ax3.legend()
        
        # 4. Market Cap vs ROCE Scatter
        print("  • Creating Market Cap vs ROCE scatter plot...")
        ax4 = plt.subplot(5, 2, 4)
        scatter = ax4.scatter(df['Mar Cap Rs.Cr.'], df['ROCE %'], 
                             c=df['Fundamental_Score'], s=100, 
                             cmap='RdYlGn', edgecolor='black', linewidth=1, alpha=0.7)
        ax4.set_xlabel('Market Cap (Rs. Cr.)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('ROCE %', fontsize=12, fontweight='bold')
        ax4.set_title('Market Cap vs ROCE (Colored by Score)', fontsize=14, fontweight='bold', pad=20)
        ax4.set_xscale('log')
        plt.colorbar(scatter, ax=ax4, label='Fundamental Score')
        
        # 5. Top 10 Dividend Yielders
        print("  • Creating Dividend Yield chart...")
        ax5 = plt.subplot(5, 2, 5)
        top_div = df.nlargest(10, 'Div Yld %')
        bars = ax5.barh(range(len(top_div)), top_div['Div Yld %'], 
                        color='#FF6B6B', edgecolor='black')
        ax5.set_yticks(range(len(top_div)))
        ax5.set_yticklabels(top_div['Name'], fontsize=10)
        ax5.set_xlabel('Dividend Yield %', fontsize=12, fontweight='bold')
        ax5.set_title('Top 10 Companies by Dividend Yield', fontsize=14, fontweight='bold', pad=20)
        ax5.invert_yaxis()
        for i, v in enumerate(top_div['Div Yld %']):
            ax5.text(v + 0.1, i, f'{v:.2f}%', va='center', fontsize=9, fontweight='bold')
        
        # 6. Quarterly Profit Growth
        print("  • Creating Profit Growth chart...")
        ax6 = plt.subplot(5, 2, 6)
        top_profit = df.nlargest(10, 'Qtr Profit Var %')
        colors_profit = ['#27ae60' if x > 0 else '#e74c3c' for x in top_profit['Qtr Profit Var %']]
        bars = ax6.barh(range(len(top_profit)), top_profit['Qtr Profit Var %'], 
                        color=colors_profit, edgecolor='black')
        ax6.set_yticks(range(len(top_profit)))
        ax6.set_yticklabels(top_profit['Name'], fontsize=10)
        ax6.set_xlabel('Quarterly Profit Growth %', fontsize=12, fontweight='bold')
        ax6.set_title('Top 10 Companies by Profit Growth', fontsize=14, fontweight='bold', pad=20)
        ax6.invert_yaxis()
        for i, v in enumerate(top_profit['Qtr Profit Var %']):
            ax6.text(v + 2, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        # 7. Quarterly Sales Growth
        print("  • Creating Sales Growth chart...")
        ax7 = plt.subplot(5, 2, 7)
        top_sales = df.nlargest(10, 'Qtr Sales Var %')
        bars = ax7.barh(range(len(top_sales)), top_sales['Qtr Sales Var %'], 
                        color='#9b59b6', edgecolor='black')
        ax7.set_yticks(range(len(top_sales)))
        ax7.set_yticklabels(top_sales['Name'], fontsize=10)
        ax7.set_xlabel('Quarterly Sales Growth %', fontsize=12, fontweight='bold')
        ax7.set_title('Top 10 Companies by Sales Growth', fontsize=14, fontweight='bold', pad=20)
        ax7.invert_yaxis()
        for i, v in enumerate(top_sales['Qtr Sales Var %']):
            ax7.text(v + 2, i, f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        # 8. Sector Distribution
        print("  • Creating Sector Distribution chart...")
        ax8 = plt.subplot(5, 2, 8)
        sectors = []
        for name in df['Name']:
            if any(x in name.lower() for x in ['bank', 'finance', 'finserv', 'life', 'insuran']):
                sectors.append('Finance & Banking')
            elif any(x in name.lower() for x in ['tech', 'infosys', 'tcs', 'hcl', 'wipro']):
                sectors.append('IT & Technology')
            elif any(x in name.lower() for x in ['pharma', 'lab', 'cipla', 'reddy', 'hospital']):
                sectors.append('Healthcare')
            elif any(x in name.lower() for x in ['auto', 'motor', 'maruti', 'eicher', 'bajaj']):
                sectors.append('Automobile')
            elif any(x in name.lower() for x in ['steel', 'cement', 'hindalco', 'ultratech']):
                sectors.append('Materials')
            elif any(x in name.lower() for x in ['power', 'ntpc', 'coal', 'ongc']):
                sectors.append('Energy & Power')
            else:
                sectors.append('Others')
        
        df['Sector'] = sectors
        sector_counts = df['Sector'].value_counts()
        colors_sector = plt.cm.Set3(range(len(sector_counts)))
        wedges, texts, autotexts = ax8.pie(sector_counts, labels=sector_counts.index, 
                                             autopct='%1.1f%%', colors=colors_sector, 
                                             startangle=90, textprops={'fontsize': 10})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax8.set_title('Sector Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # 9. P/E vs Price Scatter
        print("  • Creating P/E vs Price scatter plot...")
        ax9 = plt.subplot(5, 2, 9)
        scatter = ax9.scatter(df['P/E'], df['CMP Rs.'], c=df['ROCE %'], 
                             s=df['Mar Cap Rs.Cr.']/5000, cmap='viridis', 
                             edgecolor='black', linewidth=1, alpha=0.6)
        ax9.set_xlabel('P/E Ratio', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Current Market Price (Rs.)', fontsize=12, fontweight='bold')
        ax9.set_title('P/E vs Price (Size: Market Cap, Color: ROCE)', 
                     fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(scatter, ax=ax9, label='ROCE %')
        
        # 10. Profit vs Sales Growth Comparison
        print("  • Creating Growth Comparison chart...")
        ax10 = plt.subplot(5, 2, 10)
        top_overall = df.nlargest(8, 'Fundamental_Score')
        x = np.arange(len(top_overall))
        width = 0.35
        bars1 = ax10.bar(x - width/2, top_overall['Qtr Profit Var %'], width, 
                         label='Profit Growth %', color='#2ecc71', edgecolor='black')
        bars2 = ax10.bar(x + width/2, top_overall['Qtr Sales Var %'], width, 
                         label='Sales Growth %', color='#3498db', edgecolor='black')
        ax10.set_xlabel('Company', fontsize=12, fontweight='bold')
        ax10.set_ylabel('Growth %', fontsize=12, fontweight='bold')
        ax10.set_title('Profit vs Sales Growth (Top 8)', fontsize=14, fontweight='bold', pad=20)
        ax10.set_xticks(x)
        ax10.set_xticklabels(top_overall['Name'], rotation=45, ha='right', fontsize=9)
        ax10.legend()
        ax10.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Charts saved to: {save_path}")
        plt.close()
    
    def export_results(self, csv_path):
        """Export results to CSV"""
        if self.strong_companies is None:
            self.calculate_fundamental_score()
        
        self.strong_companies.to_csv(csv_path, index=False)
        print(f"✓ Results exported to: {csv_path}")
    
    def generate_detailed_report(self):
        """Generate a detailed text report"""
        if self.strong_companies is None:
            self.calculate_fundamental_score()
        
        df = self.strong_companies
        
        print("\n" + "="*80)
        print("DETAILED FUNDAMENTAL ANALYSIS REPORT")
        print("="*80)
        print(f"Report Date: {datetime.now().strftime('%B %d, %Y')}")
        print(f"Total Companies Analyzed: {len(self.df)}")
        print(f"Companies with Strong Fundamentals: {len(df)}")
        print("="*80)
        
        # ROCE Analysis
        print("\n[ROCE ANALYSIS]")
        print(f"  Highest ROCE: {df['ROCE %'].max():.2f}% ({df.loc[df['ROCE %'].idxmax(), 'Name']})")
        print(f"  Lowest ROCE: {df['ROCE %'].min():.2f}% ({df.loc[df['ROCE %'].idxmin(), 'Name']})")
        print(f"  Average ROCE: {df['ROCE %'].mean():.2f}%")
        print(f"  Companies with ROCE > 30%: {len(df[df['ROCE %'] > 30])}")
        print(f"  Companies with ROCE > 20%: {len(df[df['ROCE %'] > 20])}")
        
        # P/E Analysis
        print("\n[P/E RATIO ANALYSIS]")
        print(f"  Lowest P/E: {df['P/E'].min():.2f} ({df.loc[df['P/E'].idxmin(), 'Name']})")
        print(f"  Highest P/E: {df['P/E'].max():.2f} ({df.loc[df['P/E'].idxmax(), 'Name']})")
        print(f"  Average P/E: {df['P/E'].mean():.2f}")
        print(f"  Companies with P/E < 20: {len(df[df['P/E'] < 20])}")
        print(f"  Companies with P/E < 25: {len(df[df['P/E'] < 25])}")
        
        # Growth Analysis
        print("\n[GROWTH ANALYSIS]")
        profit_pos = df[df['Qtr Profit Var %'] > 0]
        sales_pos = df[df['Qtr Sales Var %'] > 0]
        print(f"  Companies with Positive Profit Growth: {len(profit_pos)} ({len(profit_pos)/len(df)*100:.1f}%)")
        print(f"  Companies with Positive Sales Growth: {len(sales_pos)} ({len(sales_pos)/len(df)*100:.1f}%)")
        print(f"  Highest Profit Growth: {df['Qtr Profit Var %'].max():.2f}% ({df.loc[df['Qtr Profit Var %'].idxmax(), 'Name']})")
        print(f"  Highest Sales Growth: {df['Qtr Sales Var %'].max():.2f}% ({df.loc[df['Qtr Sales Var %'].idxmax(), 'Name']})")
        
        # Dividend Analysis
        print("\n[DIVIDEND ANALYSIS]")
        print(f"  Highest Dividend Yield: {df['Div Yld %'].max():.2f}% ({df.loc[df['Div Yld %'].idxmax(), 'Name']})")
        print(f"  Average Dividend Yield: {df['Div Yld %'].mean():.2f}%")
        print(f"  Companies with Div Yield > 2%: {len(df[df['Div Yld %'] > 2])}")
        print(f"  Companies with Div Yield > 3%: {len(df[df['Div Yld %'] > 3])}")
        
        # Market Cap Analysis
        print("\n[MARKET CAP ANALYSIS]")
        large_cap = df[df['Mar Cap Rs.Cr.'] > 100000]
        mid_cap = df[(df['Mar Cap Rs.Cr.'] >= 50000) & (df['Mar Cap Rs.Cr.'] <= 100000)]
        print(f"  Total Market Cap: Rs. {df['Mar Cap Rs.Cr.'].sum():,.2f} Cr.")
        print(f"  Average Market Cap: Rs. {df['Mar Cap Rs.Cr.'].mean():,.2f} Cr.")
        print(f"  Large Cap (>1 Lakh Cr): {len(large_cap)} companies")
        print(f"  Mid Cap (50K-1 Lakh Cr): {len(mid_cap)} companies")
        
        # Score Distribution
        print("\n[FUNDAMENTAL SCORE BREAKDOWN]")
        for score in sorted(df['Fundamental_Score'].unique(), reverse=True):
            count = len(df[df['Fundamental_Score'] == score])
            pct = count / len(df) * 100
            print(f"  Score {score}: {count} companies ({pct:.1f}%)")
        
        print("\n" + "="*80)
        
        # Top 15 Companies Table
        print("\nTOP 15 COMPANIES BY FUNDAMENTAL SCORE:")
        print("-"*80)
        top_15 = df.head(15)[['Name', 'CMP Rs.', 'P/E', 'ROCE %', 'Qtr Profit Var %', 
                              'Qtr Sales Var %', 'Div Yld %', 'Fundamental_Score']]
        print(top_15.to_string(index=False))
        print("="*80)

from pathlib import Path
import traceback

def main():
    """Main execution function"""

    print("\n" + "="*80)
    print(" "*20 + "FUNDAMENTAL ANALYSIS TOOL")
    print("="*80)

    # Resolve project root (myProjectFile/)
    # File is at: myProjectFile/src/fundamental_analysis/fundamental.py
    # So parents[2] = myProjectFile/
    BASE_DIR = Path(__file__).resolve().parents[2]

    # Define standardized paths
    input_csv    = BASE_DIR / "data" / "raw"       / "Top_50_Companies_Data.csv"
    output_csv   = BASE_DIR / "data" / "processed" / "strong_fundamental_companies.csv"
    output_chart = BASE_DIR / "data" / "reports"   / "fundamental" / "fundamental_analysis_charts.png"

    # Ensure output directories exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_chart.parent.mkdir(parents=True, exist_ok=True)

    try:
        analyzer = FundamentalAnalyzer(input_csv)

        # Calculate fundamental scores
        strong_companies = analyzer.calculate_fundamental_score()

        # Get summary statistics
        analyzer.get_summary_statistics()

        # Get top performers
        analyzer.get_top_performers()

        # Generate detailed report
        analyzer.generate_detailed_report()

        # Create visualizations
        analyzer.create_visualizations(output_chart)

        # Export results
        analyzer.export_results(output_csv)

        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated Files:")
        print(f"  1. {output_csv}")
        print(f"  2. {output_chart}")
        print("="*80)

    except FileNotFoundError:
        print(f"\n❌ Error: File '{input_csv}' not found!")
        print("Please ensure the file exists in data/raw/")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
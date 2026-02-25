"""
Advanced Fundamental Analysis - Extended Features
==================================================
This script provides additional analysis tools including:
- Correlation analysis
- Risk-return analysis
- Comparative sector analysis
- Interactive visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from pathlib import Path
import traceback


class AdvancedAnalyzer:
    """Advanced analysis tools for fundamental data"""
    
    def __init__(self, df):
        """Initialize with dataframe"""
        self.df = df
    
    def correlation_analysis(self, save_path):
        """Analyze correlations between financial metrics"""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        # Select numeric columns
        numeric_cols = ['P/E', 'ROCE %', 'Div Yld %', 'Qtr Profit Var %', 
                       'Qtr Sales Var %', 'Mar Cap Rs.Cr.']
        corr_matrix = self.df[numeric_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, 
                   linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Financial Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation heatmap saved: {save_path}")
        plt.close()
        
        # Print significant correlations
        print("\nKey Correlations (|r| > 0.3):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.3:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    direction = "Positive" if corr_value > 0 else "Negative"
                    print(f"  • {col1} ↔ {col2}: {corr_value:.3f} ({direction})")
    
    def risk_return_analysis(self, save_path):
        """Analyze risk-return profile using ROCE as return and P/E as risk proxy"""
        print("\n" + "="*80)
        print("RISK-RETURN ANALYSIS")
        print("="*80)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Risk-Return Scatter (P/E as risk, ROCE as return)
        scatter = ax1.scatter(self.df['P/E'], self.df['ROCE %'], 
                             c=self.df['Fundamental_Score'], s=150,
                             cmap='RdYlGn', edgecolor='black', linewidth=1.5, alpha=0.7)
        
        # Add quadrant lines
        median_pe = self.df['P/E'].median()
        median_roce = self.df['ROCE %'].median()
        ax1.axvline(median_pe, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(median_roce, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Label quadrants
        ax1.text(median_pe * 0.3, median_roce * 1.7, 'Low Risk\nHigh Return', 
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax1.text(median_pe * 1.7, median_roce * 1.7, 'High Risk\nHigh Return', 
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax1.text(median_pe * 0.3, median_roce * 0.3, 'Low Risk\nLow Return', 
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax1.text(median_pe * 1.7, median_roce * 0.3, 'High Risk\nLow Return', 
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        ax1.set_xlabel('P/E Ratio (Risk Proxy)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ROCE % (Return)', fontsize=12, fontweight='bold')
        ax1.set_title('Risk-Return Profile', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(scatter, ax=ax1, label='Fundamental Score')
        
        # Efficiency Frontier (Dividend Yield vs ROCE)
        scatter2 = ax2.scatter(self.df['Div Yld %'], self.df['ROCE %'], 
                              c=self.df['P/E'], s=150,
                              cmap='viridis', edgecolor='black', linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Dividend Yield %', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ROCE %', fontsize=12, fontweight='bold')
        ax2.set_title('Dividend Yield vs ROCE (Color: P/E)', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(scatter2, ax=ax2, label='P/E Ratio')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Risk-return analysis saved: {save_path}")
        plt.close()
        
        # Identify best risk-adjusted companies
        print("\nBest Risk-Adjusted Companies (Low P/E, High ROCE):")
        risk_adjusted = self.df[(self.df['P/E'] < median_pe) & (self.df['ROCE %'] > median_roce)]
        risk_adjusted = risk_adjusted.sort_values('ROCE %', ascending=False).head(10)
        print(risk_adjusted[['Name', 'P/E', 'ROCE %', 'Fundamental_Score']].to_string(index=False))
    
    def sector_comparative_analysis(self, save_path):
        """Compare sectors based on fundamental metrics"""
        print("\n" + "="*80)
        print("SECTOR COMPARATIVE ANALYSIS")
        print("="*80)
        
        # Categorize sectors
        sectors = []
        for name in self.df['Name']:
            if any(x in name.lower() for x in ['bank', 'finance', 'finserv', 'life', 'insuran']):
                sectors.append('Finance')
            elif any(x in name.lower() for x in ['tech', 'infosys', 'tcs', 'hcl', 'wipro']):
                sectors.append('IT')
            elif any(x in name.lower() for x in ['pharma', 'lab', 'cipla', 'reddy', 'hospital']):
                sectors.append('Healthcare')
            elif any(x in name.lower() for x in ['auto', 'motor', 'maruti', 'eicher', 'bajaj']):
                sectors.append('Automobile')
            elif any(x in name.lower() for x in ['power', 'ntpc', 'coal', 'ongc']):
                sectors.append('Energy')
            else:
                sectors.append('Others')
        
        self.df['Sector'] = sectors
        
        # Create sector comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Average ROCE by Sector
        sector_roce = self.df.groupby('Sector')['ROCE %'].mean().sort_values(ascending=True)
        axes[0, 0].barh(sector_roce.index, sector_roce.values, color='#4CAF50', edgecolor='black')
        axes[0, 0].set_xlabel('Average ROCE %', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Average ROCE by Sector', fontsize=13, fontweight='bold')
        for i, v in enumerate(sector_roce.values):
            axes[0, 0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
        
        # Average P/E by Sector
        sector_pe = self.df.groupby('Sector')['P/E'].mean().sort_values(ascending=True)
        axes[0, 1].barh(sector_pe.index, sector_pe.values, color='#3498db', edgecolor='black')
        axes[0, 1].set_xlabel('Average P/E Ratio', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Average P/E by Sector', fontsize=13, fontweight='bold')
        for i, v in enumerate(sector_pe.values):
            axes[0, 1].text(v + 1, i, f'{v:.1f}', va='center', fontsize=9)
        
        # Average Profit Growth by Sector
        sector_profit = self.df.groupby('Sector')['Qtr Profit Var %'].mean().sort_values(ascending=True)
        colors = ['#27ae60' if x > 0 else '#e74c3c' for x in sector_profit.values]
        axes[1, 0].barh(sector_profit.index, sector_profit.values, color=colors, edgecolor='black')
        axes[1, 0].set_xlabel('Average Profit Growth %', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Average Profit Growth by Sector', fontsize=13, fontweight='bold')
        axes[1, 0].axvline(0, color='black', linewidth=0.8)
        for i, v in enumerate(sector_profit.values):
            axes[1, 0].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        
        # Average Dividend Yield by Sector
        sector_div = self.df.groupby('Sector')['Div Yld %'].mean().sort_values(ascending=True)
        axes[1, 1].barh(sector_div.index, sector_div.values, color='#FF6B6B', edgecolor='black')
        axes[1, 1].set_xlabel('Average Dividend Yield %', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Average Dividend Yield by Sector', fontsize=13, fontweight='bold')
        for i, v in enumerate(sector_div.values):
            axes[1, 1].text(v + 0.05, i, f'{v:.2f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sector comparison saved: {save_path}")
        plt.close()
        
        # Print sector statistics
        print("\nSector Performance Summary:")
        sector_stats = self.df.groupby('Sector').agg({
            'ROCE %': 'mean',
            'P/E': 'mean',
            'Qtr Profit Var %': 'mean',
            'Div Yld %': 'mean',
            'Name': 'count'
        }).round(2)
        sector_stats.columns = ['Avg ROCE%', 'Avg P/E', 'Avg Profit Gr%', 'Avg Div%', 'Count']
        print(sector_stats.to_string())
    
    def valuation_analysis(self, save_path):
        """Analyze valuation metrics across companies"""
        print("\n" + "="*80)
        print("VALUATION ANALYSIS")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # P/E Distribution with box plot
        axes[0, 0].boxplot(self.df['P/E'], vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', edgecolor='black'),
                          medianprops=dict(color='red', linewidth=2))
        axes[0, 0].set_ylabel('P/E Ratio', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('P/E Ratio Distribution (Box Plot)', fontsize=13, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # ROCE Distribution with box plot
        axes[0, 1].boxplot(self.df['ROCE %'], vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightgreen', edgecolor='black'),
                          medianprops=dict(color='red', linewidth=2))
        axes[0, 1].set_ylabel('ROCE %', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('ROCE Distribution (Box Plot)', fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Price vs Market Cap
        scatter = axes[1, 0].scatter(self.df['Mar Cap Rs.Cr.'], self.df['CMP Rs.'], 
                                    c=self.df['ROCE %'], s=100, cmap='plasma',
                                    edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Market Cap (Rs. Cr.)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Current Price (Rs.)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Price vs Market Cap (Color: ROCE)', fontsize=13, fontweight='bold')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        plt.colorbar(scatter, ax=axes[1, 0], label='ROCE %')
        
        # Growth vs Valuation
        scatter = axes[1, 1].scatter(self.df['P/E'], self.df['Qtr Profit Var %'], 
                                    c=self.df['ROCE %'], s=100, cmap='viridis',
                                    edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('P/E Ratio', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Quarterly Profit Growth %', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Valuation vs Growth (Color: ROCE)', fontsize=13, fontweight='bold')
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=axes[1, 1], label='ROCE %')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Valuation analysis saved: {save_path}")
        plt.close()
        
        # Statistical summary
        print("\nValuation Statistics:")
        print(f"  P/E Ratio - Mean: {self.df['P/E'].mean():.2f}, Median: {self.df['P/E'].median():.2f}")
        print(f"  P/E Ratio - Q1: {self.df['P/E'].quantile(0.25):.2f}, Q3: {self.df['P/E'].quantile(0.75):.2f}")
        print(f"  ROCE - Mean: {self.df['ROCE %'].mean():.2f}%, Median: {self.df['ROCE %'].median():.2f}%")
        
        # Identify undervalued companies (Low P/E, High ROCE, Positive Growth)
        undervalued = self.df[
            (self.df['P/E'] < self.df['P/E'].quantile(0.33)) &
            (self.df['ROCE %'] > self.df['ROCE %'].quantile(0.67)) &
            (self.df['Qtr Profit Var %'] > 0)
        ].sort_values('P/E')
        
        if len(undervalued) > 0:
            print(f"\nPotentially Undervalued Companies ({len(undervalued)} found):")
            print("  (Low P/E + High ROCE + Positive Growth)")
            print(undervalued[['Name', 'P/E', 'ROCE %', 'Qtr Profit Var %']].head(10).to_string(index=False))
    
    def growth_momentum_analysis(self, save_path):
        """Analyze growth and momentum indicators"""
        print("\n" + "="*80)
        print("GROWTH & MOMENTUM ANALYSIS")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Profit Growth vs Sales Growth
        scatter = axes[0, 0].scatter(self.df['Qtr Sales Var %'], self.df['Qtr Profit Var %'],
                                    c=self.df['Fundamental_Score'], s=150, cmap='RdYlGn',
                                    edgecolor='black', linewidth=1.5, alpha=0.7)
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[0, 0].set_xlabel('Sales Growth %', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Profit Growth %', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Profit vs Sales Growth', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=axes[0, 0], label='Score')
        
        # Add quadrant labels
        max_x = self.df['Qtr Sales Var %'].max()
        max_y = self.df['Qtr Profit Var %'].max()
        axes[0, 0].text(max_x * 0.7, max_y * 0.7, 'High Growth', fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Growth Distribution
        growth_categories = pd.cut(self.df['Qtr Profit Var %'], 
                                  bins=[-np.inf, 0, 10, 20, np.inf],
                                  labels=['Negative', '0-10%', '10-20%', '20%+'])
        growth_counts = growth_categories.value_counts().sort_index()
        colors_growth = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
        axes[0, 1].bar(range(len(growth_counts)), growth_counts.values, 
                      color=colors_growth, edgecolor='black', linewidth=1.5)
        axes[0, 1].set_xticks(range(len(growth_counts)))
        axes[0, 1].set_xticklabels(growth_counts.index, fontsize=10)
        axes[0, 1].set_ylabel('Number of Companies', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Profit Growth Distribution', fontsize=13, fontweight='bold')
        for i, v in enumerate(growth_counts.values):
            axes[0, 1].text(i, v + 0.3, str(v), ha='center', fontsize=10, fontweight='bold')
        
        # Top Growers
        top_growers = self.df.nlargest(12, 'Qtr Profit Var %')
        bars = axes[1, 0].barh(range(len(top_growers)), top_growers['Qtr Profit Var %'],
                              color='#27ae60', edgecolor='black')
        axes[1, 0].set_yticks(range(len(top_growers)))
        axes[1, 0].set_yticklabels(top_growers['Name'], fontsize=9)
        axes[1, 0].set_xlabel('Profit Growth %', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Top 12 Profit Growers', fontsize=13, fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        # Momentum Score (combining growth and ROCE)
        self.df['Momentum_Score'] = (
            (self.df['Qtr Profit Var %'] > 10).astype(int) +
            (self.df['Qtr Sales Var %'] > 10).astype(int) +
            (self.df['ROCE %'] > 20).astype(int)
        )
        
        momentum_dist = self.df['Momentum_Score'].value_counts().sort_index()
        axes[1, 1].bar(momentum_dist.index, momentum_dist.values,
                      color=['#e74c3c', '#f39c12', '#3498db', '#27ae60'],
                      edgecolor='black', linewidth=1.5)
        axes[1, 1].set_xlabel('Momentum Score', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Number of Companies', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Growth Momentum Score Distribution', fontsize=13, fontweight='bold')
        axes[1, 1].set_xticks([0, 1, 2, 3])
        for i, v in enumerate(momentum_dist.values):
            axes[1, 1].text(momentum_dist.index[i], v + 0.3, str(v), 
                           ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Growth momentum analysis saved: {save_path}")
        plt.close()
        
        # High momentum companies
        high_momentum = self.df[self.df['Momentum_Score'] == 3]
        if len(high_momentum) > 0:
            print(f"\nHigh Momentum Companies (Score 3/3): {len(high_momentum)} companies")
            print(high_momentum[['Name', 'Qtr Profit Var %', 'Qtr Sales Var %', 'ROCE %']].to_string(index=False))


def run_advanced_analysis():
    """Run all advanced analyses"""
    print("\n" + "="*80)
    print(" "*15 + "ADVANCED FUNDAMENTAL ANALYSIS")
    print("="*80)

    # Resolve project root (myProjectFile/)
    # File is at: myProjectFile/src/fundamental_analysis/advanced.py
    # So parents[2] = myProjectFile/
    BASE_DIR = Path(__file__).resolve().parents[2]

    # Define standardized paths
    input_csv     = BASE_DIR / "data" / "processed"   / "strong_fundamental_companies.csv"
    reports_dir   = BASE_DIR / "data" / "reports"     / "fundamental"

    # Chart output paths
    chart_correlation  = reports_dir / "correlation_heatmap.png"
    chart_risk_return  = reports_dir / "risk_return_analysis.png"
    chart_sector       = reports_dir / "sector_comparison.png"
    chart_valuation    = reports_dir / "valuation_analysis.png"
    chart_momentum     = reports_dir / "growth_momentum_analysis.png"

    # Ensure output directory exists
    reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_csv)
        analyzer = AdvancedAnalyzer(df)
        
        # Run all analyses — each method now receives its own save path
        analyzer.correlation_analysis(chart_correlation)
        analyzer.risk_return_analysis(chart_risk_return)
        analyzer.sector_comparative_analysis(chart_sector)
        analyzer.valuation_analysis(chart_valuation)
        analyzer.growth_momentum_analysis(chart_momentum)
        
        print("\n" + "="*80)
        print("✓ ADVANCED ANALYSIS COMPLETED!")
        print("="*80)
        print("\nGenerated Files:")
        print(f"  1. {chart_correlation}")
        print(f"  2. {chart_risk_return}")
        print(f"  3. {chart_sector}")
        print(f"  4. {chart_valuation}")
        print(f"  5. {chart_momentum}")
        print("="*80)
        
    except FileNotFoundError:
        print(f"\n❌ Error: File '{input_csv}' not found!")
        print("Please run fundamental.py first to generate the processed data file.")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    run_advanced_analysis()
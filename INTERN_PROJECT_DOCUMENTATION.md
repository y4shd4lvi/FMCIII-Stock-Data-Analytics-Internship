# NSE Analytica — NIFTY 50 Stock Analysis System
### Internship Project Documentation

> **Author:** Intern  
> **Mentor Presentation Draft**  
> **Tech Stack:** Python · Pandas · Matplotlib · Streamlit  
> **Domain:** Quantitative Finance · Stock Market Analysis  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Data Pipeline](#3-data-pipeline)
4. [Module 1 — NSE Data Utilities](#4-module-1--nse-data-utilities)
5. [Module 2 — Fundamental Analysis](#5-module-2--fundamental-analysis)
6. [Module 3 — Volume Analysis](#6-module-3--volume-analysis)
7. [Module 4 — Trend Analysis](#7-module-4--trend-analysis)
8. [Streamlit Presentation Layer](#8-streamlit-presentation-layer)
9. [Technical Methods & Metrics Reference](#9-technical-methods--metrics-reference)
10. [Design Principles](#10-design-principles)
11. [How to Run](#11-how-to-run)
12. [Key Learnings](#12-key-learnings)

---

## 1. Project Overview

**NSE Analytica** is a modular, end-to-end stock analysis pipeline built for the **NIFTY 50** index — India's benchmark index of the 50 largest companies listed on the National Stock Exchange (NSE).

The system is designed around three pillars of stock analysis used by professional investors:

| Pillar | Question It Answers |
|---|---|
| **Fundamental Analysis** | Is this company financially healthy and fairly valued? |
| **Volume Analysis** | Is smart money accumulating or distributing this stock? |
| **Trend Analysis** | What direction is the stock moving, and how strong is that trend? |

The project also includes a **live data downloader** that pulls official NSE bhavcopy data, and a **Streamlit web interface** that makes all analysis accessible through a clean dashboard — no command line needed.

---

## 2. Project Structure

```
myProjectFile/
│
├── NSEenv/                          # Python virtual environment
│
├── app/                             # 🖥️  Streamlit UI layer
│   ├── main.py                      # Entry point — sidebar nav + global CSS
│   └── pages/
│       ├── p0_overview.py           # Project status dashboard
│       ├── p1_downloader.py         # NSE downloader UI
│       ├── p2_fundamental.py        # Fundamental analysis UI
│       ├── p3_volume.py             # Volume analysis UI
│       └── p4_trend.py              # Trend analysis UI
│
├── src/                             # 🧠  Business logic (OOP modules)
│   ├── __init__.py
│   │
│   ├── fundamental_analysis/
│   │   ├── __init__.py
│   │   ├── fundamental.py           # FundamentalAnalyzer class
│   │   └── advanced.py              # AdvancedAnalyzer class
│   │
│   ├── volume_analysis/
│   │   ├── __init__.py
│   │   ├── volume.py                # Nifty50VolumeAnalyzer class
│   │   └── visualization.py         # VolumeResultsVisualizer class
│   │
│   ├── trend_analysis/
│   │   ├── __init__.py
│   │   └── trend.py                 # Nifty50TrendAnalyzer class
│   │
│   └── nse/
│       ├── __init__.py
│       ├── downloader.py            # NSEBhavcopyDownloader class
│       └── preprocess.py            # BhavcopyPreprocessor class
│
├── data/                            # 💾  Data storage (layered)
│   ├── raw/
│   │   ├── bhavcopy.csv             # Master bhavcopy (downloaded from NSE)
│   │   ├── Top_50_Companies_Data.csv# Fundamental data input
│   │   └── bhavcopies/              # Temporary daily bhavcopy files
│   │
│   ├── processed/
│   │   ├── strong_fundamental_companies.csv
│   │   └── nifty50_volume_analysis_report.csv
│   │
│   └── reports/
│       ├── fundamental/             # PNG charts from fundamental analysis
│       ├── volume/                  # PNG charts from volume analysis
│       └── trend/
│           ├── reports/             # STOCKNAME_report.txt (one per stock)
│           └── charts/              # STOCKNAME_chart.png (one per stock)
│
├── requirements.txt
└── README.md
```

### Architecture Principle

The project follows a strict **three-layer separation**:

```
UI Layer (app/)
    ↓  calls
Business Logic Layer (src/)
    ↓  reads/writes
Data Layer (data/)
```

This means the Streamlit pages never contain analysis logic — they only call the classes from `src/` and display results. This makes the codebase maintainable and testable.

---

## 3. Data Pipeline

The entire data flow is linear and deterministic:

```
NSE Website
    │
    ▼
NSEBhavcopyDownloader          ← downloads PR.zip for each trading day
    │
    ▼
BhavcopyPreprocessor            ← cleans dates, numerics, categoricals
    │
    ▼
data/raw/bhavcopy.csv           ← master dataset (all NIFTY 50, all dates)
    │
    ├──▶ Nifty50VolumeAnalyzer  ──▶ data/processed/nifty50_volume_analysis_report.csv
    │
    └──▶ Nifty50TrendAnalyzer   ──▶ data/reports/trend/reports/*.txt
                                ──▶ data/reports/trend/charts/*.png

data/raw/Top_50_Companies_Data.csv
    │
    └──▶ FundamentalAnalyzer    ──▶ data/processed/strong_fundamental_companies.csv
             │                  ──▶ data/reports/fundamental/*.png
             ▼
         AdvancedAnalyzer       ──▶ data/reports/fundamental/*.png (6 extra charts)
```

---

## 4. Module 1 — NSE Data Utilities

**Location:** `src/nse/`

### 4.1 NSEBhavcopyDownloader

**What it does:** Downloads the official **Bhavcopy** (end-of-day price report) from the NSE API for any date range, and appends all daily files into one consolidated master CSV.

**Bhavcopy** is NSE's official daily price and volume report — it contains every equity's open, high, low, close, volume, and 52-week range for that trading day.

#### Key Methods

| Method | Description |
|---|---|
| `download_range(start, end)` | Iterates through each trading day, downloads and saves daily bhavcopy CSV |
| `_extract_pr_csv(zip_bytes)` | Unzips the downloaded PR.zip and extracts the CM (capital market) CSV inside |
| `incremental_append(delete_after_append)` | Merges all new daily CSVs into master bhavcopy.csv; skips already-loaded dates |
| `run(start, end)` | Convenience method: download → append → delete daily files in one call |

#### Important Design Decisions

- **Incremental loading** — already-processed dates are tracked so re-running never duplicates rows
- **Auto-delete daily files** — after appending to master, temporary per-day files are deleted to keep `data/raw/` clean
- **Session-based HTTP** — uses `requests.Session` with NSE's Referer header to bypass bot detection
- **Streamlit-friendly logging** — all output goes through a configurable `log_callback` (defaults to `print`, can be replaced with `st.write`)

---

### 4.2 BhavcopyPreprocessor

**What it does:** Cleans and type-converts the raw bhavcopy CSV so downstream analysis modules receive consistent, correctly typed data.

#### Preprocessing Steps (method chaining pipeline)

```python
preprocessor.load_data()
            .convert_date()              # TRADE_DATE → datetime64
            .convert_price_columns()     # Remove commas, coerce to float
            .convert_trade_columns()     # NET_TRDQTY, NET_TRDVAL → numeric
            .convert_categorical_columns() # MKT, SECURITY → category dtype
            .convert_date_parts()        # Extract year, month, day columns
```

#### Why Each Step Matters

| Step | Problem Solved |
|---|---|
| `convert_date()` | NSE stores dates as strings like "2026-01-05"; parsing enables date arithmetic |
| `convert_price_columns()` | NSE bhavcopy often has commas in numbers (e.g. "1,234.50"); pandas reads these as strings |
| `convert_categorical_columns()` | Reduces memory usage significantly when 50 stocks repeat thousands of times |
| `convert_date_parts()` | Enables fast date-based filtering without re-parsing strings |

---

## 5. Module 2 — Fundamental Analysis

**Location:** `src/fundamental_analysis/`  
**Input:** `data/raw/Top_50_Companies_Data.csv`  
**Outputs:** `data/processed/strong_fundamental_companies.csv` + 7 PNG charts

Fundamental analysis evaluates a company's **financial health and intrinsic value** using reported financial data — independent of stock price movements.

---

### 5.1 FundamentalAnalyzer

#### The Scoring System

Each company is scored 0–5 based on five criteria:

| Criterion | Threshold | What It Measures |
|---|---|---|
| **P/E Ratio** | < 25 | Valuation — how much investors pay per rupee of earnings |
| **ROCE %** | > 15% | Capital efficiency — profit generated per rupee of capital employed |
| **Qtr Profit Var %** | > 0% | Profitability growth — earnings improving quarter-over-quarter |
| **Div Yld %** | > 1% | Shareholder returns — income paid to investors |
| **Qtr Sales Var %** | > 10% | Revenue expansion — top-line business growth |

Companies scoring **≥ 3 out of 5** are classified as **fundamentally strong**.

#### Metrics Explained

**P/E Ratio (Price-to-Earnings)**
```
P/E = Current Market Price / Earnings Per Share
```
- Low P/E → stock may be undervalued relative to earnings
- High P/E → investors expect high future growth (or stock is overpriced)
- Threshold of 25 is standard for Indian large-cap equities

**ROCE — Return on Capital Employed**
```
ROCE = EBIT / Capital Employed × 100
     = Earnings Before Interest & Tax / (Total Assets - Current Liabilities) × 100
```
- Measures how efficiently management uses capital to generate profit
- ROCE > 15% is considered healthy for NIFTY 50 companies
- ROCE > 30% is considered exceptional

**Quarterly Sales & Profit Variance**
```
Qtr Profit Var % = (Current Qtr Profit - Previous Qtr Profit) / Previous Qtr Profit × 100
```
- Captures business momentum
- Positive variance = company is growing
- Filters out companies in financial distress

**Dividend Yield**
```
Div Yield % = Annual Dividend Per Share / Current Market Price × 100
```
- Measures income return on investment
- Important for value and income investors

#### Visualizations Generated

| Chart | What It Shows |
|---|---|
| Score Distribution | Bar chart of how many companies scored 3, 4, or 5 |
| Top 10 by ROCE | Horizontal bar — most capital-efficient companies |
| P/E Distribution | Histogram with mean/median lines |
| Market Cap vs ROCE Scatter | Bubble chart — size = market cap, color = score |
| Top 10 Dividend Yielders | Horizontal bar chart |
| Top 10 Profit Growth | Color-coded (green = positive, red = negative) |
| Top 10 Sales Growth | Horizontal bar chart |
| Sector Distribution | Pie chart — Finance, IT, Healthcare, Automobile, etc. |
| P/E vs Price Scatter | Bubble chart — size = market cap, color = ROCE |
| Profit vs Sales Comparison | Grouped bar for top 8 companies |

---

### 5.2 AdvancedAnalyzer

Takes the filtered `strong_fundamental_companies.csv` and runs deeper multi-dimensional analysis.

#### Methods & Their Purpose

**`correlation_analysis()`**
- Builds a **Pearson correlation matrix** across: P/E, ROCE, Div Yield, Profit Growth, Sales Growth, Market Cap
- Rendered as a lower-triangle heatmap (seaborn)
- Identifies which metrics move together (e.g., does high ROCE correlate with high dividend yield?)

**`risk_return_analysis()`**
- **Axis:** P/E Ratio (X = risk proxy) vs ROCE % (Y = return proxy)
- Divides chart into 4 quadrants using median values:
  - Low Risk / High Return → ideal investment zone
  - High Risk / High Return → growth plays
  - Low Risk / Low Return → defensive positions
  - High Risk / Low Return → avoid

**`sector_comparative_analysis()`**
- Groups companies into 6 sectors: Finance, IT, Healthcare, Automobile, Energy, Others
- Compares average ROCE, P/E, Profit Growth, Dividend Yield per sector
- Identifies which sectors are most attractive as a group

**`valuation_analysis()`**
- Box plots for P/E and ROCE distribution (shows outliers and spread)
- Log-scale scatter of Price vs Market Cap
- Identifies **potentially undervalued companies**: bottom-tercile P/E + top-tercile ROCE + positive profit growth

**`growth_momentum_analysis()`**
- **Momentum Score (0–3):** 1 point each for Profit Growth > 10%, Sales Growth > 10%, ROCE > 20%
- Companies scoring 3/3 are flagged as **high momentum** picks
- Profit growth distribution categorized: Negative / 0–10% / 10–20% / 20%+

---

## 6. Module 3 — Volume Analysis

**Location:** `src/volume_analysis/`  
**Input:** `data/raw/bhavcopy.csv`  
**Output:** `data/processed/nifty50_volume_analysis_report.csv`

Volume analysis is based on the principle: **price shows what is happening, volume shows why**. Large institutional investors (mutual funds, FIIs) cannot hide their activity — it shows up in volume patterns.

---

### 6.1 Nifty50VolumeAnalyzer

#### Volume Metrics Calculated

**Volume Moving Averages**
```
VOL_MA_5  = 5-day  rolling mean of NET_TRDQTY
VOL_MA_20 = 20-day rolling mean of NET_TRDQTY
VOL_MA_50 = 50-day rolling mean of NET_TRDQTY
```

**Volume Ratio**
```
VOL_RATIO_20 = Today's Volume / 20-day Average Volume
```
- Ratio > 1.0 → above-average interest
- Ratio > 2.0 → Volume Spike (potential institutional activity)

**On-Balance Volume (OBV)**
```
If Close > Previous Close:  OBV = Previous OBV + Today's Volume
If Close < Previous Close:  OBV = Previous OBV - Today's Volume
If Close = Previous Close:  OBV = Previous OBV
```
- Rising OBV while price rises → confirmed uptrend
- Rising OBV while price falls → bullish divergence (accumulation in disguise)
- Developed by Joe Granville (1963) — one of the oldest volume indicators

**Accumulation/Distribution (A/D) Line**
```
Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
Money Flow Volume     = Money Flow Multiplier × Volume
A/D Line              = Cumulative sum of Money Flow Volume
```
- Developed by Marc Chaikin
- Measures the degree to which a stock is being accumulated (bought) vs distributed (sold)
- Ranges from -1 to +1: positive = more buying pressure than selling

**Volume Trend**
```
VOL_TREND = "Increasing" if VOL_MA_5 > VOL_MA_20 else "Decreasing"
```
Short-term volume above long-term volume average indicates growing interest.

---

#### Breakout / Breakdown Detection

```python
near_52w_high = Close >= 52W_High × 0.98   # within 2% of 52-week high
near_52w_low  = Close <= 52W_Low  × 1.02   # within 2% of 52-week low
volume_confirmed = VOL_RATIO_20 > 1.2       # above-average volume

BREAKOUT_CONFIRMED  → near_52w_high AND volume_confirmed AND price_change > +2%
BREAKOUT_FORMING    → near_52w_high AND price_change > +1%
BREAKDOWN_CONFIRMED → near_52w_low  AND volume_confirmed AND price_change < -2%
BREAKDOWN_FORMING   → near_52w_low  AND price_change < -1%
```

Volume-confirmed breakouts have significantly higher follow-through probability than low-volume breakouts.

---

#### Risk Scoring (0–100)

| Factor | Max Points | Logic |
|---|---|---|
| Price Volatility | 30 | Std dev of 20-day returns: >3% = 30pts, >2% = 20pts, >1% = 10pts |
| Volume Inconsistency | 25 | Coefficient of variation of volume: >1.0 = 25pts |
| 52-Week Position | 20 | At extremes (top 5% or bottom 5%) = 20pts |
| Recent Price Drop | 15 | Single-day drop > 3% = 15pts |
| Low Liquidity | 10 | VOL_RATIO_20 < 0.5 (trading well below average) = 10pts |

Risk Levels: **LOW** (0–40) · **MEDIUM** (41–70) · **HIGH** (71–100)

---

#### Recommendation Engine

Four signal categories, each contributing buy or sell points:

| Signal | Buy Points | Sell Points |
|---|---|---|
| STRONG_BULLISH trend | +2 | — |
| BULLISH trend | +1 | — |
| STRONG_ACCUMULATION | +2 | — |
| BREAKOUT_CONFIRMED | +2 | — |
| Profit Probability ≥ 70% | +1 | — |
| STRONG_BEARISH trend | — | +2 |
| STRONG_DISTRIBUTION | — | +2 |
| BREAKDOWN_CONFIRMED | — | +2 |
| Risk Score > 70 | — | +1 |

**Final Call:**

| Condition | Recommendation |
|---|---|
| Buy signals ≥ 4 AND risk < 60 | `STRONG_BUY` |
| Buy signals ≥ 2 AND sell = 0 | `BUY` |
| Sell signals ≥ 4 OR risk > 80 | `STRONG_SELL` |
| Sell signals ≥ 2 AND buy = 0 | `SELL` |
| Otherwise | `HOLD` |

---

## 7. Module 4 — Trend Analysis

**Location:** `src/trend_analysis/`  
**Input:** `data/raw/bhavcopy.csv`  
**Output:** Per-stock `.txt` reports + `.png` charts in `data/reports/trend/`

Trend analysis uses **price action** and **moving average** techniques to classify the direction and strength of each stock's trend.

---

### 7.1 Nifty50TrendAnalyzer

#### Moving Averages

```
MA_50  = 50-day  Simple Moving Average of Close Price
MA_200 = 200-day Simple Moving Average of Close Price
```

The **200-day MA** is the most widely watched indicator by institutional investors worldwide:
- Price **above** rising 200-day MA → long-term uptrend
- Price **below** falling 200-day MA → long-term downtrend
- The 200-day MA acts as dynamic support/resistance

---

#### Swing Point Detection

Swing points are **local price extremes** — the turning points that define trend structure.

```python
# Swing High: local maximum using a 11-bar (5+1+5) rolling window
swing_high = HIGH_PRICE.rolling(window=11, center=True)
             .apply(lambda x: x[5] == x.max())

# Swing Low: local minimum using same window
swing_low  = LOW_PRICE.rolling(window=11, center=True)
             .apply(lambda x: x[5] == x.min())
```

---

#### Market Structure Analysis (Price Action)

Institutional traders read markets through **swing point sequences**:

```
UPTREND   = Higher High (HH) + Higher Low (HL)
            → Each rally exceeds the previous rally high
            → Each pullback stays above the previous pullback low

DOWNTREND = Lower High (LH) + Lower Low (LL)
            → Each rally fails below the previous rally high
            → Each pullback breaks below the previous low

SIDEWAYS  = Neither HH/HL nor LH/LL pattern — range-bound
```

This is the foundational concept of **Dow Theory** (Charles Dow, 1900) and is used in modern **Smart Money Concepts (SMC)** trading.

---

#### Golden Cross & Death Cross Detection

```
Golden Cross = 50-day MA crosses ABOVE 200-day MA
             → Bullish long-term signal
             → Historically associated with sustained uptrends

Death Cross  = 50-day MA crosses BELOW 200-day MA
             → Bearish long-term signal
             → Historically associated with sustained downtrends
```

The system also tracks:
- **Days since cross** — recent crosses (< 60 days) are weighted more heavily
- **Price confirmation** — whether price is above/below the cross level currently

---

#### Trend Strength Scoring (–12 to +12)

| Factor | +Points | –Points |
|---|---|---|
| Market Structure | +3 (HH/HL) | –3 (LH/LL) |
| 200-day MA Position | +2 (above) | –2 (below) |
| 200-day MA Slope | +2 (rising) | –2 (falling) |
| Recent Crossover (< 60 days) | +3 (golden) | –3 (death) |
| Price vs MAs alignment | +2 (Price > MA50 > MA200) | –2 (Price < MA50 < MA200) |

**Classification:**

| Score | Label |
|---|---|
| ≥ 7 | 🟢 STRONG UPTREND |
| 3 to 6 | 🟡 WEAK/DEVELOPING UPTREND |
| –2 to 2 | ⚪ SIDEWAYS / RANGE-BOUND |
| –6 to –3 | 🟠 WEAK DOWNTREND |
| ≤ –7 | 🔴 STRONG DOWNTREND |

---

#### Per-Stock Output

For each of the 50 NIFTY stocks, the system generates:

**Text Report** (`STOCKNAME_report.txt`) containing:
1. Current price
2. Market structure pattern + detail
3. 200-day MA value, position (above/below), slope, institutional positioning
4. Golden/Death cross date, price at cross, confirmation status
5. Trend classification + composite score

**Chart** (`STOCKNAME_chart.png`) containing:
1. Price line + 50-day MA + 200-day MA
2. Swing high/low markers (▲▼)
3. Golden/Death cross vertical lines with annotations
4. Volume bar chart subplot

---

## 8. Streamlit Presentation Layer

**Location:** `app/`  
**Run with:** `streamlit run app/main.py`

### Architecture

```
main.py
  ├── Global CSS (dark theme, custom fonts, component styles)
  ├── Sidebar radio navigation
  └── Routes to page modules via render() functions
```

### Pages

| Page | File | Key Features |
|---|---|---|
| Overview | `p0_overview.py` | File existence checks, report counts, pipeline guide |
| NSE Downloader | `p1_downloader.py` | Date range picker, live log stream, preprocess toggle |
| Fundamental Analysis | `p2_fundamental.py` | Run basic + advanced, metrics, filterable table, chart selector |
| Volume Analysis | `p3_volume.py` | Run button, 3-filter bar, top-5 detail expanders |
| Trend Analysis | `p4_trend.py` | Run button, stock report browser, chart browser, summary viewer |

### Design Choices

- **Dark theme** with teal accent (`#00d4aa`) — chosen for financial data readability (dark backgrounds reduce eye strain for data-heavy screens)
- **Fonts:** Bebas Neue (headers), DM Sans (body), DM Mono (data/numbers)
- **Streamlit-friendly logging:** Every analysis class accepts a `log_callback` parameter — in the UI this streams live progress updates into a styled terminal-like box
- **No logic in UI files** — pages only call `src/` classes; they never manipulate data directly

---

## 9. Technical Methods & Metrics Reference

Quick reference table for mentor discussion:

| Metric / Method | Category | Formula / Logic | Interpretation |
|---|---|---|---|
| P/E Ratio | Fundamental | Price / EPS | Lower = cheaper valuation |
| ROCE % | Fundamental | EBIT / Capital Employed × 100 | Higher = more efficient |
| Fundamental Score | Fundamental | Sum of 5 binary criteria (0–5) | ≥ 3 = strong |
| Momentum Score | Fundamental | Sum of 3 growth criteria (0–3) | 3 = high momentum |
| OBV | Volume | Cumulative signed volume | Rising = accumulation |
| A/D Line | Volume | Cumulative (MFM × Volume) | Rising = buying pressure |
| Volume Ratio | Volume | Today Vol / 20-day Avg Vol | >2 = spike |
| Risk Score | Volume | Weighted sum of 5 risk factors (0–100) | <40 = low risk |
| Profit Probability | Volume | Adjusted base 50% score | >70% = favorable |
| MA 50/200 | Trend | Simple moving averages | Direction & support |
| Golden Cross | Trend | MA50 crosses above MA200 | Bullish long-term |
| Death Cross | Trend | MA50 crosses below MA200 | Bearish long-term |
| HH/HL Pattern | Trend | Swing point sequence comparison | Uptrend structure |
| LH/LL Pattern | Trend | Swing point sequence comparison | Downtrend structure |
| Trend Score | Trend | Composite –12 to +12 | ≥7 = strong uptrend |

---

## 10. Design Principles

The codebase was built with the following engineering principles:

### Separation of Concerns
Every layer has one job:
- `src/` → analysis only, no UI
- `app/` → display only, no analysis logic  
- `data/` → storage only, no processing

### OOP Throughout
Every analysis component is a class with clear `__init__`, method chaining where appropriate, and a single `process()` or `run()` entry point. This makes it easy to:
- Import classes independently
- Test individual methods
- Extend with new features

### Streamlit-Ready Design
All classes support a `log_callback` parameter:
```python
# Script mode (default)
analyzer = NSEBhavcopyDownloader()                     # uses print()

# Streamlit mode
analyzer = NSEBhavcopyDownloader(log_callback=st.write) # streams to UI
```

### Path Resolution
All file paths use `pathlib.Path(__file__).resolve().parents[N]` to compute the project root at runtime — no hardcoded absolute paths. The project works on any machine regardless of where it is installed.

### Incremental Processing
The bhavcopy downloader tracks already-processed dates and skips them on re-runs. Analysis modules check for existing outputs before re-running expensive computations.

---

## 11. How to Run

### Setup
```bash
# 1. Clone / navigate to project
cd myProjectFile

# 2. Create and activate virtual environment
python -m venv NSEenv
source NSEenv/bin/activate        # Mac/Linux
NSEenv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Recommended requirements.txt
```
streamlit
pandas
numpy
matplotlib
seaborn
scipy
requests
pathlib
```

### Running the Dashboard
```bash
streamlit run app/main.py
```

### Running Modules Standalone
```bash
# Download bhavcopy
python src/nse/downloader.py

# Preprocess
python src/nse/preprocess.py

# Fundamental analysis
python src/fundamental_analysis/fundamental.py

# Advanced fundamental
python src/fundamental_analysis/advanced.py

# Volume analysis
python src/volume_analysis/volume.py

# Trend analysis
python src/trend_analysis/trend.py
```

---

## 12. Key Learnings

Through building this project, the following concepts were applied hands-on:

| Area | What Was Learned |
|---|---|
| **Python OOP** | Designing reusable classes with clear interfaces, method chaining pattern |
| **Pandas** | Data cleaning pipelines, rolling windows, groupby aggregations, dtype optimization |
| **Financial Analysis** | P/E, ROCE, OBV, A/D Line, moving averages, Dow Theory, swing points |
| **Data Engineering** | Incremental data loading, deduplication, multi-file merging |
| **API Integration** | Session-based HTTP requests, handling zip files in memory, NSE data formats |
| **Streamlit** | Multi-page apps, live logging callbacks, file download buttons, dataframe display |
| **Software Architecture** | Three-layer separation, path-agnostic code with `pathlib`, modular design |
| **Matplotlib/Seaborn** | Multi-subplot figures, scatter plots with color encoding, correlation heatmaps |

---

*Document prepared for mentor review · NSE Analytica Internship Project*

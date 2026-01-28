# Portfolio Optimization Project

## Overview
This project applies **time series forecasting** and **Modern Portfolio Theory (MPT)** to enhance portfolio management for **Guide Me in Finance (GMF) Investments**. 

We analyze three assets:
* **TSLA** (Tesla) – High-growth, high-volatility stock
* **BND** (Vanguard Total Bond Market ETF) – Low-risk, stable income
* **SPY** (S&P 500 ETF) – Moderate-risk, broad market exposure

**Data period**: January 1, 2015 – January 15, 2026 (fetched via YFinance).

### Core Objectives
1.  **Preprocess and explore** historical data (Task 1).
2.  **Build and compare** ARIMA/SARIMA and LSTM forecasting models for TSLA (Task 2).
3.  **Generate future forecasts** and extract actionable insights (Task 3).
4.  **Optimize portfolio allocations** using hybrid expected returns (Task 4).
5.  **Backtest strategy** vs. benchmark on 2025–2026 data (Task 5).

**Key Technologies**: Python, pandas, numpy, matplotlib, seaborn, statsmodels, pmdarima, scikit-learn, yfinance.

---

## Project Structure
```text
portfolio-optimization/
├── .vscode/                        # VS Code settings
├── .github/
│   └── workflows/
│       └── unittests.yml           # CI/CD for unit tests
├── data/
│   ├── raw/                        # Original fetched data
│   └── processed/
│       └── cleaned_data.csv        # Cleaned & processed dataset
├── notebooks/
│   ├── data_preprocessing_eda.ipynb    # Task 1: Preprocessing & EDA
│   ├── time_series_modeling.ipynb      # Task 2: ARIMA/SARIMA + LSTM modeling
│   ├── market_trend_forecasting.ipynb  # Task 3: Future forecasting & analysis
│   ├── portfolio_optimization.ipynb    # Task 4: MPT optimization & Efficient Frontier
│   └── strategy_backtesting.ipynb      # Task 5: Backtesting & performance comparison
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Fetch, clean, calculate returns
│   ├── eda_utils.py                # Plotting, ADF test, risk metrics
│   ├── models.py                   # ARIMA & LSTM implementation
│   └── portfolio_utils.py          # Covariance, optimization, backtest functions
├── tests/                          # Unit tests
├── scripts/                        # Helper scripts
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
## Setup Instructions
1. Clone the repositoryBashgit clone <your-repository-url>
cd portfolio-optimization
2. Create & activate virtual environmentBash# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependenciesBashpip install -r requirements.txt
Packages include: pandas, numpy, matplotlib, seaborn, yfinance, statsmodels, pmdarima, scikit-learn.4. Run NotebooksExecute the notebooks in the following order:Task 1 → Task 2 → Task 3 → Task 4 → Task 5Note: Internet access is required for the initial YFinance data fetch in Task 1.Tasks Overview & Key ResultsTask 1: Data Preprocessing & EDAStationarity: ADF test showed Prices are non-stationary ($p \approx 0.82$), while Returns are stationary ($p \approx 0.0$).Risk Metrics:VaR (95%): TSLA -5.25%, SPY -1.67%, BND -0.48%Sharpe Ratio: TSLA 0.823, SPY 0.804, BND 0.381Task 2: Time Series Forecasting ModelsModels: ARIMA/SARIMA vs. LSTM.Split: Train (2015–2024) and Test (2025–2026).Result: LSTM captured non-linear patterns, while ARIMA provided a baseline for trend direction.Task 3: Forecast Future Market TrendsModel: SARIMAX(1,1,0) retrained on full data.Horizon: 252 trading days (~12 months) ahead.Diagnostics: High kurtosis (11.34) indicates fat tails; heteroskedasticity suggests volatility clustering.Task 4: Portfolio OptimizationMax Sharpe Portfolio: BND 55.9%, SPY 44.1%, TSLA 0.0%Return: 7.04%, Vol: 8.68%, Sharpe: 0.811Min Volatility Portfolio: BND 94.3%, SPY 5.7%, TSLA 0.0%Return: 2.58%, Vol: 5.26%Insight: TSLA was excluded from these specific optimizations due to high variance and limited diversification benefits.Task 5: Strategy BacktestingBenchmark: 60/40 (SPY/BND)Results:Strategy (Aggressive): 20.0% Total Return | -30.6% Max DrawdownBenchmark: 15.9% Total Return | -11.3% Max DrawdownConclusion: The aggressive strategy offers higher returns but significantly higher risk.DeliverablesModular Python code in src/.Five Jupyter notebooks with full analysis and visualizations.Processed datasets.Unit tests and CI workflow.
## Authors
Abenezer Sileshi 
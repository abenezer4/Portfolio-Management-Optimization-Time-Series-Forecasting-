# Portfolio Optimization Project

## Overview
This project focuses on time series forecasting and portfolio optimization for financial assets (TSLA, BND, SPY). It aims to analyze historical data, build forecasting models for Tesla (TSLA), and optimize a portfolio using these insights.

## Project Structure
```
portfolio-optimization/
├── .vscode/            # VS Code settings
├── .github/workflows/  # CI/CD workflows
├── data/               # Data storage
│   └── processed/      # Processed data
├── notebooks/          # Jupyter Notebooks
│   ├── data_preprocessing_eda.ipynb  # Task 1: Preprocessing & EDA
│   └── time_series_modeling.ipynb    # Task 2: Forecasting Models
├── src/                # Source code
│   ├── data_loader.py  # Data fetching and cleaning
│   ├── eda_utils.py    # Visualization and analysis tools
│   └── models.py       # ARIMA and LSTM model implementations
├── tests/              # Unit tests
├── scripts/            # Helper scripts
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd portfolio-optimization
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Task 1: Data Preprocessing & EDA
Run the notebook `notebooks/data_preprocessing_eda.ipynb` to:
- Fetch historical data for TSLA, BND, SPY from YFinance (2015-2026).
- Clean and normalize the data.
- Visualize price trends, returns, and volatility.
- Perform stationarity tests (ADF).
- Calculate risk metrics (VaR, Sharpe Ratio).

## Task 2: Time Series Modeling
Run the notebook `notebooks/time_series_modeling.ipynb` to:
- Split data into training (2015-2024) and testing (2025-2026) sets.
- Train and evaluate ARIMA/SARIMA models.
- Train and evaluate LSTM models.
- Compare model performance (MAE, RMSE, MAPE).
- Visualize forecasts against actual data.

## Deliverables
- **Code:** Python scripts in `src/` and notebooks in `notebooks/`.
- **Analysis:** EDA results and Model evaluation in notebooks.
- **Documentation:** This README and inline comments.

## Authors
[Your Name]

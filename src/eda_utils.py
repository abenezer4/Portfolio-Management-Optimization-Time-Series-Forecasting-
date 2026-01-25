import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def plot_closing_prices(df, title="Closing Prices"):
    """Plots the closing prices of the assets."""
    plt.figure(figsize=(14, 7))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_daily_returns(returns, title="Daily Returns"):
    """Plots the daily returns."""
    plt.figure(figsize=(14, 7))
    for column in returns.columns:
        plt.plot(returns.index, returns[column], label=column, alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_volatility(returns, window=30):
    """Calculates and plots rolling volatility (standard deviation)."""
    rolling_std = returns.rolling(window=window).std()
    plt.figure(figsize=(14, 7))
    for column in rolling_std.columns:
        plt.plot(rolling_std.index, rolling_std[column], label=f'{column} {window}-Day Rolling Std')
    plt.title(f'Rolling Volatility ({window}-Day Window)')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Std Dev)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return rolling_std

def perform_adf_test(series):
    """Performs Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    if result[1] <= 0.05:
        print("Result: The series is stationary.")
    else:
        print("Result: The series is non-stationary.")
    return result

def calculate_risk_metrics(returns, confidence_level=0.05):
    """Calculates Value at Risk (VaR) and Sharpe Ratio."""
    metrics = {}
    
    # VaR
    var = returns.quantile(confidence_level)
    
    # Sharpe Ratio (assuming 0 risk-free rate for simplicity, annualized)
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = mean_return / volatility
    
    metrics['VaR_95'] = var
    metrics['Sharpe_Ratio'] = sharpe_ratio
    
    return pd.DataFrame(metrics)

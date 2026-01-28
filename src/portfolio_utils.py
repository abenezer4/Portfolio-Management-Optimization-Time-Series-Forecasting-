import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting

def calculate_covariance_matrix(prices_df):
    """
    Calculates the covariance matrix of daily returns.
    """
    return risk_models.sample_cov(prices_df)

def calculate_historical_returns(prices_df):
    """
    Calculates the annualized historical returns (mean daily returns * 252).
    """
    return expected_returns.mean_historical_return(prices_df)

def optimize_portfolio(expected_ret, cov_matrix):
    """
    Optimizes the portfolio for Maximum Sharpe Ratio and Minimum Volatility.
    
    Args:
        expected_ret (pd.Series): Expected returns for each asset.
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
        
    Returns:
        dict: A dictionary containing weights and performance for both portfolios.
    """
    results = {}
    
    # 1. Max Sharpe Ratio
    ef = EfficientFrontier(expected_ret, cov_matrix)
    weights_sharpe = ef.max_sharpe()
    cleaned_weights_sharpe = ef.clean_weights()
    perf_sharpe = ef.portfolio_performance(verbose=False)
    
    results['Max_Sharpe'] = {
        'weights': cleaned_weights_sharpe,
        'performance': perf_sharpe # (return, volatility, sharpe)
    }
    
    # 2. Min Volatility
    ef_min = EfficientFrontier(expected_ret, cov_matrix)
    weights_min = ef_min.min_volatility()
    cleaned_weights_min = ef_min.clean_weights()
    perf_min = ef_min.portfolio_performance(verbose=False)
    
    results['Min_Volatility'] = {
        'weights': cleaned_weights_min,
        'performance': perf_min
    }
    
    return results

def plot_efficient_frontier(expected_ret, cov_matrix, optimal_results):
    """
    Plots the Efficient Frontier and marks the optimal portfolios.
    """
    ef = EfficientFrontier(expected_ret, cov_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
    
    # Extract optimal points
    ret_sharpe, vol_sharpe, _ = optimal_results['Max_Sharpe']['performance']
    ret_min, vol_min, _ = optimal_results['Min_Volatility']['performance']
    
    # Plot markers
    ax.scatter(vol_sharpe, ret_sharpe, marker='*', s=200, c='r', label='Max Sharpe')
    ax.scatter(vol_min, ret_min, marker='*', s=200, c='b', label='Min Volatility')
    
    ax.set_title("Efficient Frontier")
    ax.legend()
    plt.show()

def run_backtest(prices, weights, benchmark_weights=None):
    """
    Simulates a buy-and-hold strategy backtest.
    
    Args:
        prices (pd.DataFrame): Daily closing prices for the backtest period.
        weights (dict): Portfolio weights (e.g., {'TSLA': 0.5, 'BND': 0.5}).
        benchmark_weights (dict): Benchmark weights.
        
    Returns:
        pd.DataFrame: Cumulative returns for strategy and benchmark.
    """
    returns = prices.pct_change().dropna()
    
    # Calculate portfolio returns
    # weights is a dict, we need to map it to the columns
    w_vector = [weights.get(col, 0) for col in returns.columns]
    portfolio_daily_ret = returns.dot(w_vector)
    portfolio_cum_ret = (1 + portfolio_daily_ret).cumprod()
    
    result_df = pd.DataFrame({'Strategy': portfolio_cum_ret})
    
    if benchmark_weights:
        b_vector = [benchmark_weights.get(col, 0) for col in returns.columns]
        benchmark_daily_ret = returns.dot(b_vector)
        benchmark_cum_ret = (1 + benchmark_daily_ret).cumprod()
        result_df['Benchmark'] = benchmark_cum_ret
        
    return result_df

def calculate_backtest_metrics(cumulative_returns):
    """
    Calculates Total Return, Annualized Return, Sharpe, Max Drawdown.
    """
    metrics = {}
    
    for col in cumulative_returns.columns:
        series = cumulative_returns[col]
        total_return = series.iloc[-1] - 1
        
        # Annualized Return (assuming daily data)
        days = len(series)
        annualized_return = (1 + total_return) ** (252 / days) - 1
        
        # Sharpe Ratio (calculate from daily returns derived from cum returns)
        daily_rets = series.pct_change().dropna()
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
        
        # Max Drawdown
        rolling_max = series.cummax()
        drawdown = (series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        metrics[col] = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown
        }
        
    return pd.DataFrame(metrics)

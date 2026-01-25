import yfinance as yf
import pandas as pd
import numpy as np

def load_data():
    try:
        data=pd.read_csv('../data/raw/market_data.csv', index_col=0, parse_dates=True)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def fetch_data(tickers, start_date, end_date):
    """
    Fetches historical financial data and handles column naming changes.
    """
    try:
        # We use auto_adjust=True to get adjusted prices in the 'Close' column
        # Or we can keep default and access 'Adj Close' if it exists
        data = yf.download(tickers, start=start_date, end=end_date)
        
        # Check for 'Adj Close' first, fallback to 'Close'
        if 'Adj Close' in data.columns:
            df = data['Adj Close']
        else:
            df = data['Close']
            
        # Ensure result is a DataFrame (even for single ticker)
        if isinstance(df, pd.Series):
            df = df.to_frame()
            
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def clean_data(df):
    """
    Cleans the financial data.
    
    Args:
        df (pd.DataFrame): The raw data DataFrame.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Missing values detected. Filling with forward fill, then backward fill.")
        df = df.ffill().bfill()
        
    # Ensure correct data types (should be float)
    df = df.astype(float)
    
    return df

def calculate_returns(df):
    """
    Calculates daily percentage returns.
    
    Args:
        df (pd.DataFrame): DataFrame with prices.
        
    Returns:
        pd.DataFrame: DataFrame with daily percentage changes.
    """
    returns = df.pct_change().dropna()
    return returns

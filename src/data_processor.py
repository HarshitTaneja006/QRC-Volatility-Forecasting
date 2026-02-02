import yfinance as yf
import numpy as np
import pandas as pd

def download_and_process_data(ticker='SPY', start='2020-01-01', end='2024-01-01'):
    """
    Downloads data and calculates Log Returns.
    """
    print(f"Downloading data for {ticker}...")
    try:
        data = yf.download(ticker, start=start, end=end, multi_level_index=False)
    except TypeError:
        # Fallback for older yfinance versions
        data = yf.download(ticker, start=start, end=end)
    
    # Select the correct column
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    
    # Calculate Log Returns
    # Formula: ln(P_t / P_{t-1})
    data['Log_Returns'] = np.log(data[price_col] / data[price_col].shift(1))
    
    # Drop the first NaN row
    data.dropna(inplace=True)
    
    print(f"Data loaded: {len(data)} trading days.")
    return data

def create_windows(data_series, window_size=5):
    """
    Creates sliding windows for time-series forecasting.
    Input: [r1, r2, r3, r4, r5, r6...]
    Output X: [[r1...r5], [r2...r6]...]
    Output y: [|r6|, |r7|...] (Absolute return is volatility proxy)
    """
    X = []
    y = []
    values = data_series.values
    
    for i in range(len(values) - window_size):
        # Input: The window
        X.append(values[i : i + window_size])
        # Target: The NEXT day's volatility (Absolute Value)
        y.append(np.abs(values[i + window_size]))
        
    return np.array(X), np.array(y)
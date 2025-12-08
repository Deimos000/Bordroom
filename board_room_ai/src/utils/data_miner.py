import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import datetime

def get_math_data(start_date="2000-01-01", end_date="2007-06-01"):
    """
    Fetches mathematical data for the simulation.
    Target: GOOGL
    Competitors: MSFT
    Macro: T10Y2Y, VIXCLS

    Falls back to synthetic data if download fails.
    """
    # 1. Target & Competitors
    tickers = ["GOOGL", "MSFT"]

    print(f"Downloading stock data for {tickers} from {start_date} to {end_date}...")
    stocks = pd.DataFrame()
    try:
        # yfinance download might return MultiIndex or different structure
        # auto_adjust=True is default now, but let's be explicit if needed
        # We try-except the download
        downloaded = yf.download(tickers, start=start_date, end=end_date)
        if not downloaded.empty:
            if 'Adj Close' in downloaded:
                stocks = downloaded['Adj Close']
            elif 'Close' in downloaded:
                stocks = downloaded['Close']
            else:
                # If structure is different (e.g. just columns with tickers)
                stocks = downloaded
    except Exception as e:
        print(f"Error downloading stock data: {e}")

    # 2. Macro Indicators (FRED Database)
    print("Downloading macro data from FRED...")
    macro = pd.DataFrame()
    try:
        macro = web.DataReader(['T10Y2Y', 'VIXCLS'], 'fred', start_date, end_date)
    except Exception as e:
        print(f"Error downloading macro data: {e}")

    # Check if we have data, if not generate synthetic
    if stocks.empty or macro.empty:
        print("Data download failed or incomplete. Generating synthetic data for simulation...")
        return generate_synthetic_data(start_date, end_date)

    # Merge and Clean
    data = pd.concat([stocks, macro], axis=1).ffill()
    return data

def generate_synthetic_data(start_date, end_date):
    """
    Generates synthetic stock and macro data.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(dates)

    # Random walk for stocks
    np.random.seed(42)
    googl_start = 50
    msft_start = 30

    googl = googl_start * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
    msft = msft_start * np.cumprod(1 + np.random.normal(0.0003, 0.015, n))

    # Macro
    # Yield curve: fluctuates around 0.5, dips below 0 sometimes
    t10y2y = 0.5 + np.cumsum(np.random.normal(0, 0.05, n))

    # VIX: Mean reverting around 20
    vix = 20 + np.random.normal(0, 5, n)

    data = pd.DataFrame({
        'GOOGL': googl,
        'MSFT': msft,
        'T10Y2Y': t10y2y,
        'VIXCLS': vix
    }, index=dates)

    return data

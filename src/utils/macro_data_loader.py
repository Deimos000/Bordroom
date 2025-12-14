import os
import requests
import datetime
import pandas as pd
import yfinance as yf
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
OUTPUT_FILE = os.path.join(DATA_DIR, 'global_macro_history.csv')

# Standard Institutional Tickers
TICKERS = {
    "Gold": "GC=F",
    "Copper": "HG=F",
    "Oil": "BZ=F",       # Brent Crude
    "Yield10Y": "^TNX",  # 10 Year Treasury Yield
    "Yield3M": "^IRX",   # 13 Week Treasury Yield
    "VIX": "^VIX"        # Volatility Index
}

# GDELT Keywords to construct the "Global Sentiment" Index
GDELT_QUERY = "(conflict OR war OR military OR crisis OR inflation OR recession)"

class MacroDataLoader:
    """
    Autonomous Data Fetcher.
    Builds the 'World History' dataset for the Doomsday Agent to learn from.
    """
    
    def __init__(self):
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

    def fetch_financials(self, period="5y"):
        """Downloads daily macro data from Yahoo Finance."""
        print(f"📉 Fetching Financial Data ({period})...")
        
        # Download all tickers at once
        data = yf.download(list(TICKERS.values()), period=period, progress=False)['Close']
        
        # Rename columns to friendly names
        # Invert the dictionary to map Ticker -> Name
        inv_map = {v: k for k, v in TICKERS.items()}
        data = data.rename(columns=inv_map)
        
        # Filter only the columns we asked for (handling Yahoo's multi-index if it happens)
        data = data[[TICKERS[name] for name in TICKERS if TICKERS[name] in data.columns]]
        data.columns = [inv_map[col] for col in data.columns]
        
        # Yahoo returns Yields as whole numbers (e.g., 4.5), we want them as is.
        # Handle missing weekends/holidays later
        return data

    def fetch_gdelt_history(self, days=1800):
        """
        Queries GDELT API for the 'Tone' of global conflict news over the last X days.
        """
        print(f"🌍 Fetching GDELT Geopolitical Tone ({days} days)...")
        
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        # We use the TimelineTone mode to get a daily timeseries
        params = {
            "query": GDELT_QUERY,
            "mode": "TimelineTone",
            "format": "json",
            "timespan": f"{days}days"
        }
        
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                print(f"❌ GDELT API Error: {r.status_code}")
                return pd.DataFrame()
                
            data = r.json()
            timeline = data.get('timeline', [{}])[0].get('data', [])
            
            if not timeline:
                print("⚠️ No GDELT data found.")
                return pd.DataFrame()

            # Parse JSON to DataFrame
            # Format: [{'date': '20230101T000000Z', 'value': -2.5}, ...]
            df = pd.DataFrame(timeline)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.rename(columns={'value': 'GDELT_Tone'}, inplace=True)
            
            # GDELT is actually too granular (15 min updates sometimes), 
            # we resample to Daily Average
            df = df.resample('D').mean()
            
            # Normalize GDELT Tone immediately for the neural net
            # Usually ranges -10 to +10. We want roughly -1 to 1.
            df['GDELT_Tone'] = df['GDELT_Tone'] / 10.0
            
            return df
            
        except Exception as e:
            print(f"❌ GDELT Exception: {e}")
            return pd.DataFrame()

    def build_dataset(self):
        """Orchestrates the merge and save."""
        
        # 1. Get Financials
        fin_df = self.fetch_financials(period="5y") # 5 years is ~1260 trading days
        
        # 2. Get Politics (GDELT)
        gdelt_df = self.fetch_gdelt_history(days=365*5)
        
        # 3. Merge
        # We accept that Financials have no data on weekends, but GDELT does.
        # We left join on Financials (Trading Days are what matter for the prediction).
        print("🔗 Merging Datasets...")
        
        # Ensure indices are timezone-naive for merging
        fin_df.index = fin_df.index.tz_localize(None)
        if not gdelt_df.empty:
            gdelt_df.index = gdelt_df.index.tz_localize(None)
        
        full_df = fin_df.join(gdelt_df, how='left')
        
        # 4. Clean
        # Fill missing GDELT days with previous day's tone (News lingers)
        full_df['GDELT_Tone'] = full_df['GDELT_Tone'].fillna(method='ffill')
        
        # Fill any remaining NaNs (holidays)
        full_df = full_df.fillna(method='ffill').fillna(method='bfill')
        
        # 5. Save
        full_df.to_csv(OUTPUT_FILE)
        print(f"✅ DATASET BUILT: {OUTPUT_FILE}")
        print(f"   Shape: {full_df.shape}")
        print(f"   Columns: {list(full_df.columns)}")
        
        return full_df

if __name__ == "__main__":
    # Allow running this file directly to generate the CSV manually
    loader = MacroDataLoader()
    loader.build_dataset()
# src/utils/macro_data_tools.py
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from src.settings import doomsday_config as config

class MacroMiner:
    """
    Forensic Macro Data Miner.
    Fetches raw inputs for the Triangulation Model.
    """

    @staticmethod
    def fetch_gdelt_sentiment(days=7):
        """
        Queries GDELT 2.0 Doc API for average tone matching our keywords.
        Returns a normalized score (-1 to 1).
        """
        # Construct a complex query: (Keyword1 OR Keyword2) AND (Location1 OR Location2)
        keywords = " OR ".join([f'"{k}"' for k in config.GDELT_KEYWORDS])
        locations = " OR ".join([f'"{l}"' for l in config.GDELT_LOCATIONS])
        query = f"({keywords}) ({locations})"
        
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "TimelineTone",
            "format": "json",
            "timespan": f"{days}days"
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            timeline = data.get('timeline', [{}])[0].get('data', [])
            
            if not timeline:
                return 0.0

            # GDELT Tone: -10 (War) to +10 (Peace). We want to normalize to -1 to 1.
            # We invert the logic if necessary based on your preference, 
            # but usually Negative Tone = Risk.
            tones = [float(entry['value']) for entry in timeline]
            avg_tone = np.mean(tones)
            
            # Normalize: Clip between -10 and 10, divide by 10.
            return np.clip(avg_tone / 10.0, -1.0, 1.0)

        except Exception as e:
            print(f"[!] GDELT Fetch Error: {e}")
            return 0.0

    @staticmethod
    def fetch_market_data():
        """
        Fetches Bonds and Commodities to calculate Yield Curve and Gold/Copper Ratio.
        """
        required_tickers = list(config.TICKERS.values())
        
        try:
            # Download last 5 days to get latest closing prices
            df = yf.download(required_tickers, period="5d", progress=False)['Close']
            
            # Extract latest values safely
            data = {
                "gold": df[config.TICKERS["GOLD"]].iloc[-1],
                "copper": df[config.TICKERS["COPPER"]].iloc[-1],
                "oil": df[config.TICKERS["OIL_BRENT"]].iloc[-1],
                "bond_10y": df[config.TICKERS["BOND_10Y"]].iloc[-1],
                "bond_3m": df[config.TICKERS["BOND_3M"]].iloc[-1],
                "vix": df[config.TICKERS["VIX"]].iloc[-1]
            }
            return data
        except Exception as e:
            print(f"[!] Yahoo Finance Error: {e}")
            return None

    @staticmethod
    def normalize_score(value, min_val, max_val, invert=False):
        """
        Standardizes any number into a -1 (Bad) to 1 (Good) range.
        If invert=True, Higher Value = Bad Score (e.g. VIX, Gold/Copper Ratio).
        """
        # 1. Normalize to 0-1
        norm = (value - min_val) / (max_val - min_val)
        # 2. Shift to -1 to 1
        score = (norm * 2) - 1
        # 3. Clip
        score = np.clip(score, -1.0, 1.0)
        
        if invert:
            return score * -1
        return score
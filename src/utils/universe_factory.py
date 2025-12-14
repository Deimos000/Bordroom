# src/utils/universe_factory.py
import pandas as pd
import requests
import io
import ssl

# --- 1. SSL Certificate Fix (Bypasses verification errors) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

class UniverseFactory:
    """
    Fetches stock tickers from Wikipedia.
    Includes anti-blocking headers to avoid HTTP 403 Forbidden errors.
    """

    @staticmethod
    def _get_wiki_table(url, table_index, symbol_col):
        try:
            # Fake a browser visit to avoid being blocked by Wikipedia
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            # 1. Fetch raw HTML
            response = requests.get(url, headers=headers)
            response.raise_for_status() # Raise error if status is 4xx or 5xx
            
            # 2. Parse with Pandas
            # We use io.StringIO because pd.read_html sometimes struggles with raw strings in newer versions
            df = pd.read_html(io.StringIO(response.text))[table_index]
            
            # 3. Clean Tickers
            tickers = df[symbol_col].tolist()
            
            # Replace dot with dash (e.g., BRK.B -> BRK-B) for Yahoo Finance compatibility
            clean_tickers = [str(t).replace('.', '-') for t in tickers]
            
            return clean_tickers

        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            # Return a small fallback list so the program doesn't crash completely
            return ["NVDA", "AMD", "MSFT", "AAPL", "GOOGL", "AMZN", "TSLA", "META", "JPM", "BAC"]

    @staticmethod
    def get_sp500(): 
        print("🌌 Scanning S&P 500...")
        return UniverseFactory._get_wiki_table(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0, 'Symbol')

    @staticmethod
    def get_sp400_midcap():
        print("🌌 Scanning S&P 400 (Mid Cap)...")
        return UniverseFactory._get_wiki_table(
            'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', 0, 'Symbol')

    @staticmethod
    def get_sp600_smallcap():
        print("🌌 Scanning S&P 600 (Small Cap)...")
        return UniverseFactory._get_wiki_table(
            'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', 0, 'Symbol')

    @staticmethod
    def get_massive_universe():
        """Returns ~1500 liquid US stocks (S&P 1500)"""
        u = UniverseFactory.get_sp500() + \
            UniverseFactory.get_sp400_midcap() + \
            UniverseFactory.get_sp600_smallcap()
        
        # Deduplicate list
        return list(set(u))
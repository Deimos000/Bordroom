import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import time
import random
import warnings

# Suppress yfinance warnings about "No data found" for specific tickers
warnings.filterwarnings("ignore")

class DataMiner:
    """
    LEVEL 7 DATA ENGINE (UPDATED)
    Responsible for fetching massive datasets (Price + Text + Fundamentals)
    efficiently using parallelism while respecting API limits.
    """

    @staticmethod
    def get_universe_data(tickers, start_date, end_date, chunk_size=100):
        """
        Fetches daily price data (OHLCV) for a large list of tickers.
        Uses smaller batching and sleep intervals to avoid API timeouts on massive universes.
        
        Args:
            chunk_size (int): Reduced to 100 to prevent 'Too Many Requests' errors.
        """
        print(f"⛏️  Bulk Mining Price Data for {len(tickers)} assets...")
        
        full_data = {}
        
        # 1. Process in Chunks
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            batch_num = (i // chunk_size) + 1
            total_batches = (len(tickers) // chunk_size) + 1
            
            print(f"   ... Processing Batch {batch_num}/{total_batches} ({len(chunk)} tickers)")
            
            try:
                # 'threads=True' uses yfinance's internal threading for prices
                # 'repair=True' attempts to fix missing 100x splits or data errors
                data = yf.download(
                    chunk, 
                    start=start_date, 
                    end=end_date, 
                    group_by='ticker', 
                    auto_adjust=True, 
                    progress=False, 
                    threads=True
                )
                
                # 2. Parse the result
                if data.empty:
                    print(f"   ⚠️ Batch {batch_num} returned empty.")
                    continue

                # Case A: Single Ticker in Chunk (returns flat DataFrame)
                if len(chunk) == 1:
                    t = chunk[0]
                    # Check if data is valid (sometimes returns empty DF with columns)
                    if len(data) > 50:
                        # Clean: Forward fill gaps, drop remaining NaNs
                        df = data.ffill().dropna()
                        full_data[t] = df

                # Case B: Multiple Tickers (returns MultiIndex DataFrame)
                else:
                    # The columns level 0 contains the Tickers
                    # valid_cols are the tickers that actually returned data
                    valid_cols = data.columns.get_level_values(0).unique()
                    
                    for t in chunk:
                        if t in valid_cols:
                            try:
                                # Extract specific ticker dataframe
                                df = data[t].copy()
                                
                                # Data Quality Check
                                # 1. Check length
                                if len(df) < 50: continue
                                
                                # 2. Check for missing columns
                                required_cols = ['Close', 'Volume', 'High', 'Low']
                                if not all(col in df.columns for col in required_cols): continue

                                # 3. Clean
                                df = df.ffill().dropna()
                                full_data[t] = df
                            except Exception:
                                continue

                # 3. Rate Limit Protection (Cool Down)
                # Vital when scraping 1000+ stocks
                print(f"   ... Batch {batch_num} done. Cooling down...")
                time.sleep(2.0) 

            except Exception as e:
                print(f"   ⚠️ Batch {batch_num} Critical Error: {e}")
                time.sleep(5) # Longer cool down on error

        print(f"✅ Prices Secured for {len(full_data)}/{len(tickers)} tickers.")
        return full_data

    @staticmethod
    def _fetch_single_metadata(ticker):
        """
        Worker function for ThreadPoolExecutor.
        Fetches static info for ONE ticker.
        """
        try:
            # Jitter: Random sleep 0.1-0.5s to prevent 20 threads hitting Yahoo simultaneously
            time.sleep(random.uniform(0.1, 0.5))
            
            # Ticker object
            obj = yf.Ticker(ticker)
            
            # Accessing .info triggers the API request
            info = obj.info
            
            # 1. Get Text (Business Description)
            # Use 'longBusinessSummary' (detailed) or 'sector' (fallback)
            desc = info.get('longBusinessSummary', "")
            if not desc:
                desc = info.get('shortBusinessSummary', "")
            
            # If absolutely no text, we can't build a semantic graph node
            if not desc or len(desc) < 20:
                return None

            # 2. Get Fundamentals (Node Features)
            # We use .get(key, 0) to ensure no crashes on missing data
            pe = info.get('trailingPE', 0) or 0
            pb = info.get('priceToBook', 0) or 0
            debt = info.get('debtToEquity', 0) or 0
            margins = info.get('profitMargins', 0) or 0
            
            # Market Cap (Log Scale to normalize huge numbers like Apple vs Small Caps)
            # Default to 1B (1e9) if missing to avoid log(0) errors
            mkt_cap = info.get('marketCap', 1e9)
            if mkt_cap is None: mkt_cap = 1e9
            log_cap = np.log(float(mkt_cap))
            
            beta = info.get('beta', 1.0)
            if beta is None: beta = 1.0

            # Feature Vector: [PE, PB, Debt, Margins, LogCap, Beta]
            fund_vec = [pe, pb, debt, margins, log_cap, beta]
            
            return (ticker, fund_vec, desc)

        except Exception:
            # Silently fail for this specific ticker so the loop continues
            return None

    @staticmethod
    def get_fundamentals_and_text(tickers):
        """
        Fetches 10-K summaries and Fundamental Ratios.
        USES MULTI-THREADING to speed up the process 20x.
        """
        print(f"📜 Parallel Mining Intelligence (Text & Fundamentals) for {len(tickers)} assets...")
        
        valid_tickers = []
        fundamental_map = {}
        descriptions = []
        
        # 1. Configure Thread Pool
        # max_workers=15 is safer than 20 for massive batches to avoid 429 Too Many Requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            # Submit all tasks
            future_to_ticker = {executor.submit(DataMiner._fetch_single_metadata, t): t for t in tickers}
            
            # Process as they complete
            completed_count = 0
            total_count = len(tickers)
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                result = future.result()
                completed_count += 1
                
                # Progress Bar
                if completed_count % 50 == 0:
                    print(f"   ... Parsed {completed_count}/{total_count} companies")

                if result:
                    t, vec, desc = result
                    valid_tickers.append(t)
                    fundamental_map[t] = vec
                    descriptions.append(desc)

        print(f"✅ Intelligence Secured for {len(valid_tickers)} companies.")
        print(f"   (Dropped {len(tickers) - len(valid_tickers)} tickers due to missing data)")
        
        return valid_tickers, fundamental_map, descriptions
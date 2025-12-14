import time
import json
import random
import logging
import threading
import queue
import re
import requests
import sqlite3
from datetime import datetime
from bs4 import BeautifulSoup

# --- Third-party imports ---
try:
    import trafilatura
    import yfinance as yf
    from duckduckgo_search import DDGS
    from rapidfuzz import process, fuzz
except ImportError:
    print("❌ Missing Dependencies. Run: pip install trafilatura yfinance duckduckgo-search rapidfuzz")
    raise

# --- Custom Modules ---
# Adjust these imports based on your specific folder structure
try:
    from src.utils.atlas_db import AtlasDB
    from src.utils.vector_memory import VectorMemory
    from src.utils.local_llm import LocalLLM
except ImportError:
    from atlas_db import AtlasDB
    from vector_memory import VectorMemory
    from local_llm import LocalLLM

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ForensicAgent")

# =========================================================================
# HELPER 1: SEC FETCHING LOGIC (The "Source of Truth")
# =========================================================================
class SECFetcher:
    """
    Fetches official 10-K filings directly from the SEC.
    """
    def __init__(self):
        # SEC requires a contact email in User-Agent
        self.headers = {'User-Agent': 'MarketForensicsBot/2.0 (admin@example.com)'}
        self.cik_cache = {}

    def get_10k_text(self, ticker):
        """Downloads and extracts Item 1 (Business) & Item 1A (Risk) from latest 10-K."""
        try:
            # 1. Get CIK (Central Index Key)
            cik = self._get_cik(ticker)
            if not cik: return None

            # 2. Find Latest 10-K Metadata
            time.sleep(0.2) # Rate limit politeness
            url_meta = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = requests.get(url_meta, headers=self.headers).json()
            
            recent = resp['filings']['recent']
            accession, primary_doc = None, None
            
            for i, form in enumerate(recent['form']):
                if form == '10-K':
                    accession = recent['accessionNumber'][i]
                    primary_doc = recent['primaryDocument'][i]
                    break
            
            if not accession: return None
            
            # 3. Construct URL and Download
            folder = accession.replace("-", "")
            final_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{primary_doc}"
            
            r = requests.get(final_url, headers=self.headers)
            
            # 4. Extract Relevant Sections (Business + Risk)
            # We grab a large chunk around "Item 1" and let the Scout filter it later.
            soup = BeautifulSoup(r.content, 'lxml')
            text = soup.get_text(" ", strip=True)
            
            # Heuristic: Find "Item 1. Business" and "Item 2. Properties"
            lower_text = text.lower()
            start = lower_text.find("item 1. business")
            if start == -1: start = 0
            
            end = lower_text.find("item 2. properties")
            if end == -1: end = start + 100000 # Fallback to grabbing 100k chars
            
            clean_text = text[start:end]
            return f"SOURCE: OFFICIAL SEC 10-K FILING\n\n{clean_text}"

        except Exception as e:
            logger.warning(f"SEC Fetch failed for {ticker}: {e}")
            return None

    def _get_cik(self, ticker):
        if ticker in self.cik_cache: return self.cik_cache[ticker]
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            data = requests.get(url, headers=self.headers).json()
            for entry in data.values():
                if entry['ticker'] == ticker.upper():
                    cik = str(entry['cik_str']).zfill(10)
                    self.cik_cache[ticker] = cik
                    return cik
        except:
            pass
        return None

# =========================================================================
# HELPER 2: ENTITY RESOLVER (Fuzzy Matching)
# =========================================================================
class EntityResolver:
    """
    Resolves "Hon Hai Precision" -> "2317.TW" using Fuzzy Matching & DB Cache.
    """
    def __init__(self, db_instance):
        self.db = db_instance
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # Fuzzy Matching Memory
        self.fuzzy_lookup = {} 
        self.last_fuzzy_refresh = 0
        
        self.manual_overrides = {
            "FACEBOOK": "META", "META PLATFORMS": "META",
            "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
            "TSMC": "TSM", "FOXCONN": "2317.TW",
            "SUPPLIER": None, "CUSTOMER": None # Stop words
        }

    def _refresh_fuzzy_cache(self):
        """Loads all known companies from DB into RAM for fast fuzzy matching."""
        now = time.time()
        if self.fuzzy_lookup and (now - self.last_fuzzy_refresh < 600):
            return

        with self.cache_lock:
            try:
                conn = sqlite3.connect(self.db.db_path) 
                cursor = conn.cursor()
                cursor.execute("SELECT ticker, name, aliases FROM assets")
                rows = cursor.fetchall()
                conn.close()

                temp_lookup = {}
                for ticker, name, aliases_json in rows:
                    if name: temp_lookup[name.upper()] = ticker
                    if aliases_json:
                        try:
                            for a in json.loads(aliases_json):
                                temp_lookup[a.upper()] = ticker
                        except: pass
                
                self.fuzzy_lookup = temp_lookup
                self.last_fuzzy_refresh = now
            except Exception as e:
                logger.error(f"Fuzzy Cache Build Error: {e}")

    def resolve(self, name):
        if not name or len(name) < 2: return None
        clean_name = name.strip().upper()
        
        if clean_name in self.manual_overrides: return self.manual_overrides[clean_name]

        # 1. Exact Cache
        with self.cache_lock:
            if clean_name in self.cache: return self.cache[clean_name]

        # 2. Database Exact
        db_res = self.db.resolve_entity(clean_name)
        if db_res:
            self._update_cache(clean_name, db_res)
            return db_res

        # 3. Fuzzy Match (The Magic)
        self._refresh_fuzzy_cache() 
        match = process.extractOne(clean_name, self.fuzzy_lookup.keys(), scorer=fuzz.token_sort_ratio)
        if match:
            best_name, score, _ = match
            if score >= 90.0:
                ticker = self.fuzzy_lookup[best_name]
                self._update_cache(clean_name, ticker)
                return ticker

        # 4. Web Search Fallback
        candidate = self._search_web_for_ticker(name)
        if candidate:
            self.db.save_asset(candidate, name=name, aliases=[name])
            self._update_cache(clean_name, candidate)
            return candidate
        
        self._update_cache(clean_name, None)
        return None

    def _update_cache(self, name, ticker):
        with self.cache_lock:
            self.cache[name] = ticker

    def _search_web_for_ticker(self, company_name):
        """Asks DuckDuckGo for the ticker."""
        try:
            query = f"stock ticker symbol for {company_name}"
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=1))
                if results:
                    match = re.search(r'\b([A-Z]{2,5})\b', results[0]['body']) 
                    if match: return match.group(1)
        except:
            pass
        return None

# =========================================================================
# MAIN CLASS: FORENSIC AGENT
# =========================================================================
class ForensicAgent:
    def __init__(self, max_workers=3):
        logger.info("🕵️‍♂️ Initializing Scout & Sniper Forensic Agent...")

        # --- Architecture Components ---
        self.db = AtlasDB()            
        self.memory = VectorMemory()   
        self.brain = LocalLLM() # This handles the model swapping
        self.resolver = EntityResolver(self.db)
        self.sec_fetcher = SECFetcher()

        # --- State Management ---
        self.state_lock = threading.Lock()
        self.processed_tickers = set()
        self.processed_urls = set()
        
        # --- Task Pipelines ---
        self.task_queue = queue.PriorityQueue() # (Priority, Depth, Ticker)
        self.scrape_queue = queue.Queue(maxsize=50) # (URL, Ticker, Depth)
        self.analysis_queue = queue.Queue(maxsize=20) # (Text, Metadata...)
        
        self.stop_event = threading.Event()
        self.num_scraper_threads = max_workers

    def start_investigation(self, start_ticker, max_depth=2):
        """
        Main entry point. Starts the machine.
        """
        self.max_depth = max_depth
        start_ticker = start_ticker.upper()
        
        # Priority 0 = Highest
        self.task_queue.put((0, 0, start_ticker))
        
        threads = []
        
        # 1. Manager (Strategist)
        t_mgr = threading.Thread(target=self._manager_worker, name="Manager")
        t_mgr.start()
        threads.append(t_mgr)

        # 2. Scrapers (Harvesters)
        for i in range(self.num_scraper_threads):
            t_scr = threading.Thread(target=self._scraper_worker, name=f"Scraper-{i}")
            t_scr.start()
            threads.append(t_scr)

        # 3. Analyst (Scout & Sniper)
        t_ana = threading.Thread(target=self._analyst_worker, name="Analyst")
        t_ana.start()
        threads.append(t_ana)

        print(f"\n🚀 PIPELINE ACTIVE. Investigating {start_ticker} (Max Depth: {max_depth})...\n")

        try:
            # Monitor loop
            while not self.stop_event.is_set():
                q1 = self.task_queue.qsize()
                q2 = self.scrape_queue.qsize()
                q3 = self.analysis_queue.qsize()
                
                # If everything is empty and threads are idle, we are done.
                if q1 == 0 and q2 == 0 and q3 == 0:
                    time.sleep(5)
                    if self.task_queue.empty() and self.scrape_queue.empty() and self.analysis_queue.empty():
                        print("✅ Investigation Complete.")
                        self.stop_event.set()
                        break
                time.sleep(2)
        except KeyboardInterrupt:
            print("\n🛑 Stop Signal Received.")
            self.stop_event.set()

        for t in threads:
            t.join(timeout=2.0)
        
        self._generate_report(start_ticker)

    # ==========================
    # WORKER 1: MANAGER (Strategy)
    # ==========================
    def _manager_worker(self):
        """Decides what to fetch (SEC vs Web) based on queue."""
        while not self.stop_event.is_set():
            try:
                priority, depth, ticker = self.task_queue.get(timeout=2)
            except queue.Empty:
                continue

            with self.state_lock:
                if ticker in self.processed_tickers:
                    self.task_queue.task_done()
                    continue
                self.processed_tickers.add(ticker)

            logger.info(f"🔍 Analyzing: {ticker} (Depth {depth})")
            
            # PHASE 1: Official SEC Data (Depth 0 & 1 only)
            if depth <= 1:
                logger.info(f"📜 Retrieving 10-K for {ticker}")
                sec_text = self.sec_fetcher.get_10k_text(ticker)
                if sec_text:
                    self.analysis_queue.put({
                        "text": sec_text,
                        "url": "SEC_EDGAR",
                        "ticker": ticker,
                        "depth": depth,
                        "is_sec": True
                    })
            
            # PHASE 2: Web Reconnaissance
            queries = [
                f"{ticker} suppliers list 2024",
                f"{ticker} biggest customers revenue",
                f"{ticker} regulatory fines lawsuit",
                f"{ticker} supply chain shortages"
            ]
            
            found_urls = set()
            try:
                with DDGS() as ddgs:
                    for q in queries:
                        if self.stop_event.is_set(): break
                        time.sleep(random.uniform(1.5, 3.0)) # Anti-ban sleep
                        results = list(ddgs.text(q, max_results=2))
                        for r in results:
                            url = r.get('href')
                            if url and url not in self.processed_urls:
                                found_urls.add(url)
                                with self.state_lock:
                                    self.processed_urls.add(url)
            except Exception as e:
                logger.warning(f"Search failed: {e}")

            for url in found_urls:
                self.scrape_queue.put((url, ticker, depth))
            
            self.task_queue.task_done()

    # ==========================
    # WORKER 2: SCRAPER (Harvest)
    # ==========================
    def _scraper_worker(self):
        """Downloads raw HTML and extracts main text."""
        while not self.stop_event.is_set():
            try:
                url, ticker, depth = self.scrape_queue.get(timeout=2)
            except queue.Empty:
                continue

            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(downloaded)
                    if text and len(text) > 300:
                        self.analysis_queue.put({
                            "text": text,
                            "url": url,
                            "ticker": ticker,
                            "depth": depth,
                            "is_sec": False
                        })
            except Exception as e:
                logger.error(f"Scrape error {url}: {e}")
            finally:
                self.scrape_queue.task_done()

    # ==========================
    # WORKER 3: ANALYST (Scout & Sniper)
    # ==========================
    def _analyst_worker(self):
        """
        The Intelligent Engine.
        1. RAG Memory Storage.
        2. SCOUT (Phi-3.5): Filters noise.
        3. SNIPER (Qwen-2.5): Extracts JSON.
        4. CRITIC (Qwen-2.5): Verifies facts.
        """
        while not self.stop_event.is_set():
            try:
                data = self.analysis_queue.get(timeout=5)
            except queue.Empty:
                continue

            ticker = data['ticker']
            raw_text = data['text']
            url = data['url']
            depth = data['depth']
            is_sec = data.get('is_sec', False)

            # 1. RAG Storage
            category = "SEC_10K" if is_sec else "news"
            self.memory.memorize(ticker, url, raw_text[:8000], category=category)

            # 2. Chunking (Split massive text into 6k char blocks)
            chunks = self._create_sliding_windows(raw_text, window_size=6000)
            new_leads = set()

            for chunk in chunks:
                if self.stop_event.is_set(): break
                
                # --- STEP A: THE SCOUT (Filter Noise) ---
                # This uses the fast Phi-3.5 model to discard useless chunks
                high_signal_text = self.brain.scout(chunk)
                
                if not high_signal_text: 
                    continue # Skip empty chunks

                # --- STEP B: THE SNIPER (Extract Data) ---
                # This swaps to Qwen 2.5 automatically
                relationships = self.brain.extract_relationships(high_signal_text, ticker)
                
                for item in relationships:
                    target_name = item.get('target_entity')
                    if not target_name: continue
                    
                    # Resolve Name -> Ticker
                    target_ticker = self.resolver.resolve(target_name)
                    
                    if target_ticker and target_ticker != ticker:
                        
                        # --- STEP C: THE CRITIC (Verification) ---
                        # If it's a web rumor (not SEC), double-check it.
                        pass_verification = True
                        if not is_sec:
                            if item.get('type') == 'lawsuit' or item.get('sentiment', 0) < -0.4:
                                pass_verification = self.brain.validate_claim(
                                    claim=f"{target_ticker} has a lawsuit/issue with {ticker}",
                                    context=high_signal_text
                                )
                        
                        if not pass_verification:
                            logger.info(f"🚫 CRITIC: Rejected rumor {target_ticker} vs {ticker}")
                            continue

                        # Save to Knowledge Graph
                        conf = 0.99 if is_sec else item.get('confidence', 0.5)
                        strength = 0.9 if is_sec else item.get('strength', 0.5)

                        self.db.save_relationship(
                            source=ticker,
                            target=target_ticker,
                            rel_type=item.get('type', 'related'),
                            reason=item.get('details', 'N/A'),
                            strength=strength, 
                            sentiment=item.get('sentiment', 0.0),
                            confidence=conf,
                            source_url=url
                        )
                        self.db.save_asset(target_ticker, aliases=item.get('aliases', []))
                        
                        new_leads.add(target_ticker)

            # 3. Recursive Discovery
            if depth < self.max_depth:
                for lead in new_leads:
                    with self.state_lock:
                        if lead not in self.processed_tickers:
                            self.task_queue.put((1, depth + 1, lead))

            self.analysis_queue.task_done()

    def _create_sliding_windows(self, text, window_size=5000, overlap=500):
        if len(text) <= window_size: return [text]
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + window_size])
            start += (window_size - overlap)
        return chunks

    def _generate_report(self, root_ticker):
        print(f"\n📊 --- FORENSIC REPORT: {root_ticker} ---")
        try:
            G = self.db.get_market_graph(min_impact=0.0)
            print(f"Nodes in Graph: {len(G.nodes)}")
            
            if root_ticker in G:
                neighbors = sorted(
                    G[root_ticker].items(), 
                    key=lambda x: x[1].get('weight', 0), 
                    reverse=True
                )[:5]
                print("\nTop Active Relationships (Impact > 0.1):")
                for n, attr in neighbors:
                    print(f"  -> {n} ({attr.get('relation')}): Impact={attr.get('weight'):.3f}")

                print(f"\n📉 Simulating -15% Supply Shock to {root_ticker}...")
                impact = self.db.simulate_shockwave(root_ticker, shock_percent=-0.15)
                if not impact.empty:
                    print(impact.head(5).to_string(index=False))
                else:
                    print("No significant downstream impact detected.")
        except Exception as e:
            print(f"Report generation failed: {e}")

# =========================================================================
# ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    import os
    if not os.path.exists("data"): os.makedirs("data")
        
    # Initialize and Run
    agent = ForensicAgent(max_workers=3)
    agent.start_investigation("NVDA", max_depth=1)
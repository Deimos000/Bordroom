import requests
import re
import time
import logging
import json
from bs4 import BeautifulSoup
from datetime import datetime

# --- Custom Modules ---
try:
    from src.utils.atlas_db import AtlasDB
    from src.utils.vector_memory import VectorMemory
    from src.utils.local_llm import LocalLLM
except ImportError:
    # Fallback for flat directory structures
    from atlas_db import AtlasDB
    from vector_memory import VectorMemory
    from local_llm import LocalLLM

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SEC_AGENT] %(levelname)s - %(message)s'
)
logger = logging.getLogger("SECForensicAgent")

class SECForensicAgent:
    def __init__(self):
        """
        Specialized Agent for extracting 'Sworn Testimony' (10-K Filings).
        Uses 'Scout & Sniper' architecture to process large documents efficiently.
        """
        logger.info("🏛️ Initializing SEC Forensic Agent (Scout & Sniper Enabled)...")
        
        # Connect to Shared Brain components
        self.db = AtlasDB()
        self.memory = VectorMemory()
        self.brain = LocalLLM()
        
        # --- SEC COMPLIANCE ---
        # IMPORTANT: Replace 'research_account@example.com' with your actual email.
        # The SEC requires a User-Agent in the format: "AppName/Version (Email)"
        self.headers = {
            'User-Agent': 'StockPredictorBot/2.5 (research_account@example.com)',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        # Cache for CIK Lookups
        self.cik_map = {}
        self._load_ticker_map()

        # Compile Regex patterns once for efficiency
        # Handles: "Item 1.", "ITEM 1", "Item 1. Business", "Item 1 \n Business"
        self.pat_business = re.compile(r'item\s+1\.?\s*business', re.IGNORECASE)
        self.pat_risk = re.compile(r'item\s+1a\.?\s*risk\s+factors', re.IGNORECASE)
        self.pat_properties = re.compile(r'item\s+2\.?\s*properties', re.IGNORECASE)

    def _load_ticker_map(self):
        """Downloads the official Ticker -> CIK map from SEC."""
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 200:
                data = resp.json()
                for entry in data.values():
                    # CIK must be 10 digits (padded with zeros)
                    self.cik_map[entry['ticker']] = str(entry['cik_str']).zfill(10)
                logger.info(f"✅ Loaded {len(self.cik_map)} tickers from SEC.")
            else:
                logger.error("Failed to load SEC ticker map.")
        except Exception as e:
            logger.error(f"Error loading ticker map: {e}")

    def get_latest_10k_metadata(self, ticker):
        """
        Finds the URL and DATE of the most recent 10-K filing.
        Returns: (url, report_date_str)
        """
        ticker = ticker.upper()
        cik = self.cik_map.get(ticker)
        
        if not cik:
            logger.warning(f"❌ CIK not found for {ticker}")
            return None, None

        try:
            # Rate limit politeness (SEC limit is 10 requests/sec max)
            time.sleep(0.15)
            
            # Fetch submission history
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = requests.get(url, headers=self.headers)
            data = resp.json()
            
            recent = data['filings']['recent']
            
            # Iterate to find latest 10-K
            for i, form in enumerate(recent['form']):
                if form == '10-K':
                    accession = recent['accessionNumber'][i]
                    primary_doc = recent['primaryDocument'][i]
                    report_date = recent['reportDate'][i] # YYYY-MM-DD
                    
                    # Construct URL: 000123-45-6789 -> 000123456789
                    folder = accession.replace("-", "")
                    filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{primary_doc}"
                    
                    logger.info(f"📄 Found 10-K for {ticker} (Date: {report_date})")
                    return filing_url, report_date
            
            logger.warning(f"No 10-K found in recent filings for {ticker}.")
            return None, None

        except Exception as e:
            logger.error(f"Error fetching 10-K metadata for {ticker}: {e}")
            return None, None

    def _extract_relevant_sections(self, html_content):
        """
        Robust Regex parsing to grab 'Item 1. Business' and 'Item 1A. Risk Factors'.
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Get text with strict separators to help regex find boundaries
            text = soup.get_text("\n", strip=True)
            
            # 1. Find Start Indices using Regex
            # We search the first 200k chars for headers to save time
            header_search_area = text[:200000]
            
            match_bus = self.pat_business.search(header_search_area)
            match_risk = self.pat_risk.search(header_search_area)
            match_prop = self.pat_properties.search(header_search_area)
            
            start_idx = 0
            end_idx = len(text)

            # Heuristic: Prefer "Business" start, fallback to 0
            if match_bus:
                start_idx = match_bus.start()
            elif match_risk:
                start_idx = match_risk.start()
            
            # Heuristic: Stop at "Properties", fallback to a generic length
            if match_prop:
                end_idx = match_prop.start()
            else:
                # If no end found, take 50k chars from start
                end_idx = start_idx + 50000

            relevant_text = text[start_idx:end_idx]
            
            # Sanity check: If regex failed completely, return raw chunk
            if len(relevant_text) < 1000:
                logger.warning("⚠️ Regex parsed too little text. Reverting to naive slice.")
                return text[:50000]

            return relevant_text

        except Exception as e:
            logger.error(f"Parsing error: {e}. Returning raw text slice.")
            return str(html_content)[:50000]

    def analyze_filing(self, ticker):
        """
        Main orchestration function.
        1. Downloads 10-K.
        2. Cleans Text via Regex.
        3. Saves RAW text to Vector Memory (for future RAG/Fact checking).
        4. Uses SCOUT (Phi-3.5) to find high-signal paragraphs.
        5. Uses SNIPER (Qwen/Llama) to extract relationships from Scout output.
        """
        url, date = self.get_latest_10k_metadata(ticker)
        if not url: return

        try:
            logger.info(f"⬇️ Downloading 10-K for {ticker}...")
            resp = requests.get(url, headers=self.headers)
            
            # 1. Parse Text
            clean_text = self._extract_relevant_sections(resp.content)
            logger.info(f"✅ Extracted {len(clean_text)} chars of analysis text.")
            
            # 2. Chunking (Standard sliding window)
            window_size = 6000 
            overlap = 500
            chunks = []
            start = 0
            while start < len(clean_text):
                chunks.append(clean_text[start:start+window_size])
                start += (window_size - overlap)
            
            logger.info(f"🧠 Processing {len(chunks)} chunks using Scout & Sniper...")
            
            for i, chunk in enumerate(chunks):
                
                # --- A. Vector Memory (RAG) ---
                # We save the raw chunk so the "Bullshit Checker" has context later.
                # Even if the Scout filters it out for the graph, we might need it for a specific query later.
                self.memory.memorize(
                    ticker=ticker,
                    source_url=url,
                    text=chunk,
                    chunk_index=i,
                    category="SEC_10K"
                )
                
                # --- B. The Scout (Phase 1) ---
                # Fast model (Phi-3.5) filters noise.
                # If this returns None, the chunk is irrelevant.
                scouted_text = self.brain.scout(chunk)
                
                if not scouted_text:
                    logger.debug(f"💤 Chunk {i}: No signal found by Scout. Skipping.")
                    continue
                
                # --- C. The Sniper (Phase 2) ---
                # Smart model (Qwen/Llama) extracts JSON from the filtered text.
                # We add a preamble to tell the Sniper this is official 10-K data.
                formatted_input = f"SOURCE: OFFICIAL SEC 10-K FILING ({date}).\n\n{scouted_text}"
                
                relationships = self.brain.extract_relationships(formatted_input, ticker)
                
                for item in relationships:
                    target = item.get('target_entity')
                    if not target: continue
                    
                    # Logic: If it's in a 10-K, confidence is inherently high.
                    # We override the LLM's confidence because this is sworn testimony.
                    confidence = 0.95
                    structural_strength = 0.9
                    
                    logger.info(f"🔗 SEC FOUND: {ticker} -> {item['type']} -> {target}")
                    
                    # Save to Graph DB
                    self.db.save_relationship(
                        source=ticker,
                        target=target,
                        rel_type=item.get('type'),
                        reason=f"SEC 10-K ({date}): {item.get('details', item.get('reason'))}",
                        strength=structural_strength, 
                        sentiment=item.get('sentiment', 0),
                        confidence=confidence,
                        source_url=url
                    )
                    
                    # Ensure node exists in Assets table
                    self.db.save_asset(target, aliases=item.get('aliases', []))

            logger.info(f"🎉 Analysis complete for {ticker}. Knowledge Graph updated.")

        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")

# --- Execution ---
if __name__ == "__main__":
    # Ensure data directory exists
    import os
    if not os.path.exists("data"):
        os.makedirs("data")

    agent = SECForensicAgent()
    
    # Example: Analyze a specific ticker
    target_ticker = "NVDA" 
    
    print(f"\n🚀 Starting SEC Forensic Run for {target_ticker}")
    agent.analyze_filing(target_ticker)
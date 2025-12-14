import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# Project Imports
try:
    from src.utils.atlas_db import AtlasDB
    from src.utils.universe_factory import UniverseFactory
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.utils.atlas_db import AtlasDB
    from src.utils.universe_factory import UniverseFactory

class RelationAgent:
    def __init__(self):
        self.db = AtlasDB()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    def _fetch_info(self, ticker):
        """Downloads company profile/bio."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'summary': info.get('longBusinessSummary', '')
            }
        except Exception as e:
            print(f"⚠️ Could not fetch info for {ticker}: {e}")
            return None

    def _extract_reason(self, text_a, text_b, feature_names, top_n=5):
        """
        Finds the specific keywords shared between two descriptions 
        to construct a reason sentence.
        """
        # Simple intersection logic for explanation
        # (A real production system might use LLMs here, but this is fast/free)
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        # Intersection of words that are also in our TF-IDF vocabulary (important words)
        # Note: This is a simplified "why". 
        common = words_a.intersection(words_b)
        
        # Filter for meaningful words (len > 4) to avoid "the", "and", "company"
        meaningful = [w for w in common if len(w) > 4 and w in feature_names]
        
        if not meaningful:
            return "Statistically similar business descriptions."
            
        joined_keywords = ", ".join(list(meaningful)[:top_n])
        return f"High semantic overlap in business areas: {joined_keywords}."

    def build_graph_for_ticker(self, target_ticker, universe_tickers=None):
        """
        1. Fetches data for Target.
        2. Compares Target against the Universe.
        3. Saves connections with REASONS to SQL.
        """
        print(f"🕵️  RelationAgent: Mapping universe for {target_ticker}...")
        
        # 1. Ensure Target is in DB
        target_info = self._fetch_info(target_ticker)
        if not target_info or not target_info['summary']:
            print("❌ Target has no business summary. Cannot map.")
            return
            
        self.db.save_asset(target_info['ticker'], target_info['name'], 
                           target_info['sector'], target_info['summary'])

        # 2. Populate Universe (If not in DB yet)
        # If universe_tickers is None, we default to S&P 500 from your factory
        if universe_tickers is None:
            universe_tickers = UniverseFactory.get_sp500()[:100] # Limit to 100 for speed testing
        
        # Check which ones we already have to save time
        existing_df = self.db.get_all_descriptions()
        existing_tickers = set(existing_df['ticker'].tolist())
        
        new_docs = []
        new_tickers = []
        
        # Fetch missing tickers
        for t in universe_tickers:
            if t not in existing_tickers and t != target_ticker:
                info = self._fetch_info(t)
                if info and info['summary']:
                    self.db.save_asset(info['ticker'], info['name'], info['sector'], info['summary'])
                    new_docs.append(info['summary'])
                    new_tickers.append(t)
        
        # 3. Reload Full Dataset for NLP
        full_df = self.db.get_all_descriptions()
        if full_df.empty: return

        # Calculate Similarity
        print("🧠  Calculating semantic relationships...")
        tfidf_matrix = self.vectorizer.fit_transform(full_df['business_summary'])
        feature_names = set(self.vectorizer.get_feature_names_out())
        
        # Find index of target
        try:
            target_idx = full_df[full_df['ticker'] == target_ticker].index[0]
        except IndexError:
            return

        # Cosine Similarity of Target vs All
        cosine_sim = cosine_similarity(tfidf_matrix[target_idx:target_idx+1], tfidf_matrix).flatten()
        
        # 4. Extract Connections & Reasons
        related_indices = cosine_sim.argsort()[::-1] # Sort desc
        
        count = 0
        for idx in related_indices:
            other_ticker = full_df.iloc[idx]['ticker']
            score = cosine_sim[idx]
            
            if other_ticker == target_ticker: continue # Skip self
            if score < 0.15: break # Cutoff threshold
            if count >= 100: break # Max limit per direction

            # Generate the "WHY"
            target_text = full_df.iloc[target_idx]['business_summary']
            other_text = full_df.iloc[idx]['business_summary']
            
            reason_sentence = self._extract_reason(target_text, other_text, feature_names)
            
            # Save "Competitor/Peer" relationship (Bidirectional)
            # (Note: NLP finds peers. Finding pure 'suppliers' usually requires expensive data, 
            # so we label these as 'semantic_peer' for now)
            self.db.save_relationship(target_ticker, other_ticker, "semantic_peer", reason_sentence, float(score))
            self.db.save_relationship(other_ticker, target_ticker, "semantic_peer", reason_sentence, float(score))
            
            count += 1
            
        print(f"✅ Found {count} relevant connections for {target_ticker}.")

    def analyze_ticker(self, ticker):
        """
        The main public function to print the report.
        """
        # 1. Build/Update the graph
        self.build_graph_for_ticker(ticker)
        
        # 2. Query the SQL DB
        outgoing, incoming = self.db.get_network_for_stock(ticker, limit=20)
        
        print(f"\n{'='*60}")
        print(f"🌐 KNOWLEDGE GRAPH REPORT: {ticker}")
        print(f"{'='*60}")
        
        print(f"\n--- RELATED ENTITIES ({len(outgoing)} found) ---")
        if not outgoing.empty:
            for _, row in outgoing.iterrows():
                print(f"🔗 {row['ticker']:<6} | Strength: {row['strength']:.2f}")
                print(f"   💡 Reason: {row['reason']}")
                print("-" * 40)
        else:
            print("No strong connections found.")
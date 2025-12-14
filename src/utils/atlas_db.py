import sqlite3
import pandas as pd
import networkx as nx
import os
import json
import logging
from datetime import datetime, timedelta
import math
import threading

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ATLAS_DB] %(levelname)s - %(message)s')
logger = logging.getLogger("AtlasDB")

class AtlasDB:
    def __init__(self, db_path="data/market_memory.db"):
        """
        The Core Knowledge Graph Database.
        Manages Entities (Assets) and Edges (Relationships).
        """
        # Ensure the directory exists
        try:
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        except OSError:
            pass

        self.db_path = db_path
        self.conn_lock = threading.Lock()
        self._init_db()

    def _get_connection(self):
        """Returns a thread-safe connection object."""
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        """Initialize SQL tables if they don't exist."""
        with self.conn_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 1. ASSETS (Nodes)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS assets (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    aliases TEXT, -- JSON list
                    sector TEXT,
                    last_updated TIMESTAMP
                )
            ''')

            # 2. RELATIONSHIPS (Edges)
            # structural_strength: How essential is this link? (0.0 to 1.0)
            # active_impact: How relevant is this news RIGHT NOW? (Calculated, not stored, but we store the date)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    target TEXT,
                    type TEXT,       -- supplier, competitor, customer, etc.
                    reason TEXT,     -- Short text explanation
                    structural_strength REAL, -- The base reality of the link (0.0 - 1.0)
                    sentiment REAL,  -- -1.0 to 1.0
                    confidence REAL, -- AI Confidence score
                    first_verified TIMESTAMP,
                    last_verified TIMESTAMP,
                    source_url TEXT,
                    UNIQUE(source, target, type)
                )
            ''')

            conn.commit()
            conn.close()

    # ==========================
    # ENTITY MANAGEMENT
    # ==========================
    def resolve_entity(self, name):
        """
        Tries to map a name (e.g., "Foxconn") to a Ticker (e.g., "2317.TW").
        Returns Ticker or None.
        """
        name_clean = name.strip().upper()
        
        with self.conn_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Direct Ticker Match
            cursor.execute("SELECT ticker FROM assets WHERE ticker = ?", (name_clean,))
            res = cursor.fetchone()
            if res:
                conn.close()
                return res[0]
            
            # 2. Name Match
            cursor.execute("SELECT ticker FROM assets WHERE name = ?", (name_clean,))
            res = cursor.fetchone()
            if res:
                conn.close()
                return res[0]

            # 3. Alias Search (Slow, but necessary)
            # In production, use FTS5 (Full Text Search) for this
            cursor.execute("SELECT ticker, aliases FROM assets")
            rows = cursor.fetchall()
            conn.close()

            for ticker, aliases_json in rows:
                if aliases_json:
                    try:
                        aliases = json.loads(aliases_json)
                        if any(a.upper() == name_clean for a in aliases):
                            return ticker
                    except:
                        continue
        return None

    def save_asset(self, ticker, name=None, aliases=None, sector=None):
        """Upserts an asset node."""
        if not ticker: return
        ticker = ticker.upper()
        
        with self.conn_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check existing to merge aliases
            cursor.execute("SELECT aliases FROM assets WHERE ticker = ?", (ticker,))
            row = cursor.fetchone()
            
            existing_aliases = []
            if row and row[0]:
                try:
                    existing_aliases = json.loads(row[0])
                except:
                    existing_aliases = []
            
            # Merge new aliases
            if aliases:
                existing_aliases.extend(aliases)
                existing_aliases = list(set(existing_aliases)) # Dedupe
            
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO assets (ticker, name, aliases, sector, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    aliases = excluded.aliases,
                    last_updated = excluded.last_updated,
                    name = COALESCE(excluded.name, assets.name)
            ''', (ticker, name, json.dumps(existing_aliases), sector, now))
            
            conn.commit()
            conn.close()

    # ==========================
    # RELATIONSHIP MANAGEMENT (The Core)
    # ==========================
    def save_relationship(self, source, target, rel_type, reason, strength=0.5, sentiment=0.0, confidence=0.5, source_url=""):
        """
        Upserts a relationship edge.
        If edge exists, we update the 'last_verified' date and average the sentiment.
        """
        source = source.upper()
        target = target.upper()
        if source == target: return

        now = datetime.now().isoformat()
        
        with self.conn_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check if exists
            cursor.execute('''
                SELECT id, structural_strength, sentiment, first_verified 
                FROM relationships 
                WHERE source=? AND target=? AND type=?
            ''', (source, target, rel_type))
            
            row = cursor.fetchone()
            
            if row:
                # UPDATE EXISTING
                row_id, old_strength, old_sentiment, first_date = row
                
                # Logic: Structural Strength tends to increase with verification frequency
                # We cap it at 1.0. We verify it exists, so we bump it slightly.
                new_strength = min(old_strength + 0.05, 1.0)
                if strength > old_strength: new_strength = strength # Jump if new data is very strong
                
                # Logic: Sentiment is a moving average (70% new, 30% old)
                new_sentiment = (old_sentiment * 0.3) + (sentiment * 0.7)
                
                cursor.execute('''
                    UPDATE relationships 
                    SET structural_strength=?, sentiment=?, confidence=?, last_verified=?, reason=?, source_url=?
                    WHERE id=?
                ''', (new_strength, new_sentiment, confidence, now, reason, source_url, row_id))
                
            else:
                # INSERT NEW
                cursor.execute('''
                    INSERT INTO relationships (source, target, type, reason, structural_strength, sentiment, confidence, first_verified, last_verified, source_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (source, target, rel_type, reason, strength, sentiment, confidence, now, now, source_url))

            conn.commit()
            conn.close()

    # ==========================
    # POWER LAW DECAY & PRUNING
    # ==========================
    def prune_dead_relationships(self, delete_threshold=0.1, max_age_days=1800):
        """
        Maintenance task:
        1. Deletes relationships that have low structural strength AND haven't been verified in a long time.
        2. 'Permanent' deals (like Google-Apple) usually stay because they get re-verified often or have high strength.
        """
        with self.conn_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Calculate cutoff date
            cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
            
            # Delete if: 
            # (Strength < Threshold) OR (Last Verified < 5 years ago)
            # We keep old high-strength links, but even they die if not re-verified in 5 years.
            cursor.execute('''
                DELETE FROM relationships 
                WHERE structural_strength < ? OR last_verified < ?
            ''', (delete_threshold, cutoff))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            if deleted_count > 0:
                logger.info(f"✂️ Pruned {deleted_count} dead relationships.")

    def _calculate_active_impact(self, structural_strength, last_verified_str):
        """
        The Power Law Decay Function.
        
        Formula: Impact = Strength * (Time_Decay)
        
        Where Time_Decay uses a Power Law: 
        Validation is recent? Impact is 100%.
        Validation is old? Impact drops sharply, but long tail remains.
        
        We want 'Active News' to matter for ~30 days, then fade to background 'Structure'.
        """
        try:
            last_date = datetime.fromisoformat(last_verified_str)
            days_diff = (datetime.now() - last_date).days
            
            # Avoid division by zero
            if days_diff < 0: days_diff = 0
            t = days_diff + 1 # Shift so t=1 at day 0
            
            # Power Law: y = k * x^(-alpha)
            # Alpha controls steepness. 
            # Alpha 0.5 = Gentle decay (Long memory)
            # Alpha 1.5 = Sharp decay (News cycle)
            
            # We want news impact to fade, but structure to persist.
            # So we don't decay the Strength directly, we calculate a "Current Attention Score"
            
            # For this specific function, let's model "Relevance":
            # 1 Day Old: 1.0
            # 7 Days Old: 0.37
            # 30 Days Old: 0.18
            # 365 Days Old: 0.05
            
            decay_factor = 1 / (t ** 0.5) 
            
            # Impact is a mix of how strong the link is structurally, and how fresh it is.
            # But wait! As you said: Apple-Google 2010 deal is OLD but HIGH IMPACT.
            
            # HYBRID SCORE:
            # If Structural Strength is High (>0.8), we enforce a "Floor" so it never disappears from the graph.
            # If Structural Strength is Low (<0.5), it decays to near zero.
            
            floor = 0.0
            if structural_strength > 0.8:
                floor = 0.3 # High structure links always retain 30% active weight
            elif structural_strength > 0.6:
                floor = 0.1
                
            active_impact = max((structural_strength * decay_factor), floor)
            
            return active_impact

        except Exception as e:
            logger.error(f"Decay calc error: {e}")
            return 0.0

    # ==========================
    # GRAPH EXPORT FOR AGENTS
    # ==========================
    def get_market_graph(self, filter_type=None, min_impact=0.05):
        """
        Returns a NetworkX DiGraph.
        Nodes: Tickers
        Edges: Weighted by 'Active Impact' (Decayed Strength).
        
        Args:
            filter_type: Optional (e.g., 'supplier', 'competitor')
            min_impact: Cutoff for edge inclusion (filters out noise).
        """
        G = nx.DiGraph()
        
        with self.conn_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = "SELECT source, target, type, structural_strength, sentiment, last_verified FROM relationships"
            params = []
            
            if filter_type:
                query += " WHERE type = ?"
                params.append(filter_type)
                
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            conn.close()
            
        for r in rows:
            src, tgt, r_type, str_strength, sentiment, last_ver = r
            
            # Apply Power Law Decay
            active_impact = self._calculate_active_impact(str_strength, last_ver)
            
            if active_impact >= min_impact:
                G.add_edge(
                    src, tgt, 
                    relation=r_type,
                    weight=active_impact,       # Used for graph algorithms (PageRank etc)
                    structural_strength=str_strength,
                    sentiment=sentiment,
                    last_verified=last_ver
                )
                
        return G

    def get_network_for_stock(self, ticker, depth=1):
        """
        Returns immediate neighbors and edges for a specific stock.
        Useful for the 'Context Injection' in your Forensic Agent.
        """
        ticker = ticker.upper()
        G = self.get_market_graph(min_impact=0.01) # Get almost everything for context
        
        if ticker not in G:
            return [], []
            
        # Get ego graph (local subgraph)
        if depth == 1:
            neighbors = list(G.successors(ticker)) + list(G.predecessors(ticker))
            edges = []
            for n in neighbors:
                if G.has_edge(ticker, n):
                    edges.append(G.get_edge_data(ticker, n))
                if G.has_edge(n, ticker):
                    edges.append(G.get_edge_data(n, ticker))
            return neighbors, edges
        
        return [], []

    # ==========================
    # ANALYSIS TOOLS
    # ==========================
    def simulate_shockwave(self, root_ticker, shock_percent=-0.10):
        """
        Propagates a price shock through the supply chain.
        Uses the 'Active Impact' weights.
        """
        G = self.get_market_graph(filter_type="supplier") # Only suppliers pass shock downstream? 
        # Actually: If Supplier crashes, Customer gets hurt (Supply Chain interruption).
        # Graph direction: Supplier -> Customer.
        # So we follow the edges.
        
        impacts = {root_ticker: shock_percent}
        queue = [(root_ticker, shock_percent)]
        processed = set([root_ticker])
        
        while queue:
            current_node, current_shock = queue.pop(0)
            
            # Dampening factor (shock loses energy as it travels)
            if abs(current_shock) < 0.01: continue 
            
            if current_node in G:
                customers = G.successors(current_node)
                for cust in customers:
                    if cust in processed: continue
                    
                    edge_data = G.get_edge_data(current_node, cust)
                    weight = edge_data.get('weight', 0.1) # Active Impact
                    
                    # Logic: If I depend 50% on this supplier, I take 50% of the shock?
                    # Simplified model: Transferred Shock = Shock * Weight * 0.5
                    transferred_shock = current_shock * weight * 0.5
                    
                    impacts[cust] = impacts.get(cust, 0) + transferred_shock
                    processed.add(cust)
                    queue.append((cust, transferred_shock))
                    
        return pd.DataFrame(list(impacts.items()), columns=['Ticker', 'Predicted_Impact']).sort_values(by='Predicted_Impact')

if __name__ == "__main__":
    # Test Logic
    db = AtlasDB()
    
    # 1. Add Old Strong Deal (The "Google-Apple" Scenario)
    # 5 years ago, very strong
    old_date = (datetime.now() - timedelta(days=1800)).isoformat()
    db.save_relationship("GOOGL", "AAPL", "partner", "Search Engine Deal", strength=0.95, sentiment=0.5)
    
    # Manually hack the date back to test decay
    with db.conn_lock:
        conn = db._get_connection()
        conn.execute("UPDATE relationships SET last_verified = ? WHERE source='GOOGL'", (old_date,))
        conn.commit()
        conn.close()

    # 2. Add Recent Weak Deal (Rumor)
    db.save_relationship("NVDA", "INTC", "competitor", "Rumors of foundry work", strength=0.3)

    # 3. Check Graphs
    print("\n--- GRAPH WEIGHT ANALYSIS ---")
    G = db.get_market_graph(min_impact=0.0) # Get all
    
    if G.has_edge("GOOGL", "AAPL"):
        data = G.get_edge_data("GOOGL", "AAPL")
        print(f"GOOGL->AAPL (Old/Strong): Active Impact = {data['weight']:.4f} (Base: {data['structural_strength']})")
        print("   -> Note: Even though it's 5 years old, weight should > 0 because Base > 0.8 (The Floor)")
        
    if G.has_edge("NVDA", "INTC"):
        data = G.get_edge_data("NVDA", "INTC")
        print(f"NVDA->INTC (New/Weak):    Active Impact = {data['weight']:.4f} (Base: {data['structural_strength']})")

    # 4. Test Pruning
    print("\n--- PRUNING TEST ---")
    # This should NOT delete GOOGL-AAPL (Strength 0.95 > Threshold 0.1)
    # This should NOT delete NVDA-INTC (Recent)
    db.prune_dead_relationships()
    
    # Add a dead node to test deletion
    very_old = (datetime.now() - timedelta(days=2000)).isoformat()
    db.save_relationship("DEAD", "CO", "supplier", "Old", strength=0.05)
    with db.conn_lock:
        conn = db._get_connection()
        conn.execute("UPDATE relationships SET last_verified = ? WHERE source='DEAD'", (very_old,))
        conn.commit()
    
    print("Pruning dead node...")
    db.prune_dead_relationships()
import pandas as pd
import time
import sys
import os

# -------------------------------------------------------------------------
# SETUP: Add 'src' to path so we can import our modules reliably
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from src.agents.forensic_agent import ForensicAgent
    from src.utils.atlas_db import AtlasDB
except ImportError as e:
    print("❌ Import Error: Could not find src modules.")
    print(f"Details: {e}")
    print("Make sure you are running this from the root directory of your project.")
    sys.exit(1)

# -------------------------------------------------------------------------
# 1. BUILD PHASE: The Agent goes to the web
# -------------------------------------------------------------------------
def build_knowledge_graph(target_tickers):
    """
    Uses the ForensicAgent to search the web, read 10ks/News, 
    and build the supply chain graph in SQL.
    """
    print(f"\n🏗️  INITIALIZING WORLD BUILDER")
    print(f"==========================================")
    
    agent = ForensicAgent()
    
    # We maintain a queue to allow recursive discovery
    # (e.g., If we scan NVDA and find TSMC, we add TSMC to the list)
    investigation_queue = target_tickers.copy()
    visited = set()
    
    # Safety limit to prevent it from running all night during tests
    MAX_NEW_DISCOVERIES = 5 
    
    while investigation_queue:
        current_ticker = investigation_queue.pop(0)
        
        # Skip if already analyzed in this session
        if current_ticker in visited:
            continue
            
        print(f"\n🔎  AGENT ASSIGNED: {current_ticker}")
        
        try:
            # The agent returns a list of NEW companies it found referenced in the text
            new_leads = agent.investigate(current_ticker)
            visited.add(current_ticker)
            
            # Recursive Logic: If we found a crucial supplier, should we investigate them too?
            if len(visited) < MAX_NEW_DISCOVERIES:
                for lead in new_leads:
                    if lead not in visited and lead not in investigation_queue:
                        print(f"   🔭 Curiosity Triggered: Adding {lead} to queue.")
                        investigation_queue.append(lead)
            else:
                if new_leads:
                    print(f"   🛑 Max depth reached. Ignoring {len(new_leads)} new leads.")

        except Exception as e:
            print(f"   ❌ Error analyzing {current_ticker}: {e}")
            
    print(f"\n✅ Build Phase Complete. Analyzed: {visited}")


# -------------------------------------------------------------------------
# 2. ANALYSIS PHASE: The Math Agent calculates risks
# -------------------------------------------------------------------------
def analyze_market_risks():
    """
    Connects to the AtlasDB (Graph) and calculates:
    1. Centrality (Who is the Kingpin?)
    2. Betweenness (Who is the Bottleneck?)
    3. Dependencies (Who relies on who?)
    """
    print(f"\n📊  RUNNING GRAPH MATHEMATICS")
    print(f"==========================================")
    
    db = AtlasDB()
    stats = db.get_stats()
    
    if stats['companies'] == 0:
        print("⚠️ Database is empty. Please run the Build Phase first.")
        return

    print(f"Global Knowledge: {stats['companies']} Companies | {stats['connections']} Relationships")

    # A. IDENTIFY SYSTEMIC RISKS (Centrality)
    print("\n👑  MARKET KINGS (PageRank)")
    print("------------------------------------------")
    # This identifies companies that everyone else relies on (e.g., TSMC, ASML, Nvidia)
    risk_df = db.calculate_market_centrality()
    
    if not risk_df.empty:
        # Show top 5 dominant players
        print(risk_df[['ticker', 'dominance_score', 'bottleneck_score']].head(5).to_string(index=False))
        
        # Highlight the #1 most critical stock
        top_dog = risk_df.iloc[0]['ticker']
        print(f"\n🚨 KEY FINDING: '{top_dog}' is the most critical node in your current graph.")
    else:
        print("   (Not enough data to calculate centrality yet)")

    # B. DEEP DIVE: PREDICTION CONTEXT
    # Let's verify what we know about a specific stock to help the prediction engine
    target = "NVDA" 
    print(f"\n🕵️  INTELLIGENCE REPORT: {target}")
    print("------------------------------------------")
    
    downstream, upstream = db.get_network_for_stock(target)
    
    print(f"📉 WHO RELIES ON {target}? (If {target} fails, these drop)")
    if not downstream.empty:
        for _, row in downstream.head(5).iterrows():
            print(f"   - {row['ticker']:<6} [{row['relation_type']}] Reason: {row['reason'][:60]}... (Conf: {row.get('confidence', 0.8):.1f})")
    else:
        print("   (No known customers found)")

    print(f"\n📈 WHO DOES {target} RELY ON? (If these fail, {target} drops)")
    if not upstream.empty:
        for _, row in upstream.head(5).iterrows():
            print(f"   - {row['ticker']:<6} [{row['relation_type']}] Reason: {row['reason'][:60]}... (Strength: {row['strength']})")
    else:
        print("   (No known suppliers found)")

    # C. PATHFINDING (Strategic Exposure)
    # Check if a shock in Taiwan (TSM) hits Microsoft (MSFT)
    start_node = "TSM"
    end_node = "MSFT"
    print(f"\n🔗  CONTAGION PATH: {start_node} -> {end_node}")
    path = db.find_connection_path(start_node, end_node)
    
    if path:
        print(f"   ⚠️ WARNING: Connection Found!")
        arrow_path = " -> ".join(path)
        print(f"   Path: {arrow_path}")
        print(f"   Analysis: A supply shock at {start_node} will ripple through to {end_node}.")
    else:
        print(f"   No direct dependency chain found between {start_node} and {end_node}.")


# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Define the universe to start with
    # In production, this might be your portfolio or the S&P 500 top 10
    start_tickers = ["NVDA", "AAPL"]
    
    # 2. Run the Builder (Comment this out if you just want to analyze existing DB)
    build_knowledge_graph(start_tickers)
    
    # 3. Run the Analytics
    analyze_market_risks()
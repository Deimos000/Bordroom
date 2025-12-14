import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import time
from datetime import datetime, timedelta
from tqdm import tqdm

# --- IMPORT AGENTS ---
# Ensure these files exist in src/agents/ and src/utils/
try:
    from src.agents.doomsday_agent import DoomsdayAgent
    from src.agents.smart_math_agent import SmartMathAgent
    from src.agents.forensic_agent import ForensicAgent
    from src.agents.boss_agent import BossAgent
    from src.utils.atlas_db import AtlasDB
    from src.utils.universe_factory import UniverseFactory
    # Assuming MacroDataLoader is in utils or you have a standalone script for it
    # If not, we will implement a quick fetcher inline
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Ensure you are running from the project root and your virtual env is active.")
    sys.exit(1)

# --- CONFIGURATION ---
TARGET_TICKER = "NVDA"
SIMULATION_DAYS = 180  # How far back to simulate history for Boss training
FAST_MODE = True       # If True, trains Math Agent on fewer stocks for speed
LOG_LEVEL = logging.INFO

# --- LOGGING ---
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - [ORCHESTRATOR] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Main")

def setup_folders():
    dirs = ["data", "models", "logs", "data/cache"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def fetch_global_macro_data():
    """
    Ensures we have the macro data for the Doomsday Agent.
    """
    path = "data/global_macro_history.csv"
    if os.path.exists(path):
        logger.info("✅ Global macro data found.")
        return pd.read_csv(path, index_col=0, parse_dates=True)
    
    logger.info("📉 Downloading fresh Global Macro Data...")
    # Tickers: Gold, Copper, Oil, 10Y Yield, VIX
    tickers = ["GC=F", "HG=F", "BZ=F", "^TNX", "^VIX"]
    df = yf.download(tickers, period="5y", progress=False)['Close']
    
    # Rename for Doomsday Agent compatibility
    rename_map = {
        "GC=F": "Gold", "HG=F": "Copper", "BZ=F": "Oil", 
        "^TNX": "Yield10Y", "^VIX": "VIX"
    }
    # Handle columns if multi-index (common in new yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.rename(columns=rename_map)
    
    # Add dummy GDELT if missing (Real GDELT requires API key/complex scraping)
    df['GDELT_Tone'] = np.random.uniform(-1, 1, size=len(df)) 
    
    df.to_csv(path)
    return df

def generate_boss_curriculum(target_ticker, agent_math, agent_doom, db, macro_df):
    """
    The Time Machine.
    Walks forward day-by-day to generate training examples for the Boss.
    """
    logger.info(f"⏳ Generating Boss Curriculum for {target_ticker} ({SIMULATION_DAYS} days)...")
    
    # 1. Get History for Target
    # We fetch extra buffer for the Math Agent's lookback
    start_date = (datetime.now() - timedelta(days=SIMULATION_DAYS + 100)).strftime('%Y-%m-%d')
    price_history = yf.download(target_ticker, start=start_date, progress=False)
    
    if isinstance(price_history.columns, pd.MultiIndex):
        price_history = price_history.xs(target_ticker, axis=1, level=0)
    
    training_rows = []
    
    # We start from SIMULATION_DAYS ago up to yesterday
    # We need at least 60 days prior to 'i' for the Math Agent to work
    min_idx = 60 
    max_idx = len(price_history) - 30 # Leave room for month-ahead labels
    
    if max_idx < min_idx:
        logger.error("❌ Not enough history to simulate.")
        return pd.DataFrame()

    # Iterate
    for i in tqdm(range(min_idx, max_idx), desc="Simulating Council"):
        try:
            # --- 1. DEFINE "NOW" ---
            current_date = price_history.index[i]
            current_slice = price_history.iloc[:i+1] # Data known up to today
            current_price = current_slice['Close'].iloc[-1]
            
            # --- 2. GET SUB-AGENT OPINIONS ---
            
            # A. Math Agent (Technical)
            # Predicts brackets based on current_slice
            math_preds = agent_math.predict(target_ticker, current_slice)
            if not math_preds: continue
            
            # B. Doomsday Agent (Macro)
            # Slice macro data to match date
            macro_slice = macro_df[macro_df.index <= current_date]
            if len(macro_slice) < 30: continue
            doom_res = agent_doom.predict_doomsday(macro_slice)
            
            # Extract Score (0 to 1)
            # We map "CRITICAL" to 0.9, "STABLE" to 0.2, etc.
            doom_score = 0.8 if doom_res['status'] == "CRITICAL RISK" else 0.2
            
            # C. Forensic Agent (Fundamental/Network)
            # (Note: In a perfect world, we'd query the DB for the state *at that date*)
            # For now, we use current structural centrality but apply random noise 
            # to simulate sentiment fluctuation over time.
            net = db.get_network_for_stock(target_ticker)
            centrality = min(len(net[0]) / 15.0, 1.0)
            sentiment = np.random.uniform(-0.5, 0.5) # Placeholder for historic sentiment
            
            # --- 3. CALCULATE ACTUAL OUTCOMES (Labels) ---
            # What actually happened 1 day, 1 week, and 1 month later?
            
            future_1d = price_history.iloc[i+1:i+2]
            future_1w = price_history.iloc[i+1:i+6]
            future_1m = price_history.iloc[i+1:i+22]
            
            if future_1m.empty: continue
            
            row = {
                'date': current_date,
                'current_price': current_price,
                
                # Context Inputs
                'doomsday_score': doom_score,
                'forensic_sentiment': sentiment,
                'graph_centrality': centrality,
                
                # Math Inputs (The Expert Brackets)
                'math_day_bear': math_preds['day_bear'],
                'math_day_base': math_preds['day_base'],
                'math_day_bull': math_preds['day_bull'],
                'math_week_bear': math_preds['week_bear'],
                'math_week_base': math_preds['week_base'],
                'math_week_bull': math_preds['week_bull'],
                'math_month_bear': math_preds['month_bear'],
                'math_month_base': math_preds['month_base'],
                'math_month_bull': math_preds['month_bull'],
                
                # ACTUAL TARGETS (Labels for Boss)
                'ACTUAL_day_low': future_1d['Low'].min(),
                'ACTUAL_day_close': future_1d['Close'].iloc[-1],
                'ACTUAL_day_high': future_1d['High'].max(),
                
                'ACTUAL_week_low': future_1w['Low'].min(),
                'ACTUAL_week_close': future_1w['Close'].iloc[-1],
                'ACTUAL_week_high': future_1w['High'].max(),
                
                'ACTUAL_month_low': future_1m['Low'].min(),
                'ACTUAL_month_close': future_1m['Close'].iloc[-1],
                'ACTUAL_month_high': future_1m['High'].max(),
            }
            training_rows.append(row)
            
        except Exception as e:
            continue

    return pd.DataFrame(training_rows)

def main():
    setup_folders()
    print("\n" + "="*60)
    print(f"🤖  AI FINANCIAL ORCHESTRATOR | Target: {TARGET_TICKER}")
    print("="*60 + "\n")

    # --- 1. INITIALIZE COUNCIL ---
    logger.info("Initializing Agents...")
    agent_doom = DoomsdayAgent()
    agent_math = SmartMathAgent()
    agent_forensic = ForensicAgent() # Assuming standard init
    agent_boss = BossAgent()
    db = AtlasDB()

    # --- 2. ENSURE DATA ---
    macro_df = fetch_global_macro_data()

    # --- 3. TRAIN SUB-AGENTS ---
    
    # A. Doomsday Agent
    if not agent_doom.load_brain():
        logger.warning("☢️  Doomsday Agent needs training...")
        agent_doom.train(macro_df)
    else:
        logger.info("☢️  Doomsday Agent ready.")

    # B. Math Agent
    if not agent_math.load_brain():
        logger.warning("📐 Math Agent needs training...")
        if FAST_MODE:
            logger.info("   (Fast Mode: Training on top 30 S&P stocks)")
            universe = UniverseFactory.get_sp500()[:30]
        else:
            universe = UniverseFactory.get_sp500()
        
        if TARGET_TICKER not in universe: universe.append(TARGET_TICKER)
        agent_math.train(universe)
    else:
        logger.info("📐 Math Agent ready.")
        
    # C. Forensic Agent (Graph Build)
    stats = db.get_stats() # Assuming get_stats returns dict with keys
    # Check if graph is empty (simple check if file exists or query returns 0)
    # Here we assume get_market_graph returns networkx object
    G = db.get_market_graph()
    if len(G.nodes) < 5:
        logger.warning("🕵️  Forensic Graph empty. Building mini-graph...")
        agent_forensic.start_investigation(TARGET_TICKER, max_depth=1)
    else:
        logger.info(f"🕵️  Forensic Graph ready ({len(G.nodes)} nodes).")

    # --- 4. TRAIN THE BOSS ---
    if not agent_boss.load_brain():
        logger.warning("🕴️  The Boss needs to learn strategy...")
        
        # 1. Generate History
        boss_data = generate_boss_curriculum(TARGET_TICKER, agent_math, agent_doom, db, macro_df)
        
        if not boss_data.empty:
            # 2. Train Boss
            agent_boss.train(boss_data)
        else:
            logger.error("❌ Failed to generate Boss training data.")
            sys.exit(1)
    else:
        logger.info("🕴️  The Boss is awake and ready.")

    # --- 5. LIVE PREDICTION (TODAY) ---
    print("\n" + "*"*60)
    print("🚀  STARTING LIVE COUNCIL SESSION")
    print("*"*60)
    
    # Get Data
    hist_data = yf.download(TARGET_TICKER, period="2y", progress=False)
    if isinstance(hist_data.columns, pd.MultiIndex):
        hist_data = hist_data.xs(TARGET_TICKER, axis=1, level=0)
    
    current_price = hist_data['Close'].iloc[-1]
    
    # 1. Math Opinion
    print("   📐 Math Agent thinking...")
    math_res = agent_math.predict(TARGET_TICKER, hist_data)
    
    # 2. Doomsday Opinion
    print("   ☢️  Doomsday Agent scanning...")
    doom_res = agent_doom.predict_doomsday(macro_df)
    
    # 3. Forensic Opinion
    print("   🕵️  Forensic Agent checking connections...")
    net_nodes, _ = db.get_network_for_stock(TARGET_TICKER)
    centrality = min(len(net_nodes) / 15.0, 1.0)
    # In a real app, calculate sentiment from recent news in VectorMemory
    sentiment = 0.2 # Neutral/Positive lean
    
    forensic_data = {
        'centrality': centrality,
        'sentiment': sentiment
    }

    if math_res and doom_res:
        # 4. THE BOSS DECIDES
        print("\n🕴️  THE BOSS IS DECIDING...")
        final_verdict = agent_boss.make_decision(
            ticker=TARGET_TICKER,
            current_price=current_price,
            doomsday_data=doom_res,
            forensic_data=forensic_data,
            math_data=math_res
        )
        
        # --- OUTPUT REPORT ---
        print("\n" + "="*60)
        print(f"📊 FINAL STRATEGIC REPORT: {TARGET_TICKER}")
        print("="*60)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Confidence:    {final_verdict['boss_confidence']}")
        print(f"Logic:         {final_verdict['logic']}")
        print("-" * 60)
        print(f"{'HORIZON':<10} | {'BEAR':<12} | {'TARGET (BASE)':<15} | {'BULL':<12}")
        print("-" * 60)
        print(f"{'1 Day':<10} | ${final_verdict['day_bear']:<12.2f} | ${final_verdict['day_base']:<15.2f} | ${final_verdict['day_bull']:<12.2f}")
        print(f"{'1 Week':<10} | ${final_verdict['week_bear']:<12.2f} | ${final_verdict['week_base']:<15.2f} | ${final_verdict['week_bull']:<12.2f}")
        print(f"{'1 Month':<10} | ${final_verdict['month_bear']:<12.2f} | ${final_verdict['month_base']:<15.2f} | ${final_verdict['month_bull']:<12.2f}")
        print("="*60)
        
    else:
        logger.error("❌ Sub-agents failed to produce signals.")

if __name__ == "__main__":
    main()
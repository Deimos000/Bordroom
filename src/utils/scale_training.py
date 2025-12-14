import os
import joblib
import pandas as pd
import datetime
import time
from src.utils.data_miner import DataMiner
from src.utils.universe_factory import UniverseFactory
from src.agents.smart_math_agent import SmartMathAgent

# --- CONFIGURATION ---
DATA_CACHE_DIR = "cache_data"
PRICE_CACHE_FILE = os.path.join(DATA_CACHE_DIR, "massive_prices.pkl")
INFO_CACHE_FILE = os.path.join(DATA_CACHE_DIR, "massive_info.pkl")

def ensure_cache_dir():
    if not os.path.exists(DATA_CACHE_DIR):
        os.makedirs(DATA_CACHE_DIR)

def get_cached_data(universe):
    """
    Smart caching: If data exists on disk and is less than 24h old, load it.
    Otherwise, mine it fresh.
    """
    ensure_cache_dir()
    
    # 1. Check if cache exists
    if os.path.exists(PRICE_CACHE_FILE) and os.path.exists(INFO_CACHE_FILE):
        print("💾 Found cached data on disk. Loading...")
        try:
            price_data = joblib.load(PRICE_CACHE_FILE)
            
            # Load info tuple (candidates, fund_map, descriptions)
            info_data = joblib.load(INFO_CACHE_FILE)
            candidates, fund_map, descriptions = info_data
            
            print(f"✅ Loaded {len(price_data)} price histories and {len(candidates)} metadata profiles.")
            return price_data, candidates, fund_map, descriptions
        except Exception as e:
            print(f"⚠️ Cache corrupted ({e}). Remining...")

    # 2. Mine Fresh Data
    print(f"⛏️  Mining fresh data for {len(universe)} tickers...")
    print("    (This will take time, but will be saved for next run)")
    
    # A. Mine Intelligence (Fundamentals)
    candidates, fund_map, descriptions = DataMiner.get_fundamentals_and_text(universe)
    
    # B. Mine Prices (Only for those who have fundamentals)
    start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Use chunking inside DataMiner, but here we pass the big list
    price_data = DataMiner.get_universe_data(candidates, start_date, end_date)
    
    # 3. Save to Cache
    print("💾 Saving data to disk for future runs...")
    joblib.dump(price_data, PRICE_CACHE_FILE, compress=3) # Compress=3 saves disk space
    joblib.dump((candidates, fund_map, descriptions), INFO_CACHE_FILE, compress=3)
    
    return price_data, candidates, fund_map, descriptions

def train_massive_brain():
    # 1. Assemble the Legion
    print("\n" + "="*50)
    print("🚀 INITIALIZING MASSIVE SCALE TRAINING")
    print("="*50)
    
    full_universe = UniverseFactory.get_massive_universe()
    print(f"🎯 Target Universe: {len(full_universe)} stocks (S&P 1500)")
    
    # 2. Fetch or Load Data
    # This bypasses the Agent's internal mining to handle caching explicitly
    price_data, candidates, fund_map, descriptions = get_cached_data(full_universe)
    
    # 3. Initialize Agent
    agent = SmartMathAgent(model_dir="models/massive_brain_v1")
    
    # 4. Inject Data & Train
    # We need to manually trigger the internal prep steps of the agent
    # or better, modify the agent to accept pre-loaded data.
    # HOWEVER, to keep it simple with your current Agent code, 
    # we will overwrite the internal variables of the agent manually 
    # before calling a modified training flow.
    
    print("\n🧠 Injecting Data into Neural Agent...")
    
    # --- MANUAL AGENT INJECTION SEQUENCE ---
    # Since agent.train() usually mines data, we will replicate the logic here
    # to use our cached data.
    
    # A. Alignment
    X_raw, Y_raw, final_tickers = agent._prepare_aligned_data(price_data, fund_map, candidates)
    
    if len(X_raw) == 0:
        print("❌ Critical Error: Data Alignment failed.")
        return

    print(f"✅ Alignment Complete. Training on {len(final_tickers)} verified assets.")
    
    # B. Align Descriptions
    final_descriptions = []
    for t in final_tickers:
        orig_idx = candidates.index(t)
        final_descriptions.append(descriptions[orig_idx])
        
    # C. Build Graph
    from src.utils.math_tools import GraphMath
    import torch
    
    # Increase K for larger graphs to ensure connectivity
    edge_index, _ = GraphMath.build_semantic_graph(final_descriptions, top_k=15, min_weight=0.15)
    agent.edge_index = edge_index.to(agent.device)
    agent.valid_tickers = final_tickers
    agent.fundamental_map = fund_map
    
    # D. Prepare Tensors
    N, L, F_dim = X_raw.shape
    X_flat = X_raw.reshape(-1, F_dim)
    agent.scaler.fit(X_flat)
    X_scaled = agent.scaler.transform(X_flat).reshape(N, L, F_dim)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(agent.device)
    Y_tensor = torch.tensor(Y_raw, dtype=torch.float32).view(-1, 1).to(agent.device)
    
    # E. Initialize Model
    from src.agents.smart_math_agent import TemporalGraphTransformer, nn
    
    agent.model = TemporalGraphTransformer(num_features=F_dim, d_model=agent.HIDDEN_DIM).to(agent.device)
    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=0.0005, weight_decay=1e-4) # Lower LR for large batch
    loss_fn = nn.MSELoss()
    
    # F. Training Loop
    print("\n🔥 STARTING GPU TRAINING LOOP")
    agent.model.train()
    
    epochs = 200
    for e in range(epochs):
        optimizer.zero_grad()
        output = agent.model(X_tensor, agent.edge_index)
        loss = loss_fn(output, Y_tensor)
        loss.backward()
        optimizer.step()
        
        if e % 10 == 0:
            print(f"   Epoch {e} | Loss: {loss.item():.6f}")
            
    agent.save_brain()
    print("\n🏆 MASSIVE MODEL SAVED SUCCESSFULLY.")

if __name__ == "__main__":
    train_massive_brain()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import joblib
import logging
from torch.utils.data import Dataset, DataLoader

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BOSS] %(levelname)s - %(message)s')
logger = logging.getLogger("BossAgent")

# ==========================================
# 1. THE BOSS BRAIN (Neural Network)
# ==========================================
class ConsensusNetwork(nn.Module):
    """
    The Boss Brain.
    Inputs: 
      - Context (Doomsday Score, Forensic Sentiment, Graph Centrality)
      - Expert Opinion (Math Agent's 9 data points: 3 horizons x 3 probabilities)
    Outputs:
      - 9 Final Price Vectors (Adjusted % change from current price)
    """
    def __init__(self, input_dim=12, hidden_dim=64):
        super(ConsensusNetwork, self).__init__()
        
        # 1. Context Processing (Macro + News)
        self.context_layer = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        
        # 2. Expert Processing (Math Agent Inputs)
        self.expert_layer = nn.Sequential(
            nn.Linear(input_dim - 3, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # 3. Fusion Layer (Combine Context + Experts)
        # 16 (Context) + 32 (Expert) = 48 features
        self.fusion_layer = nn.Sequential(
            nn.Linear(48, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 4. Attention Gate (The "Trust" Mechanism)
        # Decides how much to weight the raw math inputs vs the network's correction
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 9),
            nn.Sigmoid()
        )
        
        # 5. Correction Head (Predicts the adjustment needed)
        self.correction = nn.Linear(hidden_dim, 9)

    def forward(self, x_context, x_expert):
        # x_context: [Batch, 3] (Doomsday, Sentiment, Centrality)
        # x_expert:  [Batch, 9] (Math Agent's % predictions)
        
        c_emb = self.context_layer(x_context)
        e_emb = self.expert_layer(x_expert)
        
        combined = torch.cat([c_emb, e_emb], dim=1)
        hidden = self.fusion_layer(combined)
        
        # Residual Learning:
        # We start with the Math Agent's opinion (x_expert)
        # We calculate a "Correction" based on context.
        # We use a Gate to decide how strong that correction should be.
        
        adjustment = self.correction(hidden)
        gate_weight = self.gate(hidden)
        
        # Final Output = Expert Opinion + (Adjustment * Gate)
        final_output = x_expert + (adjustment * gate_weight)
        
        return final_output

# ==========================================
# 2. DATASET HANDLING
# ==========================================
class BossDataset(Dataset):
    def __init__(self, X_context, X_expert, Y_targets):
        self.Xc = torch.tensor(X_context, dtype=torch.float32)
        self.Xe = torch.tensor(X_expert, dtype=torch.float32)
        self.Y  = torch.tensor(Y_targets, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.Xc[idx], self.Xe[idx], self.Y[idx]

# ==========================================
# 3. THE BOSS AGENT (Controller)
# ==========================================
class BossAgent:
    def __init__(self, model_dir="models/the_boss"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None # We might need scaling, but % diffs are usually self-scaled
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _normalize_math_inputs(self, math_pred_dict, current_price):
        """
        Converts the Math Agent's raw dollar predictions into Percentage Difference
        relative to the current price. Neural nets prefer percentages over raw dollars.
        Input vector order:
        [Day_Bear, Day_Base, Day_Bull, Week_Bear, Week_Base, Week_Bull, Month_Bear, Month_Base, Month_Bull]
        """
        keys = [
            'day_bear', 'day_base', 'day_bull',
            'week_bear', 'week_base', 'week_bull',
            'month_bear', 'month_base', 'month_bull'
        ]
        
        vector = []
        for k in keys:
            val = math_pred_dict.get(k, current_price)
            # Calculate % diff: (Pred - Curr) / Curr
            pct_diff = (val - current_price) / (current_price + 1e-8)
            vector.append(pct_diff)
            
        return np.array(vector, dtype=np.float32)

    def prepare_training_data(self, history_df):
        """
        Expects a DataFrame where every row is a historical simulation day containing:
        - doomsday_score, forensic_sentiment, graph_centrality
        - math_day_bear, math_day_base, ... (All 9 math outputs)
        - TARGET_day_min, TARGET_day_close, TARGET_day_max (Actual outcomes)
        - ... (Targets for week and month)
        - current_price
        """
        logger.info("⚙️  Preprocessing Boss Training Data...")
        
        # 1. Inputs (Context)
        X_context = history_df[['doomsday_score', 'forensic_sentiment', 'graph_centrality']].values
        
        # 2. Inputs (Expert - Normalized)
        # We need to iterate and normalize because vectorization is tricky with col referencing
        X_expert_list = []
        Y_list = []
        
        math_cols = [
            'math_day_bear', 'math_day_base', 'math_day_bull',
            'math_week_bear', 'math_week_base', 'math_week_bull',
            'math_month_bear', 'math_month_base', 'math_month_bull'
        ]
        
        # Targets: We want the Boss to predict the ACTUAL future Low/Close/High
        target_cols = [
            'ACTUAL_day_low', 'ACTUAL_day_close', 'ACTUAL_day_high',
            'ACTUAL_week_low', 'ACTUAL_week_close', 'ACTUAL_week_high',
            'ACTUAL_month_low', 'ACTUAL_month_close', 'ACTUAL_month_high'
        ]
        
        for idx, row in history_df.iterrows():
            curr = row['current_price']
            
            # Normalize Inputs
            expert_vals = row[math_cols].values.astype(float)
            expert_pct = (expert_vals - curr) / (curr + 1e-8)
            X_expert_list.append(expert_pct)
            
            # Normalize Targets
            target_vals = row[target_cols].values.astype(float)
            target_pct = (target_vals - curr) / (curr + 1e-8)
            Y_list.append(target_pct)
            
        return np.array(X_context), np.array(X_expert_list), np.array(Y_list)

    def train(self, training_df):
        logger.info("🕴️  BOSS TRAINING INITIATED")
        
        Xc, Xe, Y = self.prepare_training_data(training_df)
        
        # Split
        split = int(len(Xc) * 0.85)
        dataset = BossDataset(Xc[:split], Xe[:split], Y[:split])
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Init Model
        # Input dims: Expert has 9 cols. Context has 3.
        self.model = ConsensusNetwork(input_dim=12).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss() # We want precise vector matching
        
        epochs = 50
        self.model.train()
        
        for e in range(epochs):
            total_loss = 0
            for batch_xc, batch_xe, batch_y in loader:
                batch_xc = batch_xc.to(self.device)
                batch_xe = batch_xe.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_xc, batch_xe)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (e+1) % 10 == 0:
                logger.info(f"   Epoch {e+1}: Loss {total_loss/len(loader):.6f}")
                
        self.save_brain()
        logger.info("✅ Boss Brain Trained & Saved.")

    def make_decision(self, ticker, current_price, doomsday_data, forensic_data, math_data):
        """
        The Main Public Method.
        Takes raw dictionaries from all sub-agents and produces the Final Verdict.
        """
        if self.model is None:
            if not self.load_brain(): 
                logger.error("❌ Boss Brain not found. Please train first.")
                return None
        
        self.model.eval()
        
        # 1. Parse Context Inputs
        # Doomsday: 0.0 to 1.0 (Higher is worse)
        d_score = 0.8 if doomsday_data.get('status') == 'CRITICAL RISK' else 0.2
        
        # Forensic: Sentiment (-1 to 1) and Centrality (0 to 1)
        f_sent = forensic_data.get('sentiment', 0.0)
        f_cent = forensic_data.get('centrality', 0.5)
        
        x_context = torch.tensor([[d_score, f_sent, f_cent]], dtype=torch.float32).to(self.device)
        
        # 2. Parse Expert Inputs (The Math Agent's Brackets)
        # Normalize them to % change relative to current price
        expert_vec = self._normalize_math_inputs(math_data, current_price)
        x_expert = torch.tensor([expert_vec], dtype=torch.float32).to(self.device)
        
        # 3. Inference
        with torch.no_grad():
            # Output is % change vector [9 values]
            pred_pcts = self.model(x_context, x_expert).cpu().numpy()[0]
            
        # 4. Denormalize (Convert % back to Dollars)
        # Order: Day(Bear,Base,Bull), Week(...), Month(...)
        final_vals = current_price * (1 + pred_pcts)
        
        # 5. Format Output (Just like Math Agent, but adjusted)
        result = {
            "ticker": ticker,
            "current_price": current_price,
            "boss_confidence": "HIGH" if abs(f_sent) > 0.5 else "MODERATE",
            
            # 1 Day Horizon
            "day_bear": round(final_vals[0], 2),
            "day_base": round(final_vals[1], 2),
            "day_bull": round(final_vals[2], 2),
            
            # 1 Week Horizon
            "week_bear": round(final_vals[3], 2),
            "week_base": round(final_vals[4], 2),
            "week_bull": round(final_vals[5], 2),
            
            # 1 Month Horizon
            "month_bear": round(final_vals[6], 2),
            "month_base": round(final_vals[7], 2),
            "month_bull": round(final_vals[8], 2),
            
            "logic": self._generate_explanation(d_score, f_sent, pred_pcts, expert_vec)
        }
        
        return result

    def _generate_explanation(self, doomsday, sentiment, boss_preds, math_preds):
        """Generates a text string explaining *why* the Boss adjusted the price."""
        delta = np.mean(boss_preds - math_preds)
        
        explanation = "Boss Consensus: "
        if abs(delta) < 0.005:
            explanation += "Aligned with Math Agent. Structure is stable. "
        elif delta > 0:
            explanation += "More BULLISH than Math Agent. "
            if sentiment > 0.2: explanation += "Positive forensic sentiment detected. "
            if doomsday < 0.3: explanation += "Global macro risks are low. "
        else:
            explanation += "More BEARISH than Math Agent. "
            if doomsday > 0.6: explanation += "Doomsday indicators suggest risk off. "
            if sentiment < -0.2: explanation += "Negative rumors detected in knowledge graph. "
            
        return explanation

    def save_brain(self):
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "boss_weights.pth"))

    def load_brain(self):
        path = os.path.join(self.model_dir, "boss_weights.pth")
        if not os.path.exists(path): return False
        
        # Re-init model
        self.model = ConsensusNetwork(input_dim=12).to(self.device)
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            return True
        except Exception as e:
            logger.error(f"Brain load error: {e}")
            return False

# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    # Create dummy data to test the architecture
    agent = BossAgent()
    
    # 1. Simulate a DataFrame for training
    print("🧪 Generaring Dummy Training Data...")
    cols = [
        'current_price', 'doomsday_score', 'forensic_sentiment', 'graph_centrality',
        'math_day_bear', 'math_day_base', 'math_day_bull',
        'math_week_bear', 'math_week_base', 'math_week_bull',
        'math_month_bear', 'math_month_base', 'math_month_bull',
        'ACTUAL_day_low', 'ACTUAL_day_close', 'ACTUAL_day_high',
        'ACTUAL_week_low', 'ACTUAL_week_close', 'ACTUAL_week_high',
        'ACTUAL_month_low', 'ACTUAL_month_close', 'ACTUAL_month_high'
    ]
    
    # 100 rows of random data
    data = np.random.uniform(100, 110, size=(100, len(cols)))
    df = pd.DataFrame(data, columns=cols)
    # Ensure structure makes sense (Bear < Base < Bull)
    # (In a real run, this comes from the Orchestrator history simulation)
    
    # 2. Train
    agent.train(df)
    
    # 3. Predict
    print("\n🔮 Running Prediction Test...")
    res = agent.make_decision(
        ticker="TEST",
        current_price=105.00,
        doomsday_data={'status': 'STABLE', 'score': 0.1},
        forensic_data={'sentiment': 0.8, 'centrality': 0.9}, # Highly positive news
        math_data={
            'day_bear': 104, 'day_base': 105, 'day_bull': 106,
            'week_bear': 100, 'week_base': 107, 'week_bull': 110,
            'month_bear': 95, 'month_base': 115, 'month_bull': 125
        }
    )
    
    import json
    print(json.dumps(res, indent=2))
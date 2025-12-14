import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

# --- Project Imports ---
# We assume math_tools.py is in src/utils/
try:
    from src.utils.math_tools import FinancialMath
    from src.settings import doomsday_config as config # Re-using config from previous step
except ImportError:
    # Fallback for local testing
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.utils.math_tools import FinancialMath
    import config # You might need a dummy config if testing isolated

# ==========================================
# 1. THE LOSS FUNCTION (Quantile Regression)
# ==========================================
class RiskQuantileLoss(nn.Module):
    """
    We don't just want to know the 'average' outcome.
    We want to know the Worst Case Scenario (95th Percentile).
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.95]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        loss = 0
        target = target.unsqueeze(-1)
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i].unsqueeze(-1)
            loss += torch.max((q - 1) * errors, q * errors).mean()
        return loss

# ==========================================
# 2. THE BRAIN (Transformer)
# ==========================================
class DoomsdayTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2):
        super(DoomsdayTransformer, self).__init__()
        
        # 1. Embed Input Features
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model)) # Max seq len 100
        
        # 2. Transformer Encoder (Finds patterns in time)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Output Heads (Predicting different Timeframes)
        # Output shape: 3 values (Low, Median, High Risk)
        self.head_short_term = nn.Linear(d_model, 3) # 1 Week out
        self.head_med_term   = nn.Linear(d_model, 3) # 1 Month out
        self.head_long_term  = nn.Linear(d_model, 3) # 3 Months out

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features]
        seq_len = x.size(1)
        
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Pass through Transformer
        x = self.transformer(x)
        
        # We only care about the *last* memory state (Today)
        last_state = x[:, -1, :]
        
        # Generate Forecasts
        return torch.stack([
            self.head_short_term(last_state),
            self.head_med_term(last_state),
            self.head_long_term(last_state)
        ], dim=1)

# ==========================================
# 3. DATASET WRAPPER
# ==========================================
class MacroDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================
# 4. THE AGENT (Controller)
# ==========================================
class DoomsdayAgent:
    def __init__(self, model_dir="models/global_event_brain"):
        self.model_dir = model_dir
        self.scaler = RobustScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.LOOKBACK = 30 # Look at past 30 days to predict future
        self.FEATURES_DIM = 0 # Will be set dynamically
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _engineer_features(self, df):
        """
        Takes raw dataframe (Gold, Copper, Oil, Yields, GDELT)
        Applies math_tools to create advanced signal features.
        """
        try:
            df = df.copy().fillna(method='ffill').fillna(0)
            
            # 1. Price/Tone Changes
            df['Gold_Ret'] = np.log(df['Gold'] / df['Gold'].shift(1))
            df['Oil_Ret'] = np.log(df['Oil'] / df['Oil'].shift(1))
            
            # 2. Advanced Math (Using your Utils)
            # RSI on Safe Havens (Is Gold overbought? Panic buying?)
            df['Gold_RSI'] = FinancialMath.get_rsi(df['Gold']) / 100.0
            
            # Volatility (Fear)
            df['VIX_Vol'] = FinancialMath.get_volatility(df['VIX'], window=10)
            
            # Yield Curve (The Recession Signal)
            df['Curve_Spread'] = df['Yield10Y'] - df['Yield3M']
            
            # GDELT Dynamics (Velocity of Bad News)
            df['GDELT_Change'] = df['GDELT_Tone'].diff()
            
            # 3. FFT Energy (Are we in a high-energy chaotic cycle?)
            # We calculate this per window in the loop, but for simple vectorization
            # we can do a rolling std dev as a proxy for energy here
            df['Chaos_Energy'] = df['Gold'].rolling(10).std() * df['VIX']
            
            df = df.replace([np.inf, -np.inf], 0).fillna(0)
            return df
        except Exception as e:
            print(f"Feature Engineering Error: {e}")
            return df

    def prepare_training_data(self, raw_df):
        """
        raw_df must have columns: ['Date', 'Gold', 'Copper', 'Oil', 'Yield10Y', 'Yield3M', 'VIX', 'GDELT_Tone']
        """
        print("⚙️  Preprocessing Global Data...")
        df = self._engineer_features(raw_df)
        
        # Select Features for the Neural Net
        feature_cols = [
            'Gold_Ret', 'Oil_Ret', 'Gold_RSI', 'Curve_Spread', 
            'VIX', 'VIX_Vol', 'GDELT_Tone', 'GDELT_Change'
        ]
        
        data_matrix = df[feature_cols].values
        self.FEATURES_DIM = len(feature_cols)
        
        # Create Windows
        X, Y = [], []
        
        # TARGET: We want to predict FUTURE MAX VIX (Fear) 
        # If model predicts VIX goes to 80, that's a crash.
        target_col_idx = df.columns.get_loc('VIX')
        
        for i in range(self.LOOKBACK, len(df) - 30):
            # Input: Past 30 days
            window = data_matrix[i-self.LOOKBACK : i]
            
            # Target: Max VIX in next 7 days, 30 days, 90 days
            future_7d = df.iloc[i:i+5]['VIX'].max()
            future_30d = df.iloc[i:i+20]['VIX'].max()
            future_90d = df.iloc[i:i+60]['VIX'].max()
            
            X.append(window)
            Y.append([future_7d, future_30d, future_90d])
            
        X = np.array(X)
        Y = np.array(Y)
        
        # Scale Inputs
        N, L, F = X.shape
        X_flat = X.reshape(-1, F)
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(N, L, F)
        
        return X_scaled, Y

    def train(self, historical_df):
        print(f"🚀 Starting Doomsday Agent Training on {self.device}...")
        
        X, Y = self.prepare_training_data(historical_df)
        
        # Split (Train on past, validate on recent past)
        split = int(len(X) * 0.85)
        X_train, X_val = X[:split], X[split:]
        Y_train, Y_val = Y[:split], Y[split:]
        
        dataset = MacroDataset(X_train, Y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize Model
        self.model = DoomsdayTransformer(num_features=self.FEATURES_DIM).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = RiskQuantileLoss(quantiles=[0.1, 0.5, 0.95]) # 10% (Best case), 50% (Avg), 95% (Worst Case)
        
        # Loop
        epochs = 20
        for e in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_x) # [Batch, 3_Horizons, 3_Quantiles]
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (e+1) % 5 == 0:
                print(f"   Epoch {e+1}: Loss {total_loss/len(loader):.4f}")
                
        self.save_brain()
        print("✅ Doomsday Agent Trained & Saved.")

    def predict_doomsday(self, current_df):
        """
        Takes the last 30 days of data and predicts the 'Worst Case' VIX for next week/month.
        """
        if self.model is None:
            if not self.load_brain(): return None
            
        self.model.eval()
        
        # Preprocess
        df = self._engineer_features(current_df)
        feature_cols = [
            'Gold_Ret', 'Oil_Ret', 'Gold_RSI', 'Curve_Spread', 
            'VIX', 'VIX_Vol', 'GDELT_Tone', 'GDELT_Change'
        ]
        
        # Get last window
        raw_window = df[feature_cols].values[-self.LOOKBACK:]
        
        # Scale
        raw_window_flat = raw_window.reshape(-1, self.FEATURES_DIM)
        scaled_window = self.scaler.transform(raw_window_flat).reshape(1, self.LOOKBACK, self.FEATURES_DIM)
        
        input_tensor = torch.tensor(scaled_window, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Preds shape: [1, 3_Horizons, 3_Quantiles]
            preds = self.model(input_tensor).cpu().numpy()[0]
            
        # Extract the "Worst Case" (95th percentile) predictions
        # Index 2 is the 0.95 quantile
        worst_case_week = preds[0][2] 
        worst_case_month = preds[1][2]
        worst_case_quarter = preds[2][2]
        
        return {
            "prediction_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "current_vix": current_df['VIX'].iloc[-1],
            "forecast_vix_max_7d": round(worst_case_week, 2),
            "forecast_vix_max_30d": round(worst_case_month, 2),
            "forecast_vix_max_90d": round(worst_case_quarter, 2),
            "status": "CRITICAL RISK" if worst_case_month > 40 else "STABLE"
        }

    def save_brain(self):
        state = {
            'state_dict': self.model.state_dict(),
            'dims': self.FEATURES_DIM
        }
        torch.save(state, os.path.join(self.model_dir, "doomsday_model.pth"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))

    def load_brain(self):
        path = os.path.join(self.model_dir, "doomsday_model.pth")
        if not os.path.exists(path):
            print("⚠️ No trained brain found.")
            return False
        
        state = torch.load(path, map_location=self.device)
        self.FEATURES_DIM = state['dims']
        self.model = DoomsdayTransformer(num_features=self.FEATURES_DIM).to(self.device)
        self.model.load_state_dict(state['state_dict'])
        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
        return True
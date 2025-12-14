import os
import joblib
import datetime
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

# --- Project Imports ---
try:
    from src.utils.data_miner import DataMiner
    from src.utils.math_tools import GraphMath, FinancialMath
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.utils.data_miner import DataMiner
    from src.utils.math_tools import GraphMath, FinancialMath

# ==========================================
# 1. QUANTILE LOSS FUNCTION (The Cone Logic)
# ==========================================
class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        preds: [Batch, Horizons, 3] (Low, Median, High)
        target: [Batch, Horizons]
        """
        loss = 0
        # Expand target to [Batch, Horizons, 1] for broadcasting
        target = target.unsqueeze(-1)
        
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i].unsqueeze(-1)
            loss += torch.max((q - 1) * errors, q * errors).mean()
            
        return loss

# ==========================================
# 2. DATASET (Kept on CPU for memory)
# ==========================================
class RollingWindowDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.Y = torch.tensor(Y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================
# 3. PROBABILISTIC NEURAL ARCHITECTURE
# ==========================================
class MultiHorizonTransformer(nn.Module):
    def __init__(self, num_features, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super(MultiHorizonTransformer, self).__init__()
        
        # Input Embedding
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))
        
        # Transformer Block
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # Multi-Head Probabilistic Output
        # Each head outputs 3 numbers: [10th percentile, 50th (Median), 90th percentile]
        self.head_day = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 3))
        self.head_week = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 3))
        self.head_month = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 3))
        self.head_year = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        
        # Use final state
        current_state = x[:, -1, :] 
        
        # Stack outputs: [Batch, 4, 3]
        return torch.stack([
            self.head_day(current_state),
            self.head_week(current_state),
            self.head_month(current_state),
            self.head_year(current_state)
        ], dim=1)

# ==========================================
# 4. AGENT CONTROLLER
# ==========================================
class SmartMathAgent:
    def __init__(self, model_dir="models/rolling_brain_v2"):
        self.model_dir = model_dir
        self.log_file = os.path.join(self.model_dir, "agent_log.txt")
        
        # --- GPU ENFORCEMENT ---
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.use_pin_memory = True
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU ACTIVATED: {gpu_name}")
        else:
            self.device = torch.device('cpu')
            self.use_pin_memory = False
            print("⚠️ GPU NOT DETECTED! Training will be slow.")
            
        self.LOOKBACK = 60      
        self.HIDDEN_DIM = 128
        
        # 10 Orig + 1 MACD + 1 ATR + 3 FFT = 15 Features
        self.N_FEATURES = 15 
        
        self.model = None
        self.scaler = RobustScaler()
        self.fundamental_map = {}
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _log(self, msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] 🤖 {msg}")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

    def _engineer_rolling_features(self, df, ticker, fund_map):
        """
        Returns X (Samples, 60, 15) and Y (Samples, 4)
        """
        if len(df) < self.LOOKBACK + 252: return None, None
        
        try:
            df = df.copy()
            
            # --- BASIC FEATURES ---
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            
            vol_mean = df['Volume'].rolling(20).mean()
            vol_std = df['Volume'].rolling(20).std()
            df['Vol_Norm'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)
            
            df['H-L'] = (df['High'] - df['Low']) / df['Close']
            df['RSI'] = FinancialMath.get_rsi(df['Close']) / 100.0
            
            # --- NEW MATH FEATURES ---
            df['MACD'] = FinancialMath.get_macd(df['Close'])
            df['ATR'] = FinancialMath.get_atr(df['High'], df['Low'], df['Close'])
            # Normalize MACD/ATR vaguely to fit network
            df['MACD'] = df['MACD'] / (df['Close'] + 1e-8) 
            df['ATR'] = df['ATR'] / (df['Close'] + 1e-8)

            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Extract columns for sliding window
            # (Note: FFT is calculated per window below)
            static_cols = ['Log_Ret', 'Vol_Norm', 'H-L', 'RSI', 'MACD', 'ATR']
            static_data = df[static_cols].values
            close_prices = df['Close'].values
            fund_vec = fund_map.get(ticker, [0]*6)
            
            X_list, Y_list = [], []
            step = 2 
            
            # Slicing Loop
            for i in range(self.LOOKBACK, len(df) - 252, step):
                # 1. Base Tech Features
                tech_window = static_data[i-self.LOOKBACK : i] # (60, 6)
                
                # 2. FFT Calculation (On the Fly per window)
                # Calculate FFT on the price curve of this specific window
                price_window = close_prices[i-self.LOOKBACK : i]
                fft_energies = FinancialMath.get_fft_energy(price_window, top_k=3) # [e1, e2, e3]
                
                # Tile FFT to match window shape (60, 3)
                fft_matrix = np.tile(fft_energies, (self.LOOKBACK, 1))
                
                # 3. Fundamentals (60, 6)
                fund_window = np.tile(fund_vec, (self.LOOKBACK, 1))
                
                # 4. Combine: 6 (Tech) + 3 (FFT) + 6 (Fund) = 15 Features
                full_window = np.concatenate([tech_window, fft_matrix, fund_window], axis=1)
                
                # Targets (Log Returns)
                p_current = close_prices[i-1]
                r_1d = np.log(close_prices[i] / p_current)
                r_1w = np.log(close_prices[i+4] / p_current)
                r_1m = np.log(close_prices[i+20] / p_current)
                r_1y = np.log(close_prices[i+251] / p_current)
                
                X_list.append(full_window)
                Y_list.append([r_1d, r_1w, r_1m, r_1y])
                
            return np.array(X_list), np.array(Y_list)
            
        except Exception as e:
            return None, None

    def train(self, universe_tickers):
        self._log(f"INITIATING ADVANCED MATH TRAINING for {len(universe_tickers)} assets.")
        
        # 1. Acquire Data
        candidates, fund_map, _ = DataMiner.get_fundamentals_and_text(universe_tickers)
        self.fundamental_map = fund_map
        
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365*12)).strftime('%Y-%m-%d')
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        price_data = DataMiner.get_universe_data(candidates, start_date, end_date)
        
        # 2. Build Memory Bank
        self._log("⚙️  Processing Time Series (Slicing with FFT & MACD)...")
        all_X, all_Y = [], []
        
        for t, df in price_data.items():
            x, y = self._engineer_rolling_features(df, t, fund_map)
            if x is not None and len(x) > 0:
                all_X.append(x)
                all_Y.append(y)
        
        if not all_X:
            self._log("❌ No valid training data generated.")
            return

        X_final = np.concatenate(all_X, axis=0)
        Y_final = np.concatenate(all_Y, axis=0)
        
        self._log(f"📚 Dataset Compiled: {len(X_final)} scenarios.")
        
        # 3. Scaling
        N, L, F = X_final.shape
        X_flat = X_final.reshape(-1, F)
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(N, L, F)
        
        # 4. WALK-FORWARD SPLIT (No Random Shuffle)
        # We split by time index roughly. Since data is concatenated by ticker, 
        # this isn't perfect time-splitting unless we sort by date, 
        # but for a "rolling" agent, splitting the bulk array 90/10 
        # creates a train set and a "unseen" test set.
        
        split_idx = int(N * 0.9)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        Y_train, Y_val = Y_final[:split_idx], Y_final[split_idx:]
        
        train_loader = DataLoader(
            RollingWindowDataset(X_train, Y_train),
            batch_size=1024, 
            shuffle=False,  # <--- IMPORTANT: KEEP ORDER
            pin_memory=self.use_pin_memory
        )
        
        val_loader = DataLoader(
            RollingWindowDataset(X_val, Y_val),
            batch_size=1024,
            shuffle=False
        )
        
        # 5. Model Init (Quantile)
        self.model = MultiHorizonTransformer(num_features=F, d_model=self.HIDDEN_DIM).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0003) # Lower LR for stability
        loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9]) # The Cones
        
        # 6. Training Loop
        self.model.train()
        epochs = 15
        
        self._log(f"🔥 STARTING GPU TRAINING LOOP ({epochs} Epochs)...")
        
        for e in range(epochs):
            total_loss = 0
            count = 0
            
            # Train
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                output = self.model(bx) # [Batch, 4, 3]
                loss = loss_fn(output, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
            
            avg_loss = total_loss / count
            
            # Validate
            self.model.eval()
            val_loss = 0
            val_count = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(self.device), vy.to(self.device)
                    v_out = self.model(vx)
                    val_loss += loss_fn(v_out, vy).item()
                    val_count += 1
            
            print(f"   Epoch {e+1}/{epochs} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss/val_count:.6f}")
                
        self.save_brain()
        self._log("✅ Model Trained and Saved.")

    def predict(self, target_ticker, df):
        if self.model is None:
            if not self.load_brain(): return None
        self.model.eval()
        
        try:
            # Re-create features
            df = df.copy()
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            
            vol_mean = df['Volume'].rolling(20).mean()
            vol_std = df['Volume'].rolling(20).std()
            df['Vol_Norm'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)
            df['H-L'] = (df['High'] - df['Low']) / df['Close']
            df['RSI'] = FinancialMath.get_rsi(df['Close']) / 100.0
            
            df['MACD'] = FinancialMath.get_macd(df['Close'])
            df['ATR'] = FinancialMath.get_atr(df['High'], df['Low'], df['Close'])
            df['MACD'] = df['MACD'] / (df['Close'] + 1e-8)
            df['ATR'] = df['ATR'] / (df['Close'] + 1e-8)
            
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            if len(df) < self.LOOKBACK: return None
            
            # Get last window
            static_cols = ['Log_Ret', 'Vol_Norm', 'H-L', 'RSI', 'MACD', 'ATR']
            tech_data = df[static_cols].values[-self.LOOKBACK:]
            
            # FFT
            price_window = df['Close'].values[-self.LOOKBACK:]
            fft_energies = FinancialMath.get_fft_energy(price_window, top_k=3)
            fft_matrix = np.tile(fft_energies, (self.LOOKBACK, 1))
            
            # Funds
            fund_vec = self.fundamental_map.get(target_ticker, [0]*6)
            curr_fund = np.tile(fund_vec, (self.LOOKBACK, 1))
            
            # Combine
            X_raw = np.concatenate([tech_data, fft_matrix, curr_fund], axis=1)
            
            # Scale
            X_flat = X_raw.reshape(-1, self.N_FEATURES)
            X_scaled = self.scaler.transform(X_flat).reshape(1, self.LOOKBACK, self.N_FEATURES)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                # preds shape: [1, 4, 3]
                preds = self.model(X_tensor).cpu().numpy()[0]
            
            return {
                "ticker": target_ticker,
                "current_price": df['Close'].iloc[-1],
                # 0=Day, 1=Week, 2=Month, 3=Year
                # 0=Bear, 1=Base, 2=Bull
                "day_bear": preds[0][0], "day_base": preds[0][1], "day_bull": preds[0][2],
                "week_bear": preds[1][0], "week_base": preds[1][1], "week_bull": preds[1][2],
                "month_bear": preds[2][0], "month_base": preds[2][1], "month_bull": preds[2][2],
                "year_bear": preds[3][0], "year_base": preds[3][1], "year_bull": preds[3][2],
            }
        except Exception as e:
            self._log(f"Prediction Error: {e}")
            return None

    def save_brain(self):
        state = {
            'state_dict': self.model.state_dict(),
            'fundamental_map': self.fundamental_map,
            'dims': (self.N_FEATURES, self.HIDDEN_DIM)
        }
        torch.save(state, os.path.join(self.model_dir, "model_state.pth"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))

    def load_brain(self):
        try:
            path = os.path.join(self.model_dir, "model_state.pth")
            if not os.path.exists(path): return False
            
            state = torch.load(path, map_location=self.device, weights_only=False)
            
            self.fundamental_map = state['fundamental_map']
            # Re-init model with correct dims
            self.N_FEATURES = state['dims'][0] 
            self.model = MultiHorizonTransformer(num_features=self.N_FEATURES, d_model=state['dims'][1]).to(self.device)
            self.model.load_state_dict(state['state_dict'])
            self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
            return True
        except Exception as e:
            self._log(f"❌ Load Error: {e}")
            return False
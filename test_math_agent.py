import os
import time
import glob
import io
import re
import ssl
import random
import requests
import warnings
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# ==========================================
# 0. CONFIGURATION
# ==========================================
warnings.filterwarnings("ignore")

# SSL Fix for Mac/Linux environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

CONFIG = {
    "LOOKBACK": 252,          # 1 Year Context
    "PREDICT_HORIZONS": [1, 5, 20, 252], # 1d, 1w, 1m, 1y
    "FEATURES": 15,           # Tech(6) + FFT(3) + Fund(6)
    "HIDDEN_DIM": 256,
    "LAYERS": 6,
    "HEADS": 8,
    "DROPOUT": 0.1,
    "BATCH_SIZE": 1024,       # Adjusted for stability
    "EPOCHS": 100,            # 100 is usually sufficient with early stopping
    "LR": 1e-4,               # Lower LR for Transformer stability
    "STRIDE": 1,              # Sample every single day
    "DATA_DIR": "data_sp500_v5",
    "MODEL_DIR": "models_sp500_v5",
}

os.makedirs(f"{CONFIG['DATA_DIR']}/raw", exist_ok=True)
os.makedirs(f"{CONFIG['DATA_DIR']}/processed", exist_ok=True)
os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 DEVICE: {device}")

# ==========================================
# 1. MATH & FEATURES
# ==========================================
class FinancialMath:
    @staticmethod
    def get_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def get_macd(series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return (macd - signal_line).fillna(0)

    @staticmethod
    def get_atr(high, low, close, window=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean().fillna(0)

    @staticmethod
    def get_fft_energy(window_data, top_k=3):
        detrended = window_data - np.mean(window_data)
        fft_vals = np.fft.rfft(detrended)
        fft_mag = np.abs(fft_vals)
        sorted_indices = np.argsort(fft_mag[1:])[::-1]
        energies = []
        for i in range(top_k):
            if i < len(sorted_indices):
                energies.append(fft_mag[sorted_indices[i] + 1])
            else:
                energies.append(0.0)
        return energies

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================
class WeightedQuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9], weights=[1.0, 0.8, 0.5, 0.2]):
        super().__init__()
        self.quantiles = quantiles
        self.weights = torch.tensor(weights).to(device)

    def forward(self, preds, target):
        # preds: [Batch, 4, 3]
        # target: [Batch, 4]
        loss = 0
        target = target.unsqueeze(-1) # [Batch, 4, 1]
        
        # Calculate loss for each horizon
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i].unsqueeze(-1)
            q_loss = torch.max((q - 1) * errors, q * errors) # [Batch, 4, 1]
            loss += q_loss.mean(dim=0).squeeze() # [4] (Average over batch)
            
        # Weighted sum across horizons (Day > Week > Month > Year)
        # We sum the quantile losses (3 values) then weight by horizon
        return (loss * self.weights).sum()

class MultiHorizonTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # 4 Heads: Day, Week, Month, Year
        # Output 3 values per head (Bear, Median, Bull)
        self.head_day = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 3))
        self.head_week = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 3))
        self.head_month = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 3))
        self.head_year = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.input_proj(x) + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        current_state = x[:, -1, :] 
        return torch.stack([
            self.head_day(current_state),
            self.head_week(current_state),
            self.head_month(current_state),
            self.head_year(current_state)
        ], dim=1)

class PreloadedDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.X = torch.tensor(x_data, dtype=torch.float32)
        self.Y = torch.tensor(y_data, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ==========================================
# 3. PIPELINE PHASES
# ==========================================

def phase_1_download():
    print("\n📦 PHASE 1: ACQUIRING DATA")
    tickers = []
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        df_list = pd.read_html(io.StringIO(requests.get(url, headers=headers).text))
        tickers = [re.sub(r'[^A-Z0-9-]', '', t.upper().replace('.', '-')) for t in df_list[0]['Symbol']]
    except:
        tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "SPY", "QQQ"]

    print(f"   Found {len(tickers)} tickers. Downloading...")
    
    chunk_size = 25
    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i+chunk_size]
        try:
            data = yf.download(batch, start="2000-01-01", progress=False, group_by='ticker', auto_adjust=True, threads=True)
            for t in batch:
                try:
                    if len(batch) > 1:
                        df = data[t].copy()
                    else:
                        df = data.copy()
                        
                    if 'Close' not in df.columns: continue
                    df = df.dropna(subset=['Close'])
                    
                    if len(df) > CONFIG['LOOKBACK'] + 300:
                        df.to_parquet(f"{CONFIG['DATA_DIR']}/raw/{t}.parquet")
                except: continue
        except: time.sleep(1)
    print("   ✅ Data Download Complete.")

def phase_2_process_corrected():
    print("\n⚙️ PHASE 2: PROCESSING & CHRONOLOGICAL SPLITTING")
    
    files = glob.glob(f"{CONFIG['DATA_DIR']}/raw/*.parquet")
    
    # Lists to hold data BEFORE concatenation
    train_X_list, train_Y_list = [], []
    val_X_list, val_Y_list = [], []
    
    print(f"   Processing {len(files)} files...")
    
    for f in tqdm(files):
        try:
            df = pd.read_parquet(f)
            df = df.ffill().dropna()
            if len(df) < CONFIG['LOOKBACK'] + 252: continue
            
            # --- Short Term Features ---
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Vol_Norm'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
            df['H-L'] = (df['High'] - df['Low']) / df['Close']
            df['RSI'] = FinancialMath.get_rsi(df['Close']) / 100.0
            df['MACD'] = FinancialMath.get_macd(df['Close']) / (df['Close'] + 1e-8)
            df['ATR'] = FinancialMath.get_atr(df['High'], df['Low'], df['Close']) / (df['Close'] + 1e-8)
            
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Feature Cols (6 features)
            feature_cols = ['Log_Ret', 'Vol_Norm', 'H-L', 'RSI', 'MACD', 'ATR']
            data_tech = df[feature_cols].values.astype(np.float32)
            data_close = df['Close'].values.astype(np.float32)
            
            # Fake Fundamentals (6 features)
            fund_vec = np.array([20.0, 3.0, 0.5, 0.15, 25.0, 1.0], dtype=np.float32)
            
            L = CONFIG['LOOKBACK']
            # Limit so we can have targets for 1 Year out (252 days)
            limit = len(df) - 252 
            
            # Temp storage for this ticker
            ticker_X = []
            ticker_Y = []
            
            # --- SLICING LOOP ---
            for i in range(L, limit, CONFIG['STRIDE']):
                
                # Inputs
                tech_win = data_tech[i-L : i] # (L, 6)
                
                # FFT on Price (Calculated per window to avoid lookahead)
                price_win = data_close[i-L : i]
                energies = FinancialMath.get_fft_energy(price_win, top_k=3)
                fft_win = np.tile(energies, (L, 1)).astype(np.float32) # (L, 3)
                
                # Fundamentals
                fund_win = np.tile(fund_vec, (L, 1)).astype(np.float32) # (L, 6)
                
                # Concatenate: 6 + 3 + 6 = 15 Features
                full_win = np.concatenate([tech_win, fft_win, fund_win], axis=1)
                
                # Targets (Log Returns)
                p_cur = data_close[i-1]
                t_1d = np.log(data_close[i] / p_cur)
                t_1w = np.log(data_close[i+4] / p_cur)
                t_1m = np.log(data_close[i+20] / p_cur)
                t_1y = np.log(data_close[i+251] / p_cur)
                
                ticker_X.append(full_win)
                ticker_Y.append([t_1d, t_1w, t_1m, t_1y])
            
            # --- CHRONOLOGICAL SPLIT PER TICKER ---
            # 90% Training, 10% Validation (Validation is strictly the end of the chart)
            if len(ticker_X) > 10:
                split_idx = int(len(ticker_X) * 0.90)
                
                train_X_list.append(np.array(ticker_X[:split_idx]))
                train_Y_list.append(np.array(ticker_Y[:split_idx]))
                
                val_X_list.append(np.array(ticker_X[split_idx:]))
                val_Y_list.append(np.array(ticker_Y[split_idx:]))

        except Exception as e:
            continue

    if len(train_X_list) == 0: 
        print("❌ No data processed.")
        return False

    print("   Merging data arrays...")
    X_train = np.concatenate(train_X_list, axis=0)
    Y_train = np.concatenate(train_Y_list, axis=0)
    X_val = np.concatenate(val_X_list, axis=0)
    Y_val = np.concatenate(val_Y_list, axis=0)
    
    print(f"   Training Samples: {len(X_train)}")
    print(f"   Validation Samples: {len(X_val)}")
    
    # --- FIT SCALER ON TRAIN ONLY ---
    print("   Fitting Scaler (No Leakage)...")
    scaler = RobustScaler()
    N_t, L, F = X_train.shape
    
    # Flatten -> Fit -> Transform
    X_train_flat = X_train.reshape(-1, F)
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(N_t, L, F)
    
    # Transform Validation using Train stats
    N_v, _, _ = X_val.shape
    X_val_flat = X_val.reshape(-1, F)
    X_val_scaled = scaler.transform(X_val_flat).reshape(N_v, L, F)
    
    # Save
    print("   Saving processed files...")
    np.save(f"{CONFIG['DATA_DIR']}/processed/X_train.npy", X_train_scaled)
    np.save(f"{CONFIG['DATA_DIR']}/processed/Y_train.npy", Y_train)
    np.save(f"{CONFIG['DATA_DIR']}/processed/X_val.npy", X_val_scaled)
    np.save(f"{CONFIG['DATA_DIR']}/processed/Y_val.npy", Y_val)
    
    joblib.dump(scaler, f"{CONFIG['MODEL_DIR']}/scaler.pkl")
    
    print("✅ Data Processing Complete.")
    return True

def phase_3_train():
    print(f"\n🔥 PHASE 3: TRAINING ({CONFIG['EPOCHS']} EPOCHS)")
    
    # Load separate files
    print("   Loading Datasets...")
    X_train = np.load(f"{CONFIG['DATA_DIR']}/processed/X_train.npy")
    Y_train = np.load(f"{CONFIG['DATA_DIR']}/processed/Y_train.npy")
    X_val = np.load(f"{CONFIG['DATA_DIR']}/processed/X_val.npy")
    Y_val = np.load(f"{CONFIG['DATA_DIR']}/processed/Y_val.npy")
    
    train_ds = PreloadedDataset(X_train, Y_train)
    val_ds = PreloadedDataset(X_val, Y_val)
    
    # SHUFFLE TRAIN = OK (because validation is chronologically separated)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, pin_memory=True)
    # SHUFFLE VAL = FALSE (keep order for visualization/logic)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)
    
    model = MultiHorizonTransformer(
        CONFIG['FEATURES'], 
        CONFIG['HIDDEN_DIM'], 
        CONFIG['HEADS'], 
        CONFIG['LAYERS'], 
        CONFIG['DROPOUT']
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Weighted loss to balance 1D vs 1Y importance
    loss_fn = WeightedQuantileLoss()
    
    best_loss = float('inf')
    early_stop_counter = 0
    patience = 10
    
    # Watcher Sample (First element of validation set)
    watch_x, watch_y = val_ds[0]
    watch_x = watch_x.unsqueeze(0).to(device)
    watch_y = watch_y.numpy()

    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        train_loss = 0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{CONFIG['EPOCHS']}")
        for bx, by in pbar:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = loss_fn(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            steps += 1
            pbar.set_postfix({'loss': f"{train_loss/steps:.4f}"})
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += loss_fn(model(bx), by).item()
                val_steps += 1
                
            # Quick visual check on watcher
            w_out = model(watch_x).cpu().numpy()[0]
            # Print 1M (Index 2) prediction vs reality
            # w_out shape: [4 horizons, 3 quantiles]
            # bear, med, bull
            real_1m = np.exp(watch_y[2]) * 100
            pred_1m = np.exp(w_out[2][1]) * 100
            # print(f"      [WATCHER 1M] Real: ${real_1m:.2f} | Pred: ${pred_1m:.2f}")

        avg_val = val_loss / val_steps
        scheduler.step(avg_val)
        
        print(f"      Val Loss: {avg_val:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val < best_loss:
            best_loss = avg_val
            early_stop_counter = 0
            torch.save(model.state_dict(), f"{CONFIG['MODEL_DIR']}/best_model.pth")
            print("      💾 Model Saved.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

    print("✅ Training Complete.")

def predict_full_cones(ticker):
    model_path = f"{CONFIG['MODEL_DIR']}/best_model.pth"
    if not os.path.exists(model_path): return
    
    model = MultiHorizonTransformer(CONFIG['FEATURES'], CONFIG['HIDDEN_DIM'], CONFIG['HEADS'], CONFIG['LAYERS'], CONFIG['DROPOUT']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Scaler
    scaler = joblib.load(f"{CONFIG['MODEL_DIR']}/scaler.pkl")
    
    print(f"\n🔮 PREDICTING FOR: {ticker}")
    try:
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(ticker, axis=1, level=1)
            except: df.columns = df.columns.droplevel(1)
            
        # Feature Engineering (Must match Phase 2)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_Norm'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
        df['H-L'] = (df['High'] - df['Low']) / df['Close']
        df['RSI'] = FinancialMath.get_rsi(df['Close']) / 100.0
        df['MACD'] = FinancialMath.get_macd(df['Close']) / (df['Close'] + 1e-8)
        df['ATR'] = FinancialMath.get_atr(df['High'], df['Low'], df['Close']) / (df['Close'] + 1e-8)
        
        df = df.fillna(0)
        
        if len(df) < CONFIG['LOOKBACK']: 
            print("Insufficient data.")
            return

        # Prepare Window
        feature_cols = ['Log_Ret', 'Vol_Norm', 'H-L', 'RSI', 'MACD', 'ATR']
        tech_win = df[feature_cols].values[-CONFIG['LOOKBACK']:].astype(np.float32)
        
        price_win = df['Close'].values[-CONFIG['LOOKBACK']:]
        energies = FinancialMath.get_fft_energy(price_win, top_k=3)
        fft_win = np.tile(energies, (CONFIG['LOOKBACK'], 1)).astype(np.float32)
        
        fund_vec = np.array([20,3,0.5,0.15,25,1], dtype=np.float32)
        fund_win = np.tile(fund_vec, (CONFIG['LOOKBACK'], 1))
        
        X_raw = np.concatenate([tech_win, fft_win, fund_win], axis=1)
        
        # Scale
        X_flat = X_raw.reshape(-1, CONFIG['FEATURES'])
        X_scaled = scaler.transform(X_flat).reshape(1, CONFIG['LOOKBACK'], CONFIG['FEATURES'])
        
        with torch.no_grad():
            preds = model(torch.tensor(X_scaled).to(device)).cpu().numpy()[0]
            
        curr = df['Close'].iloc[-1]
        
        print("-" * 65)
        print(f"📊 {ticker} | CURRENT: ${curr:.2f}")
        print("-" * 65)
        print(f"{'HORIZON':<10} | {'BEAR':<12} | {'MEDIAN':<12} | {'BULL':<12} | {'SPREAD':<10}")
        print("-" * 65)
        
        horizons = ["1 Day", "1 Week", "1 Month", "1 Year"]
        for i, h in enumerate(horizons):
            bear = curr * np.exp(preds[i][0])
            med  = curr * np.exp(preds[i][1])
            bull = curr * np.exp(preds[i][2])
            spread = ((bull - bear) / med) * 100
            print(f"{h:<10} | ${bear:<11.2f} | ${med:<11.2f} | ${bull:<11.2f} | {spread:.1f}%")
        print("-" * 65)
        
    except Exception as e:
        print(f"Prediction error: {e}")

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"🤖 AGENT V5 (STRICT CHRONOLOGICAL SPLIT) | Context: {CONFIG['LOOKBACK']} days")
    
    # 1. Check/Download
    raw_files = glob.glob(f"{CONFIG['DATA_DIR']}/raw/*.parquet")
    if len(raw_files) < 10:
        phase_1_download()
        
    # 2. Process (Force re-run if safe files don't exist)
    if not os.path.exists(f"{CONFIG['DATA_DIR']}/processed/X_train.npy"):
        phase_2_process_corrected()
    
    # 3. Train
    phase_3_train()
    
    # 4. Predict
    predict_full_cones("NVDA")
    predict_full_cones("SPY")
import os
import time
import glob
import io
import re
import ssl
import random
import requests
import warnings
import traceback
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
    "FEATURES": 18,           # Tech(6) + FFT(3) + Fund(6) + LongTerm(3)
    "HIDDEN_DIM": 256,
    "LAYERS": 6,
    "HEADS": 8,
    "DROPOUT": 0.1,
    "BATCH_SIZE": 2048,
    "EPOCHS": 500,            # High ceiling for deep learning
    "LR": 3e-4,
    "STRIDE": 1,              # Sample every single day
    "DATA_DIR": "data_sp500_v4",
    "MODEL_DIR": "models_sp500_v4",
}

os.makedirs(f"{CONFIG['DATA_DIR']}/raw", exist_ok=True)
os.makedirs(f"{CONFIG['DATA_DIR']}/processed", exist_ok=True)
os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        loss = 0
        target = target.unsqueeze(-1)
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i].unsqueeze(-1)
            loss += torch.max((q - 1) * errors, q * errors).mean()
        return loss

class MultiHorizonTransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # 4 Heads: Day, Week, Month, Year
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

class BigDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path, mmap_mode='r')
        self.Y = np.load(y_path, mmap_mode='r')
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

# ==========================================
# 3. ADVANCED LOGGING (THE WATCHER)
# ==========================================
class TrainingWatcher:
    def __init__(self, filepath):
        self.filepath = filepath
        
        # Dynamic Column Generation
        base_cols = ["Epoch", "Train_Loss", "Val_Loss"]
        horizons = ["1D", "1W", "1M", "1Y"]
        metrics = ["Real", "Bear", "Med", "Bull"] # Real = Ground Truth
        
        # Generates: 1D_Real, 1D_Bear, 1D_Med, 1D_Bull, 1W_Real...
        csv_cols = base_cols + [f"{h}_{m}" for h in horizons for m in metrics]
        
        with open(self.filepath, "w") as f:
            f.write(",".join(csv_cols) + "\n")
            
    def log(self, epoch, train_loss, val_loss, preds, true_targets, ref_price=100.0):
        """
        Logs the specific performance on the Watcher Sample.
        All prices are normalized to start at $100.00 for easy percentage comparison.
        """
        row = [f"{epoch}", f"{train_loss:.5f}", f"{val_loss:.5f}"]
        
        # Loop through horizons: 0=1D, 1=1W, 2=1M, 3=1Y
        for i in range(4):
            # 1. Calculate Real Price (Ground Truth)
            # Y contains Log Returns. P_real = $100 * exp(True_Log_Return)
            real_p = ref_price * np.exp(true_targets[i])
            
            # 2. Calculate Model Cone
            bear_p = ref_price * np.exp(preds[i][0])
            med_p  = ref_price * np.exp(preds[i][1])
            bull_p = ref_price * np.exp(preds[i][2])
            
            # 3. Add to row
            row.extend([f"{real_p:.2f}", f"{bear_p:.2f}", f"{med_p:.2f}", f"{bull_p:.2f}"])
            
        with open(self.filepath, "a") as f:
            f.write(",".join(row) + "\n")

# ==========================================
# 4. PIPELINE PHASES
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
            # Download 20 years of data
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

def phase_2_process():
    print("\n⚙️ PHASE 2: DEEP FEATURE ENGINEERING")
    files = glob.glob(f"{CONFIG['DATA_DIR']}/raw/*.parquet")
    scaler = RobustScaler()
    X_buffer, Y_buffer = [], []
    
    print(f"   Processing {len(files)} files with STRIDE={CONFIG['STRIDE']} (Max Density)...")
    
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
            
            # --- Long Term Memory Features ---
            df['SMA_200'] = df['Close'].rolling(200).mean()
            df['Dist_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
            
            df['Rolling_Max'] = df['Close'].expanding().max()
            df['Dist_ATH'] = (df['Close'] - df['Rolling_Max']) / df['Rolling_Max']
            
            df['Vol_1Y'] = df['Log_Ret'].rolling(252).std()

            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Feature Cols
            feature_cols = ['Log_Ret', 'Vol_Norm', 'H-L', 'RSI', 'MACD', 'ATR', 'Dist_SMA200', 'Dist_ATH', 'Vol_1Y']
            data_tech = df[feature_cols].values
            data_close = df['Close'].values
            
            # Placeholder Fundamentals
            fund_vec = [20.0, 3.0, 0.5, 0.15, 25.0, 1.0] 
            
            L = CONFIG['LOOKBACK']
            limit = len(df) - 252 
            
            # --- SLICING LOOP ---
            for i in range(L, limit, CONFIG['STRIDE']):
                
                # Inputs
                tech_win = data_tech[i-L : i]
                price_win = data_close[i-L : i]
                energies = FinancialMath.get_fft_energy(price_win, top_k=3)
                fft_win = np.tile(energies, (L, 1))
                fund_win = np.tile(fund_vec, (L, 1))
                full_win = np.concatenate([tech_win, fft_win, fund_win], axis=1)
                
                # Targets (Log Returns)
                p_cur = data_close[i-1]
                t_1d = np.log(data_close[i] / p_cur)
                t_1w = np.log(data_close[i+4] / p_cur)
                t_1m = np.log(data_close[i+20] / p_cur)
                t_1y = np.log(data_close[i+251] / p_cur)
                
                X_buffer.append(full_win)
                Y_buffer.append([t_1d, t_1w, t_1m, t_1y])

        except Exception: continue

    if len(X_buffer) == 0: return False

    print("   Concatenating & Scaling...")
    X_final = np.array(X_buffer, dtype=np.float32)
    Y_final = np.array(Y_buffer, dtype=np.float32)
    
    N, L, F = X_final.shape
    X_flat = X_final.reshape(-1, F)
    X_scaled = scaler.fit_transform(X_flat).reshape(N, L, F)
    
    np.save(f"{CONFIG['DATA_DIR']}/processed/X.npy", X_scaled)
    np.save(f"{CONFIG['DATA_DIR']}/processed/Y.npy", Y_final)
    joblib.dump(scaler, f"{CONFIG['MODEL_DIR']}/scaler.pkl")
    
    print(f"✅ Saved {N} samples.")
    return True

def phase_3_train():
    print(f"\n🔥 PHASE 3: TRAINING ({CONFIG['EPOCHS']} EPOCHS)")
    
    # Load Data
    ds = BigDataset(f"{CONFIG['DATA_DIR']}/processed/X.npy", f"{CONFIG['DATA_DIR']}/processed/Y.npy")
    train_len = int(0.9 * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'])
    
    # Init Model
    model = MultiHorizonTransformer(CONFIG['FEATURES'], CONFIG['HIDDEN_DIM'], CONFIG['HEADS'], CONFIG['LAYERS'], CONFIG['DROPOUT']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    loss_fn = QuantileLoss()
    
    # Init Logger
    watcher = TrainingWatcher(f"{CONFIG['MODEL_DIR']}/training_log.csv")
    
    # --- SETUP WATCHER SAMPLE ---
    # We grab the first sample from validation to "watch" deeply
    # watch_x = Input features, watch_y = True future returns
    watch_x, watch_y = val_ds[0]
    watch_x = watch_x.unsqueeze(0).to(device)
    watch_y_np = watch_y.numpy() # Move truth to CPU for logging
    
    best_loss = float('inf')
    early_stop_counter = 0
    patience_limit = 20  # Increased patience for long training
    
    print(f"   📝 Detailed logging active: {CONFIG['MODEL_DIR']}/training_log.csv")

    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        train_loss = 0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
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
            
            # --- LOGGING ---
            # Predict on the specific watcher sample
            w_out = model(watch_x).cpu().numpy()[0]
            
            # Log Real vs Predicted for all 4 horizons
            watcher.log(
                epoch=epoch+1, 
                train_loss=train_loss/steps, 
                val_loss=val_loss/val_steps, 
                preds=w_out, 
                true_targets=watch_y_np, 
                ref_price=100.0
            )

        avg_val = val_loss / val_steps
        scheduler.step(avg_val)
        
        print(f"      Val Loss: {avg_val:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Checkpoint
        if avg_val < best_loss:
            best_loss = avg_val
            early_stop_counter = 0
            torch.save(model.state_dict(), f"{CONFIG['MODEL_DIR']}/best_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience_limit:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break

    print("✅ Training Complete.")

def predict_full_cones(ticker):
    model_path = f"{CONFIG['MODEL_DIR']}/best_model.pth"
    if not os.path.exists(model_path): return
    
    model = MultiHorizonTransformer(CONFIG['FEATURES'], CONFIG['HIDDEN_DIM'], CONFIG['HEADS'], CONFIG['LAYERS'], CONFIG['DROPOUT']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(f"{CONFIG['MODEL_DIR']}/scaler.pkl")
    
    print(f"\n🔮 PREDICTING FOR: {ticker}")
    df = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        try: df = df.xs(ticker, axis=1, level=1)
        except: df.columns = df.columns.droplevel(1)
            
    # Re-Engineer Features
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Norm'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
    df['H-L'] = (df['High'] - df['Low']) / df['Close']
    df['RSI'] = FinancialMath.get_rsi(df['Close']) / 100.0
    df['MACD'] = FinancialMath.get_macd(df['Close']) / (df['Close'] + 1e-8)
    df['ATR'] = FinancialMath.get_atr(df['High'], df['Low'], df['Close']) / (df['Close'] + 1e-8)
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['Dist_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
    df['Rolling_Max'] = df['Close'].expanding().max()
    df['Dist_ATH'] = (df['Close'] - df['Rolling_Max']) / df['Rolling_Max']
    df['Vol_1Y'] = df['Log_Ret'].rolling(252).std()
    
    df = df.fillna(0)
    
    if len(df) < CONFIG['LOOKBACK']: return "Insufficient Data"

    # Extract Window
    feature_cols = ['Log_Ret', 'Vol_Norm', 'H-L', 'RSI', 'MACD', 'ATR', 'Dist_SMA200', 'Dist_ATH', 'Vol_1Y']
    tech_win = df[feature_cols].values[-CONFIG['LOOKBACK']:]
    
    price_win = df['Close'].values[-CONFIG['LOOKBACK']:]
    energies = FinancialMath.get_fft_energy(price_win, top_k=3)
    fft_win = np.tile(energies, (CONFIG['LOOKBACK'], 1))
    fund_win = np.tile([20,3,0.5,0.15,25,1], (CONFIG['LOOKBACK'], 1))
    
    X_raw = np.concatenate([tech_win, fft_win, fund_win], axis=1)
    X_scaled = scaler.transform(X_raw.reshape(-1, CONFIG['FEATURES'])).reshape(1, CONFIG['LOOKBACK'], CONFIG['FEATURES'])
    
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()[0]
        
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

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"🤖 AGENT V4 REBOOT | Context: 1 Year ({CONFIG['LOOKBACK']} days)")
    
    # 1. Check/Download
    raw_files = glob.glob(f"{CONFIG['DATA_DIR']}/raw/*.parquet")
    if len(raw_files) < 10:
        phase_1_download()
        
    # 2. Process
    if not os.path.exists(f"{CONFIG['DATA_DIR']}/processed/X.npy"):
        phase_2_process()
    
    # 3. Train
    phase_3_train()
    
    # 4. Predict
    predict_full_cones("NVDA")
    predict_full_cones("SPY")
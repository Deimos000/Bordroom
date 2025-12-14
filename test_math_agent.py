import os
import time
import datetime
import random
import warnings
import ssl
import glob
import io
import requests
import re
import traceback  # Added for detailed error logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib
import yfinance as yf
from tqdm import tqdm

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
warnings.filterwarnings("ignore")

# Fix SSL context
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

CONFIG = {
    "LOOKBACK": 60,
    "FEATURES": 15,
    "HIDDEN_DIM": 256,
    "LAYERS": 6,
    "HEADS": 8,
    "DROPOUT": 0.1,
    "BATCH_SIZE": 1024,
    "EPOCHS": 15,
    "LR": 1e-4,
    "DATA_DIR": "data_sp500",
    "MODEL_DIR": "models_sp500"
}

os.makedirs(f"{CONFIG['DATA_DIR']}/raw", exist_ok=True)
os.makedirs(f"{CONFIG['DATA_DIR']}/processed", exist_ok=True)
os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. MATH & LOGIC
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
# 2. NEURAL NETWORK
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
# 3. PIPELINE PHASES
# ==========================================

def phase_1_download():
    """Robust S&P 500 Downloader with EXTREME LOGGING"""
    print("\n📦 PHASE 1: ACQUIRING DATA (DEBUG MODE)")
    
    # 1. Fetch Ticker List
    tickers = []
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"} 
        response = requests.get(url, headers=headers)
        df_list = pd.read_html(io.StringIO(response.text))
        
        raw_tickers = df_list[0]['Symbol'].tolist()
        
        for t in raw_tickers:
            clean_t = re.sub(r'[^A-Z0-9-]', '', str(t).strip().upper().replace('.', '-'))
            if clean_t:
                tickers.append(clean_t)
                
        print(f"   ✅ Found {len(tickers)} tickers.")
        print(f"   🔎 Sample: {tickers[:5]} ... {tickers[-5:]}")
        
    except Exception as e:
        print(f"   ⚠️ Wiki scrape failed ({e}). Using fallback list.")
        tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"]

    # 2. Download Loop
    chunk_size = 10 # Reduced chunk size for clearer debugging
    downloaded_count = 0
    total_batches = (len(tickers) + chunk_size - 1) // chunk_size
    
    print(f"   Downloading in {total_batches} batches...")
    
    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i+chunk_size]
        batch_idx = i // chunk_size + 1
        
        print(f"\n   ⬇️ BATCH {batch_idx}/{total_batches}: {batch}")
        
        try:
            # Added shared=False to potentially help with map structure in newer yfinance
            data = yf.download(batch, start="2005-01-01", progress=False, auto_adjust=True, threads=True)
            
            if data.empty:
                print(f"      ❌ YF returned EMPTY dataframe for this batch.")
                continue
                
            print(f"      👀 Data Shape: {data.shape}")
            print(f"      👀 Columns Level Count: {data.columns.nlevels}")
            if data.columns.nlevels > 0:
                 print(f"      👀 First 5 Cols: {data.columns[:5].tolist()}")

            # Handle Ticker Logic
            for t in batch:
                failure_reason = "Unknown"
                try:
                    df = pd.DataFrame()
                    
                    # === DEBUGGING THE EXTRACTION LOGIC ===
                    # Case 1: MultiIndex with (Price, Ticker) format [Newer YF defaults sometimes]
                    # Case 2: MultiIndex with (Ticker, Price) format [Older YF]
                    # Case 3: Flat Index (Single ticker or auto-flattened)
                    
                    is_multi = isinstance(data.columns, pd.MultiIndex)
                    
                    if is_multi:
                        # Try to find ticker in level 1
                        if data.columns.nlevels > 1 and t in data.columns.get_level_values(1):
                            df = data.xs(t, axis=1, level=1).copy()
                        # Try to find ticker in level 0
                        elif t in data.columns.get_level_values(0):
                            df = data[t].copy()
                        else:
                            failure_reason = f"Ticker {t} not found in MultiIndex levels"
                    else:
                        # Flat index
                        if len(batch) == 1:
                            df = data.copy()
                        else:
                            # If flat but multiple tickers, check for "Close_AAPL" format or similar
                            # This usually doesn't happen with default yf.download but good to be safe
                            cols = [c for c in data.columns if t in c]
                            if cols:
                                df = data[cols].copy()
                            else:
                                failure_reason = f"Ticker {t} not found in Flat Index columns"
                    
                    # === VALIDATION ===
                    if df.empty:
                        print(f"      🔸 {t}: Skipped ({failure_reason})")
                        continue

                    # Standardize Columns
                    if 'Close' not in df.columns and 'Adj Close' in df.columns:
                        df['Close'] = df['Adj Close']
                    
                    if 'Close' in df.columns:
                        valid_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                        df = df[valid_cols]
                        df = df.dropna(subset=['Close'])
                        
                        if len(df) > 200:
                            save_path = f"{CONFIG['DATA_DIR']}/raw/{t}.parquet"
                            df.to_parquet(save_path)
                            print(f"      ✅ {t}: Saved ({len(df)} rows)")
                            downloaded_count += 1
                        else:
                            print(f"      🔸 {t}: Too short ({len(df)} rows)")
                    else:
                        print(f"      🔸 {t}: Missing 'Close' column. Has: {list(df.columns)}")
                                
                except Exception as e:
                    print(f"      ❌ {t}: CRASH -> {str(e)}")
                    # traceback.print_exc() # Uncomment if you want full stack traces
                    continue
        
        except Exception as e:
            print(f"   ❌ FATAL BATCH ERROR: {e}")
            traceback.print_exc()
            time.sleep(1)

    print(f"\n   ✅ Download complete. Saved {downloaded_count} files.")
    return downloaded_count

def phase_2_process():
    print("\n⚙️ PHASE 2: PROCESSING & ENGINEERING")
    
    files = glob.glob(f"{CONFIG['DATA_DIR']}/raw/*.parquet")
    if not files:
        print("❌ CRITICAL: No data files found. Phase 1 failed.")
        return False

    scaler = RobustScaler()
    X_buffer, Y_buffer = [], []
    
    print(f"   Processing {len(files)} files into tensors...")
    
    for f in tqdm(files):
        try:
            df = pd.read_parquet(f)
            df = df.ffill().dropna()
            
            if len(df) < CONFIG['LOOKBACK'] + 252: continue
            
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            
            vol_mean = df['Volume'].rolling(20).mean()
            vol_std = df['Volume'].rolling(20).std()
            df['Vol_Norm'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)
            
            df['H-L'] = (df['High'] - df['Low']) / df['Close']
            df['RSI'] = FinancialMath.get_rsi(df['Close']) / 100.0
            df['MACD'] = FinancialMath.get_macd(df['Close']) / (df['Close'] + 1e-8)
            df['ATR'] = FinancialMath.get_atr(df['High'], df['Low'], df['Close']) / (df['Close'] + 1e-8)
            
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            static_cols = ['Log_Ret', 'Vol_Norm', 'H-L', 'RSI', 'MACD', 'ATR']
            data_tech = df[static_cols].values
            data_close = df['Close'].values
            
            fund_vec = [20.0, 3.0, 0.5, 0.15, 25.0, 1.0] 
            
            L = CONFIG['LOOKBACK']
            limit = len(df) - 252
            
            # Stride 5
            for i in range(L, limit, 5): 
                tech_win = data_tech[i-L : i]
                price_win = data_close[i-L : i]
                energies = FinancialMath.get_fft_energy(price_win, top_k=3)
                fft_win = np.tile(energies, (L, 1))
                fund_win = np.tile(fund_vec, (L, 1))
                full_win = np.concatenate([tech_win, fft_win, fund_win], axis=1)
                
                p_cur = data_close[i-1]
                t_1d = np.log(data_close[i] / p_cur)
                t_1w = np.log(data_close[i+4] / p_cur)
                t_1m = np.log(data_close[i+20] / p_cur)
                t_1y = np.log(data_close[i+251] / p_cur)
                
                X_buffer.append(full_win)
                Y_buffer.append([t_1d, t_1w, t_1m, t_1y])
                
        except Exception:
            continue

    if len(X_buffer) == 0:
        print("❌ No valid samples generated.")
        return False

    print("   Concatenating & Scaling...")
    X_final = np.array(X_buffer, dtype=np.float32)
    Y_final = np.array(Y_buffer, dtype=np.float32)
    
    N, L, F = X_final.shape
    X_flat = X_final.reshape(-1, F)
    X_scaled = scaler.fit_transform(X_flat).reshape(N, L, F)
    
    np.save(f"{CONFIG['DATA_DIR']}/processed/X.npy", X_scaled)
    np.save(f"{CONFIG['DATA_DIR']}/processed/Y.npy", Y_final)
    joblib.dump(scaler, f"{CONFIG['MODEL_DIR']}/scaler.pkl")
    
    print(f"✅ Saved {N} samples to disk.")
    return True

def phase_3_train():
    print("\n🔥 PHASE 3: TRAINING")
    
    x_path = f"{CONFIG['DATA_DIR']}/processed/X.npy"
    y_path = f"{CONFIG['DATA_DIR']}/processed/Y.npy"
    
    if not os.path.exists(x_path):
        print("❌ Data not found.")
        return

    ds = BigDataset(x_path, y_path)
    train_len = int(0.9 * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, val_len])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'])
    
    model = MultiHorizonTransformer(CONFIG['FEATURES'], CONFIG['HIDDEN_DIM'], CONFIG['HEADS'], CONFIG['LAYERS'], CONFIG['DROPOUT']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    loss_fn = QuantileLoss()
    
    best_loss = float('inf')
    
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        train_loss = 0
        steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
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
            
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += loss_fn(model(bx), by).item()
                val_steps += 1
        
        avg_val = val_loss / val_steps
        scheduler.step(avg_val)
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), f"{CONFIG['MODEL_DIR']}/sp500_transformer.pth")
    
    print("✅ Training Complete.")

def predict_cones(ticker):
    model_path = f"{CONFIG['MODEL_DIR']}/sp500_transformer.pth"
    if not os.path.exists(model_path): return "Model not found."
    
    model = MultiHorizonTransformer(CONFIG['FEATURES'], CONFIG['HIDDEN_DIM'], CONFIG['HEADS'], CONFIG['LAYERS'], CONFIG['DROPOUT']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(f"{CONFIG['MODEL_DIR']}/scaler.pkl")
    
    print(f"   Fetching live data for {ticker}...")
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=1)
            except:
                df.columns = df.columns.droplevel(1)

        if len(df) < CONFIG['LOOKBACK']: return "Not enough data."
        
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        vol_mean = df['Volume'].rolling(20).mean()
        vol_std = df['Volume'].rolling(20).std()
        df['Vol_Norm'] = (df['Volume'] - vol_mean) / (vol_std + 1e-8)
        df['H-L'] = (df['High'] - df['Low']) / df['Close']
        df['RSI'] = FinancialMath.get_rsi(df['Close']) / 100.0
        df['MACD'] = FinancialMath.get_macd(df['Close']) / (df['Close'] + 1e-8)
        df['ATR'] = FinancialMath.get_atr(df['High'], df['Low'], df['Close']) / (df['Close'] + 1e-8)
        df = df.fillna(0)
        
        static_cols = ['Log_Ret', 'Vol_Norm', 'H-L', 'RSI', 'MACD', 'ATR']
        data_tech = df[static_cols].values[-CONFIG['LOOKBACK']:]
        
        price_win = df['Close'].values[-CONFIG['LOOKBACK']:]
        energies = FinancialMath.get_fft_energy(price_win, top_k=3)
        fft_win = np.tile(energies, (CONFIG['LOOKBACK'], 1))
        fund_win = np.tile([20,3,0.5,0.15,25,1], (CONFIG['LOOKBACK'], 1))
        
        X_raw = np.concatenate([data_tech, fft_win, fund_win], axis=1)
        X_scaled = scaler.transform(X_raw.reshape(-1, CONFIG['FEATURES'])).reshape(1, CONFIG['LOOKBACK'], CONFIG['FEATURES'])
        
        with torch.no_grad():
            preds = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()[0]
            
        p = df['Close'].iloc[-1]
        
        return {
            "price": p, 
            "1M_Bear": p * np.exp(preds[2][0]),
            "1M_Med":  p * np.exp(preds[2][1]),
            "1M_Bull": p * np.exp(preds[2][2])
        }
    except Exception as e:
        return f"Prediction failed: {e}"

# ==========================================
# 4. MAIN ORCHESTRATION
# ==========================================
if __name__ == "__main__":
    print(f"🤖 AGENT REBOOT | Device: {device}")
    
    raw_files = glob.glob(f"{CONFIG['DATA_DIR']}/raw/*.parquet")
    
    # 1. Download
    # Force download if files < 10 (or you can comment this out to force redownload)
    if len(raw_files) < 10:
        count = phase_1_download()
        if count == 0:
            print("❌ Download Failed completely. Exiting.")
            exit()
    else:
        print(f"   ✅ Data detected ({len(raw_files)} files). Skipping download.")
    
    # 2. Process
    if not os.path.exists(f"{CONFIG['DATA_DIR']}/processed/X.npy"):
        success = phase_2_process()
        if not success: exit()
        
    # 3. Train
    phase_3_train()
    
    # 4. Predict
    print("\n🔮 PREDICTION TEST: NVDA")
    res = predict_cones("NVDA")
    if isinstance(res, dict):
        print(f"   Current Price: ${res['price']:.2f}")
        print(f"   -----------------------------")
        print(f"   🐻 1M Bear:   ${res['1M_Bear']:.2f}")
        print(f"   ⚖️ 1M Median: ${res['1M_Med']:.2f}")
        print(f"   🐂 1M Bull:   ${res['1M_Bull']:.2f}")
    else:
        print(res)
import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings for clean logs
warnings.filterwarnings("ignore")

class FinancialMath:
    """
    LEVEL 8 FINANCIAL ENGINE
    Calculates advanced technical indicators and statistical properties.
    Now includes Signal Processing (FFT) and Energy dynamics.
    """

    @staticmethod
    def get_rsi(series, period=14):
        """
        Relative Strength Index (RSI).
        Returns a series scaled 0-100.
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Fill NaN values (start of series) with 50 (neutral)
        return rsi.fillna(50)

    @staticmethod
    def get_volatility(series, window=20):
        """
        Rolling volatility (standard deviation of returns).
        """
        return series.rolling(window=window).std().fillna(0)

    @staticmethod
    def get_log_returns(series):
        """
        Logarithmic returns for stationarity.
        """
        return np.log(series / series.shift(1)).fillna(0)

    @staticmethod
    def get_macd(series, fast=12, slow=26, signal=9):
        """
        Moving Average Convergence Divergence (Momentum).
        Returns the Histogram (MACD Line - Signal Line).
        """
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        # We return the histogram which indicates momentum strength/direction
        return (macd - signal_line).fillna(0)

    @staticmethod
    def get_atr(high, low, close, window=14):
        """
        Average True Range (Volatility/Energy).
        Measures the absolute price movement regardless of direction.
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        # Combine into a dataframe to find the max per row
        tr_df = pd.concat([tr1, tr2, tr3], axis=1)
        tr = tr_df.max(axis=1)
        
        return tr.rolling(window=window).mean().fillna(0)

    @staticmethod
    def get_fft_energy(window_data, top_k=3):
        """
        Signal Processing: Fast Fourier Transform (FFT).
        Extracts cyclic 'energy' from price movements.
        
        Args:
            window_data (np.array): Array of close prices for the window.
            top_k (int): Number of dominant frequencies to return.
            
        Returns:
            list: Magnitudes of the top_k strongest cyclic frequencies.
        """
        # 1. Detrend the data (FFT works best on stationary waves, not trends)
        # We subtract the mean to center it around 0
        detrended = window_data - np.mean(window_data)
        
        # 2. Compute 1D FFT (rfft is faster for real-valued inputs)
        fft_vals = np.fft.rfft(detrended) 
        
        # 3. Get Magnitudes (Absolute value of complex numbers)
        fft_mag = np.abs(fft_vals)
        
        # 4. Sort by magnitude descending
        # We skip index 0 (DC Component/Mean) as we already detrended
        sorted_indices = np.argsort(fft_mag[1:])[::-1]
        
        # 5. Extract top K energies
        top_energies = []
        for i in range(top_k):
            if i < len(sorted_indices):
                # +1 because we skipped index 0
                top_energies.append(fft_mag[sorted_indices[i] + 1])
            else:
                top_energies.append(0.0)
                
        return top_energies

    @staticmethod
    def get_hurst_exponent(series, min_lag=2, max_lag=20):
        """
        Calculates the Hurst Exponent to determine if a stock is trending (H>0.5) 
        or mean-reverting (H<0.5).
        """
        if len(series) < max_lag + 5:
            return 0.5
            
        try:
            lags = range(min_lag, max_lag)
            # Calculate the array of standard deviations of differences
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            
            # Linear fit to log-log plot
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = m[0] * 2.0
            
            # Clip bounds to keep neural network stable
            return max(0.0, min(1.0, hurst))
        except:
            return 0.5

class GraphMath:
    """
    LEVEL 7 GRAPH TOPOLOGY ENGINE
    Constructs the 'Brain' wiring connecting thousands of stocks.
    """
    
    @staticmethod
    def build_semantic_graph(descriptions, top_k=10, min_weight=0.1):
        """
        Uses NLP to find relationships between companies.
        Returns PyTorch Geometric Edge Index and Weights.
        """
        num_nodes = len(descriptions)
        print(f"🕸️  MathTools: Computing Semantic Adjacency for {num_nodes} nodes...")
        
        if num_nodes < 2:
            return torch.tensor([[],[]], dtype=torch.long), torch.tensor([], dtype=torch.float)

        # 1. TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # 2. Cosine Similarity
        sim_matrix_np = cosine_similarity(tfidf_matrix)
        
        # 3. Sparsification (Top-K Selection)
        sim_tensor = torch.tensor(sim_matrix_np, dtype=torch.float)
        
        # Find the top K+1 values (K neighbors + 1 self-loop)
        k = min(top_k + 1, num_nodes)
        top_vals, top_inds = torch.topk(sim_tensor, k=k, dim=1)
        
        source_nodes = []
        target_nodes = []
        edge_weights = []
        
        # Iterate through the Top-K results
        for i in range(num_nodes):
            for j in range(k):
                target_idx = top_inds[i, j].item()
                weight = top_vals[i, j].item()
                
                # Filter self-loops and weak connections
                if target_idx != i and weight > min_weight:
                    source_nodes.append(i)
                    target_nodes.append(target_idx)
                    edge_weights.append(weight)
        
        # 4. Convert to PyTorch Geometric Format
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_attr

    @staticmethod
    def get_graph_density(num_nodes, num_edges):
        """Helper to check if graph is too sparse or too dense."""
        if num_nodes < 2: return 0
        max_edges = num_nodes * (num_nodes - 1)
        return num_edges / max_edges
# src/settings/doomsday_config.py

# --- GLOBAL MACRO SETTINGS ---

# 1. GDELT (Political Layer)
# We filter for these specific CAMEO event root codes or keywords
GDELT_KEYWORDS = [
    "military buildup", "naval blockade", "sanctions", 
    "airspace violation", "artillery fire", "peace treaty"
]
# Focus on these key geopolitical hotspots
GDELT_LOCATIONS = ["Taiwan", "Strait of Hormuz", "South China Sea", "Ukraine"]

# 2. YAHOO FINANCE TICKERS (Macro & Real Layers)
TICKERS = {
    "GOLD": "GC=F",          # Gold Futures
    "COPPER": "HG=F",        # Copper Futures
    "OIL_BRENT": "BZ=F",     # Brent Crude Oil
    "BOND_10Y": "^TNX",      # 10-Year Treasury Yield
    "BOND_3M": "^IRX",       # 13-Week Treasury Yield
    "VIX": "^VIX"            # Volatility Index
}

# 3. TRIANGULATION WEIGHTS
# How much we trust each signal to predict the "End of Days"
WEIGHTS = {
    "POLITICAL": 0.40,  # 40% - GDELT / News Tone
    "MACRO": 0.30,      # 30% - Yield Curve / Bonds
    "REAL": 0.30        # 30% - Commodities (Gold/Copper/Oil)
}

# 4. DATABASE
DB_PATH = "market_memory.db"  # Saves in your project root
TABLE_NAME = "global_risk_log"
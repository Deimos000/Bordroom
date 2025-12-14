import os
import json
import time
import logging
from tqdm import tqdm  # You may need to run: pip install tqdm
from src.utils.local_llm import LocalLLM

# --- CONFIGURATION ---
# Folder where you drop raw text files (e.g., "NVDA_10k.txt", "TSLA_News.txt")
RAW_DATA_PATH = "data/raw_documents" 

# Intermediate file: Stores the "high signal" text filtered by the Scout
STAGING_FILE = "data/staging_signal.jsonl" 

# Final output: The structured Knowledge Graph JSON
FINAL_FILE = "data/final_knowledge_graph.json"

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [PIPELINE] %(message)s')
logger = logging.getLogger(__name__)

def ensure_paths():
    """Creates necessary folders if they don't exist."""
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)
        print(f"📁 Created directory: {RAW_DATA_PATH}")
        print(f"   👉 Please drop .txt files here (e.g., 'AAPL_2024_Report.txt')")

def parse_ticker_from_filename(filename):
    """
    Heuristic: Assumes filename starts with Ticker (e.g., 'NVDA_Report.txt').
    Defaults to 'UNKNOWN' if it can't guess.
    """
    try:
        # Split by underscore or dot
        parts = filename.replace('.', '_').split('_')
        possible_ticker = parts[0].upper()
        if len(possible_ticker) < 6 and possible_ticker.isalpha():
            return possible_ticker
        return "UNKNOWN"
    except:
        return "UNKNOWN"

# =========================================================================
# PHASE 1: THE SCOUT (Filter Noise)
# Model: Phi-3.5 (Fast)
# =========================================================================
def run_phase_1_scout(llm):
    print("\n🚜 PHASE 1: SCOUTING (Model: Phi-3.5)")
    print("   Goal: Compress raw text into high-signal intelligence.")

    # 1. Get List of Files
    files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.txt')]
    if not files:
        print("   ⚠️  No .txt files found in data/raw_documents/")
        return

    # 2. Check for Resume (Don't re-process files already in staging)
    processed_files = set()
    if os.path.exists(STAGING_FILE):
        with open(STAGING_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    processed_files.add(entry['filename'])
                except: pass
    
    files_to_do = [f for f in files if f not in processed_files]
    print(f"   📄 Queue: {len(files_to_do)} files (Skipped {len(processed_files)} already done)")

    if not files_to_do:
        return

    # 3. Process Loop
    with open(STAGING_FILE, "a", encoding='utf-8') as f_out:
        for filename in tqdm(files_to_do, desc="Scouting"):
            ticker = parse_ticker_from_filename(filename)
            file_path = os.path.join(RAW_DATA_PATH, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    raw_text = f_in.read()

                # --- CALL THE LLM ENGINE ---
                # This automatically ensures 'phi3.5' is loaded
                filtered_text = llm.scout(raw_text)

                if filtered_text:
                    record = {
                        "filename": filename,
                        "ticker": ticker,
                        "signal_text": filtered_text,
                        "timestamp": time.time()
                    }
                    f_out.write(json.dumps(record) + "\n")
                    f_out.flush() # Force save
            
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")

    print("✅ Phase 1 Complete. Signal saved to staging.")

# =========================================================================
# PHASE 2: THE SNIPER (Analysis)
# Model: Qwen 2.5 (Smart)
# =========================================================================
def run_phase_2_sniper(llm):
    print("\n🧠 PHASE 2: SNIPING (Model: Qwen 2.5)")
    print("   Goal: Extract structured JSON relationships.")

    if not os.path.exists(STAGING_FILE):
        print("   ❌ No staging data found. Run Phase 1 first.")
        return

    # 1. Read Staging Data
    records = []
    with open(STAGING_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except: pass

    print(f"   🔍 Analyzing {len(records)} high-signal segments...")
    
    results = []

    # 2. Process Loop
    for record in tqdm(records, desc="Sniping"):
        ticker = record['ticker']
        signal_text = record['signal_text']
        
        try:
            # --- CALL THE LLM ENGINE ---
            # This automatically ensures 'qwen2.5' is loaded
            facts = llm.extract_relationships(signal_text, ticker)
            
            if facts:
                for fact in facts:
                    # Enrich data with source metadata
                    fact['source_file'] = record['filename']
                    fact['source_ticker'] = ticker
                    results.append(fact)
        
        except Exception as e:
            logger.error(f"Sniper failed on {ticker}: {e}")

    # 3. Save Final JSON
    with open(FINAL_FILE, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Phase 2 Complete.")
    print(f"   📊 Extracted {len(results)} relationships.")
    print(f"   📁 Output saved to: {FINAL_FILE}")

# =========================================================================
# MAIN EXECUTION
# =========================================================================
if __name__ == "__main__":
    print("==================================================")
    print("   SCOUT & SNIPER: FINANCIAL ANALYSIS PIPELINE    ")
    print("==================================================")
    
    # 1. Setup folders
    ensure_paths()
    
    # 2. Initialize the Engine (Connects to Ollama)
    try:
        engine = LocalLLM()
    except Exception as e:
        print(f"\n❌ FATAL: Could not connect to LocalLLM. Is Ollama running?\nError: {e}")
        exit(1)
    
    # 3. Run Phase 1 (The Scout)
    # The engine loads Phi-3.5 and keeps it loaded for the whole batch
    run_phase_1_scout(engine)
    
    # 4. Run Phase 2 (The Sniper)
    # The engine detects the change, unloads Phi, and loads Qwen for the batch
    run_phase_2_sniper(engine)
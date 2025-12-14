import logging
import json
import time
import re
from datetime import datetime, timedelta

# --- Custom Modules ---
try:
    from src.utils.vector_memory import VectorMemory
    from src.utils.local_llm import LocalLLM
    from src.utils.atlas_db import AtlasDB
except ImportError:
    from vector_memory import VectorMemory
    from local_llm import LocalLLM
    from atlas_db import AtlasDB

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [BS_CHECKER] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ContradictionDetector")

class ContradictionDetector:
    """
    The 'Bullshit Checker' (Integrity Agent).
    
    ARCHITECTURE: SNIPER ONLY.
    This agent does not use the 'Scout' because it consumes data 
    that is already processed/indexed in VectorMemory.
    
    It relies on the 'Sniper' (Qwen 2.5) to perform deep logical comparisons.
    """
    def __init__(self):
        self.memory = VectorMemory()
        self.brain = LocalLLM() # This connects to the Scout/Sniper engine
        self.db = AtlasDB() 

    def check_integrity(self, ticker, topic="supply chain"):
        """
        Compares Official Narratives (SEC) vs Street Reality (News).
        """
        ticker = ticker.upper()
        logger.info(f"🕵️‍♂️ SNIPER ACTIVATED: Integrity Scan for {ticker} on '{topic}'")

        # 1. Fetch Official Stance (SEC) - The "Anchor Truth"
        official_docs = self.memory.recall(
            query=topic,
            ticker=ticker,
            n=3,
            filter_category="SEC_10K"
        )

        # 2. Fetch Street Stance (News/Web) - The "Challenger"
        street_docs = self.memory.recall(
            query=topic,
            ticker=ticker,
            n=5,
            filter_category="news"
        )

        # 3. Validation
        if not official_docs:
            logger.warning(f"⚠️ No SEC data found for {ticker} regarding {topic}.")
            return None
        
        if not street_docs:
            logger.info(f"✅ No rumors found for {ticker} regarding {topic}. Assuming quiet.")
            return None

        # 4. Date Logic & Time Context
        # Sort desc to get latest dates
        official_docs.sort(key=lambda x: x.get('date', ''), reverse=True)
        street_docs.sort(key=lambda x: x.get('date', ''), reverse=True)

        latest_official_date = official_docs[0].get('date', 'N/A')
        latest_street_date = street_docs[0].get('date', 'N/A')

        time_context = "Unknown"
        try:
            d1 = datetime.fromisoformat(latest_official_date)
            d2 = datetime.fromisoformat(latest_street_date)
            delta_days = (d2 - d1).days
            
            if delta_days > 30:
                time_context = f"CRITICAL: News is NEWER by {delta_days} days. This may represent a Pivot or New Risk."
            elif delta_days < -30:
                time_context = "IGNORE: News is OLDER than the SEC filing. The filing is the authority."
            else:
                time_context = "SIMULTANEOUS: Direct contradiction analysis required."
        except ValueError:
            pass

        # 5. Synthesize Narratives (Safety Truncated for Sniper Context)
        # We limit specific chunks to ensure the Sniper has 'thinking room'
        official_text = "\n---\n".join([f"[{d.get('date', '?')}] {d['text'][:800]}..." for d in official_docs])
        street_text = "\n---\n".join([f"[{d.get('date', '?')} - {d.get('source', 'Web')}] {d['text'][:800]}..." for d in street_docs])

        # 6. The Sniper Prompt
        # This prompts the 'reason' method in local_llm.py to load Qwen 2.5
        prompt = f"""
        TARGET: {ticker}
        TOPIC: {topic}

        === OFFICIAL NARRATIVE (SEC Filings) ===
        Last Updated: {latest_official_date}
        EVIDENCE:
        {official_text}

        === STREET NARRATIVE (News/Leaks) ===
        Last Updated: {latest_street_date}
        EVIDENCE:
        {street_text}

        === TIME CONTEXT ===
        {time_context}

        === MISSION ===
        1. Analyze the two narratives for LOGICAL CONTRADICTIONS.
        2. If the Street Narrative is NEWER and NEGATIVE, it is a "High Risk".
        3. If the Street Narrative is OLDER, it is "Noise".
        
        === REQUIRED OUTPUT (JSON) ===
        {{
            "divergence_score": <float 0.0 to 1.0>,
            "verdict": "Deception" | "Pivot" | "Safe" | "Noise",
            "explanation": "<Short, punchy reason>",
            "confidence": <float 0.0 to 1.0>
        }}
        """

        try:
            # CALLING THE SNIPER
            # The .reason() method in LocalLLM automatically ensures Qwen 2.5 is loaded
            response = self.brain.reason(
                prompt, 
                system_prompt="You are a Skeptical Forensic Accountant. You output JSON only."
            )
            
            # Parse
            analysis = self._clean_and_parse_json(response)
            
            if analysis:
                self._handle_result(ticker, topic, analysis, time_context)
                return analysis
            else:
                logger.error("Sniper returned malformed JSON.")
                return None

        except Exception as e:
            logger.error(f"Sniper check failed: {e}")
            return None

    def _handle_result(self, ticker, topic, analysis, time_context):
        score = analysis.get('divergence_score', 0)
        verdict = analysis.get('verdict', 'Unknown')
        
        logger.info(f"📋 REPORT: {ticker} | Score: {score} | Verdict: {verdict}")
        
        # Logic: If divergence is high (>0.7), we write a 'Risk' node to the AtlasDB
        if score > 0.7:
            logger.info(f"🚩 HIGH RISK DETECTED: Writing to AtlasDB...")
            self.db.save_relationship(
                source=ticker,
                target=ticker, # Self-reference = Internal Risk
                rel_type="risk_alert",
                reason=f"BS_CHECK: {verdict} on {topic}. {analysis.get('explanation')}",
                strength=score,
                sentiment=-0.9,
                confidence=analysis.get('confidence', 0.5)
            )

    def scan_portfolio(self, tickers, topics=["supply chain", "liquidity", "legal"]):
        """
        Batch process a portfolio.
        """
        for t in tickers:
            for topic in topics:
                self.check_integrity(t, topic)
                time.sleep(1) # Breath

    def _clean_and_parse_json(self, raw_text):
        try:
            match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
            text = match.group(1) if match else raw_text
            text = re.sub(r'```json', '', text, flags=re.IGNORECASE)
            text = re.sub(r'```', '', text)
            return json.loads(text)
        except:
            return None

# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    detector = ContradictionDetector()
    
    # Test on a hypothetical scenario
    print("\n🧠 Testing BS Detector (Sniper Mode)...")
    detector.check_integrity("NVDA", "supply chain")
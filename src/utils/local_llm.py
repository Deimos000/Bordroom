import logging
import time
import re
import ollama

# Try to import the robust JSON parser, fall back to standard if missing
try:
    import json_repair
except ImportError:
    import json as json_repair
    print("⚠️ WARNING: 'json_repair' not found. Install it for better reliability: pip install json_repair")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [LLM_ENGINE] %(levelname)s - %(message)s')
logger = logging.getLogger("LocalLLM")

class LocalLLM:
    def __init__(self, scout_model="phi3.5", sniper_model="qwen2.5:7b"):
        """
        The Brain of the operation.
        
        Args:
            scout_model: Fast model for high-volume text filtering (Phi-3.5 recommended).
            sniper_model: Smart model for logic and JSON extraction (Qwen 2.5 or Llama 3 recommended).
        """
        self.scout_model = scout_model
        self.sniper_model = sniper_model
        self.current_model = None
        
        # Performance settings
        self.context_window = 8192  # Increased for reading 10-K chunks
        
        self._check_connection()

    def _check_connection(self):
        """Ensures Ollama is running and models are available."""
        try:
            models_dict = ollama.list()
            # ollama.list() returns a dict with 'models' key which is a list
            installed_models = [m['name'] for m in models_dict.get('models', [])]
            
            logger.info(f"✅ Ollama Connected.")
            
            # Check if requested models exist
            if not any(self.scout_model in m for m in installed_models):
                logger.warning(f"⚠️ Scout model '{self.scout_model}' not found. Run: ollama pull {self.scout_model}")
            
            if not any(self.sniper_model in m for m in installed_models):
                logger.warning(f"⚠️ Sniper model '{self.sniper_model}' not found. Run: ollama pull {self.sniper_model}")

        except Exception as e:
            logger.critical("❌ OLLAMA NOT RUNNING. Open your terminal and run 'ollama serve'.")
            raise ConnectionError(f"Ollama unreachable: {e}")

    def _switch_brain(self, target_model):
        """
        Manages VRAM. If the requested model isn't loaded, it swaps them.
        """
        if self.current_model == target_model:
            return

        logger.info(f"🧠 LOADING MODEL: {target_model} (Unloading {self.current_model or 'None'})...")
        try:
            # Sending a dummy request forces Ollama to load the model into VRAM immediately
            # set keep_alive to -1 to keep it loaded, or a duration (e.g. 5m)
            ollama.chat(model=target_model, messages=[{'role':'user', 'content':'hi'}])
            
            self.current_model = target_model
            time.sleep(1.0) # Give the GPU a second to settle
        except Exception as e:
            logger.error(f"Failed to load {target_model}: {e}")
            raise

    # =========================================================================
    # 1. THE SCOUT (Filter Noise)
    # =========================================================================
    def scout(self, raw_text):
        """
        Reads massive text and returns ONLY relevant paragraphs.
        """
        self._switch_brain(self.scout_model)
        
        # Safety truncate to avoid blowing up context (Phi-3.5 handles 128k, but let's be safe/fast)
        safe_text = raw_text[:35000]

        prompt = f"""
        TASK: You are a financial data filter.
        INSTRUCTIONS:
        1. Read the text below.
        2. Extract ANY paragraphs that mention:
           - Supply chain relationships (suppliers, customers).
           - Specific company names (Competitors, Partners).
           - Lawsuits, fines, or regulatory investigations.
           - Future guidance or risk factors.
        3. Do NOT summarize. Copy the relevant text verbatim.
        4. If the text contains NO relevant business information, output exactly: "NO_SIGNAL"

        --- TEXT START ---
        {safe_text}
        --- TEXT END ---
        """

        try:
            response = ollama.chat(
                model=self.scout_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.0, # Deterministic
                    'num_ctx': self.context_window
                }
            )
            content = response['message']['content']
            
            if "NO_SIGNAL" in content or len(content) < 20:
                return None
            return content
        except Exception as e:
            logger.error(f"Scout failed: {e}")
            return None

    # =========================================================================
    # 2. THE SNIPER (Analysis & Extraction)
    # =========================================================================
    def extract_relationships(self, text_chunk, focus_ticker):
        """
        Extracts structured JSON relationships from high-signal text.
        """
        self._switch_brain(self.sniper_model)

        prompt = f"""
        You are a Data Extraction Engine.
        TARGET: Analyze the text specifically regarding '{focus_ticker}'.
        
        GOAL: Identify business relationships and risks.
        
        REQUIRED JSON OUTPUT FORMAT (List of Objects):
        [
            {{
                "target_entity": "Exact Company Name",
                "type": "supplier" | "customer" | "competitor" | "partner" | "lawsuit" | "regulatory",
                "details": "Brief explanation of the relationship or event",
                "sentiment": <float between -1.0 and 1.0>,
                "confidence": <float between 0.0 and 1.0>
            }}
        ]

        RULES:
        1. Output VALID JSON ONLY. No markdown, no conversational text.
        2. If no relationships are found, output: []
        3. Do NOT extract '{focus_ticker}' as the target. Find who they interact with.
        
        --- TEXT ---
        {text_chunk[:12000]}
        """
        
        try:
            response = ollama.chat(
                model=self.sniper_model,
                messages=[
                    {'role': 'system', 'content': "You are a JSON extraction machine. Output only raw JSON."},
                    {'role': 'user', 'content': prompt}
                ],
                format='json', # Ollama Native JSON mode
                options={
                    'temperature': 0.1, 
                    'num_ctx': self.context_window
                }
            )
            return self._clean_json(response['message']['content'], focus_ticker)
        except Exception as e:
            logger.error(f"Sniper extraction failed: {e}")
            return []

    # =========================================================================
    # 3. UTILITIES (The "Critic" & Reasoning)
    # =========================================================================
    def validate_claim(self, claim, context):
        """
        Verifies if a specific claim is supported by the context.
        Returns: Boolean
        """
        self._switch_brain(self.sniper_model)
        
        prompt = f"""
        VERIFICATION TASK:
        Claim: "{claim}"
        
        Context:
        "{context[:5000]}"
        
        INSTRUCTIONS:
        Is the claim strictly supported by the context? 
        Respond with JSON: {{ "is_true": boolean, "confidence": float }}
        """
        try:
            res = ollama.chat(
                model=self.sniper_model, 
                messages=[{'role':'user', 'content':prompt}], 
                format='json',
                options={'temperature':0.0}
            )
            data = json_repair.loads(res['message']['content'])
            # Default to False if confidence is low
            if isinstance(data, dict):
                return data.get('is_true', False) and data.get('confidence', 0) > 0.7
            return False
        except:
            return False

    def reason(self, prompt, system_prompt="You are a Financial Analyst."):
        """
        General purpose reasoning.
        """
        self._switch_brain(self.sniper_model)
        try:
            res = ollama.chat(
                model=self.sniper_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.2, 'num_ctx': self.context_window}
            )
            return res['message']['content']
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return "Error."

    def _clean_json(self, raw_text, focus_ticker):
        """
        Robustly parses JSON from LLM output.
        Handles Markdown fences, trailing commas, and missing quotes.
        """
        try:
            # 1. Remove Markdown Code Blocks
            text = re.sub(r'```json', '', raw_text, flags=re.IGNORECASE)
            text = re.sub(r'```', '', text).strip()
            
            # 2. Use json_repair to handle broken JSON
            data = json_repair.loads(text)
            
            # 3. Normalize to List
            if isinstance(data, dict): 
                data = [data]
            
            if not isinstance(data, list):
                return []
            
            # 4. Filter Logic
            valid = []
            focus_clean = focus_ticker.upper().strip()
            
            for item in data:
                # Ensure keys exist
                if 'target_entity' in item and 'type' in item:
                    target = str(item['target_entity']).upper().strip()
                    
                    # Prevent Self-Loops (e.g. NVDA linked to NVDA)
                    if target and target != focus_clean:
                        # Fix formatting
                        item['target_entity'] = item['target_entity'].title() 
                        valid.append(item)
                        
            return valid

        except Exception as e:
            logger.error(f"JSON Parsing Error: {e} | Raw Text: {raw_text[:100]}")
            return []

# ==========================================
# TEST EXECUTION
# ==========================================
if __name__ == "__main__":
    print("🧪 Testing LocalLLM Module...")
    
    try:
        # Initialize
        brain = LocalLLM()
        
        # Test Text (Fictional)
        sample_text = """
        Apple Inc. (AAPL) is facing supply chain constraints due to Foxconn production delays. 
        Meanwhile, Samsung Electronics has filed a lawsuit against Apple regarding screen patents.
        """
        
        print("\n--- 1. Testing Sniper (JSON Extraction) ---")
        relationships = brain.extract_relationships(sample_text, "AAPL")
        print(f"Result: {relationships}")
        
        print("\n--- 2. Testing Scout (Filtering) ---")
        signal = brain.scout(sample_text)
        print(f"Result: {signal}")

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
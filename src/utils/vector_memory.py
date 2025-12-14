import chromadb
from chromadb.utils import embedding_functions
import hashlib
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorMemory")

class VectorMemory:
    def __init__(self, db_path="data/brain_vectors"):
        """
        Initializes the Vector Database for semantic storage.
        Uses ChromaDB with local persistence.
        """
        # Ensure the directory exists
        try:
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)
        except OSError:
            pass
        
        # Initialize Persistent Client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Use a localized, high-performance embedding model
        # 'all-MiniLM-L6-v2' is standard for efficient local RAG (384 dimensions)
        try:
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.warning(f"Default embedding model failed loading, ensure sentence-transformers is installed. Error: {e}")
            # Fallback or crash depending on strictness. Retrying usually fixes it if it's a download issue.
            raise e
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="financial_knowledge",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"} # Cosine similarity for semantic search
        )
        
        logger.info(f"🧠 Vector Memory loaded. Documents indexed: {self.collection.count()}")

    def memorize(self, ticker, source_url, text, chunk_index=0, category="news"):
        """
        Saves a text chunk with metadata into the vector DB.
        
        Args:
            ticker (str): The stock symbol (e.g., "NVDA")
            source_url (str): Where the info came from
            text (str): The actual content
            chunk_index (int): The position of this text in the original doc
            category (str): 'news', 'SEC_10K', 'web_scrape'
        """
        if not text or len(text) < 20:
            return # Skip noise

        try:
            # 1. Deterministic ID Generation (Idempotency)
            # We hash Ticker + URL + ChunkIndex so re-scraping updates instead of duplicates.
            unique_str = f"{ticker}_{source_url}_{chunk_index}"
            doc_id = hashlib.md5(unique_str.encode()).hexdigest()
            
            timestamp = datetime.now().isoformat()
            
            # 2. Upsert (Insert or Update)
            self.collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[{
                    "ticker": ticker, 
                    "source": source_url, 
                    "category": category,
                    "chunk_index": chunk_index,
                    "timestamp": timestamp
                }]
            )
        except Exception as e:
            logger.error(f"Failed to memorize chunk for {ticker}: {e}")

    def recall(self, query, ticker=None, n=5, threshold=0.6, filter_category=None):
        """
        Finds the most relevant text chunks for a query.
        
        Args:
            query (str): The question (e.g., "Who supplies GPUs?")
            ticker (str): Optional filter by stock symbol.
            n (int): Number of results to return.
            threshold (float): Similarity threshold (0.0=Identical, 1.0=Opposite). 
                               Results with distance > threshold are ignored.
            filter_category (str): Optional. E.g., 'SEC_10K' to search only official docs.
            
        Returns:
            List[dict]: A list of clean result objects.
        """
        try:
            # Construct the 'where' clause dynamically for ChromaDB
            # ChromaDB uses a dict for implicit AND filtering
            where_clause = {}
            if ticker:
                where_clause["ticker"] = ticker
            if filter_category:
                where_clause["category"] = filter_category
            
            # If dict is empty, pass None to search everything
            if not where_clause:
                where_clause = None

            results = self.collection.query(
                query_texts=[query],
                n_results=n,
                where=where_clause
            )
            
            clean_results = []
            
            if results['documents']:
                # Chroma returns lists of lists (batch format)
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i]
                    
                    # Filter by relevance (Distance check)
                    # For Cosine distance, lower is better.
                    if distance > threshold:
                        continue
                    
                    doc = results['documents'][0][i]
                    meta = results['metadatas'][0][i]
                    
                    clean_results.append({
                        "text": doc,
                        "source": meta.get("source", "unknown"),
                        "category": meta.get("category", "unknown"),
                        "date": meta.get("timestamp", ""),
                        "relevance_score": round(1 - distance, 3) # Convert to 0-1 similarity score
                    })
            
            return clean_results

        except Exception as e:
            logger.error(f"Recall failed for query '{query}': {e}")
            return []

    def forget_ticker(self, ticker):
        """
        Deletes all memory associated with a specific stock.
        Useful when re-running a fresh investigation to clear old noise.
        """
        try:
            self.collection.delete(
                where={"ticker": ticker}
            )
            logger.info(f"🧹 Wiped memory for {ticker}")
        except Exception as e:
            logger.error(f"Could not wipe memory for {ticker}: {e}")

    def get_stats(self):
        """Returns database statistics."""
        try:
            return {
                "total_documents": self.collection.count(),
                "db_path": self.client._base_path
            }
        except:
            return {"status": "error"}

# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    mem = VectorMemory()
    
    print("\n--- 1. Testing Memorize (Idempotency) ---")
    mem.memorize(
        "NVDA", 
        "http://sec.gov/nvda-10k", 
        "Nvidia has firmly secured its supply chain with TSMC for the next 5 years.",
        chunk_index=0,
        category="SEC_10K"
    )
    
    mem.memorize(
        "NVDA", 
        "http://news.com/rumors", 
        "Rumors suggest TSMC might delay Nvidia orders due to earthquake damage.",
        chunk_index=0,
        category="news"
    )
    print(f"Total Docs: {mem.collection.count()}")

    print("\n--- 2. Testing Category Filtering (Crucial for Bullshit Checker) ---")
    
    # Check SEC Only
    print(">> Querying SEC ONLY:")
    sec_res = mem.recall("supply chain", ticker="NVDA", filter_category="SEC_10K")
    for r in sec_res:
        print(f"   [SEC] {r['text'][:60]}... (Score: {r['relevance_score']})")

    # Check News Only
    print(">> Querying News ONLY:")
    news_res = mem.recall("supply chain", ticker="NVDA", filter_category="news")
    for r in news_res:
        print(f"   [News] {r['text'][:60]}... (Score: {r['relevance_score']})")
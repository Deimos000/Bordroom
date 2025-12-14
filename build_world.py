from src.agents.forensic_agent import ForensicAgent
from src.utils.universe_factory import UniverseFactory

def update_world_knowledge():
    agent = ForensicAgent()
    
    # 1. Get list of stocks you care about
    # For testing, just do Top 5. For real, do sp500.
    universe = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"] 
    
    print(f"🚀 Starting World Build for {len(universe)} companies...")
    
    for ticker in universe:
        try:
            agent.build_dependency_graph(ticker)
        except Exception as e:
            print(f"❌ Error on {ticker}: {e}")

if __name__ == "__main__":
    update_world_knowledge()
# The Board Room AI

Comprehensive technical blueprint for building the "Board Room" Artificial Intelligence. This system uses an Ensemble Learning approach where distinct models ("The Idiots") analyze data through different lenses, and a Meta-Model ("The Boss") makes the final call.

## Structure

*   `configs/`: Configuration files (Dossier).
*   `src/`: Source code.
    *   `agents/`: The "Idiots" (Models).
    *   `boss/`: The Meta-Learner.
    *   `utils/`: Data fetching and utilities.
*   `main.py`: The simulation engine.
*   `app.py`: Streamlit dashboard.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the simulation:
    ```bash
    python board_room_ai/main.py
    ```
    This will generate `simulation_results.csv`.
    *Note: This requires internet access to fetch data from Yahoo Finance and FRED. If running in a restricted environment, it might fail or need mock data.*

3.  Run the dashboard:
    ```bash
    streamlit run board_room_ai/app.py
    ```

## Architecture

### Phase 1: The Dossier
`configs/alphabet_dossier.json` defines keywords and targets.

### Phase 2: Data Mining
`src/utils/data_miner.py` fetches financial data.
`src/utils/text_miner.py` fetches/mocks news data.

### Phase 3: Mathematical Idiots
*   **Trend Setter**: LSTM model (`src/agents/trend_setter.py`).
*   **Fed Watcher**: Random Forest (`src/agents/fed_watcher.py`).

### Phase 4: Psychological Idiots
*   **Sentiment Analyst**: FinBERT (`src/agents/sentiment_analyst.py`).
*   **Conspiracy Theorist**: Insider trading logic (`src/agents/conspiracist.py`).

### Phase 5: The Boss
*   **The Boss**: XGBoost Regressor (`src/boss/meta_learner.py`).

### Phase 6: Walk-Forward Engine
`main.py` runs the loop from 2004 to 2007.

### Phase 7: Visualization
`app.py` visualizes the results.

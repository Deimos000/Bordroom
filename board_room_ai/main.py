import pandas as pd
import numpy as np
import datetime
import json
import os
import time

# Ensure environment variables for transformers/tensorflow if needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.utils.data_miner import get_math_data
from src.utils.text_miner import TextMiner
from src.agents.trend_setter import TrendSetter
from src.agents.fed_watcher import FedWatcher
from src.agents.sentiment_analyst import SentimentAnalyst
from src.agents.conspiracist import Conspiracist
from src.boss.meta_learner import TheBoss

def load_dossier():
    path = os.path.join(os.path.dirname(__file__), 'configs', 'alphabet_dossier.json')
    with open(path, 'r') as f:
        return json.load(f)

def run_simulation():
    print("Starting Board Room Simulation...")

    # Configuration
    dossier = load_dossier()
    start_date = "2000-01-01"
    simulation_start_date = "2004-01-01"
    end_date = "2007-06-01"

    # Initialize Agents
    trend_setter = TrendSetter()
    fed_watcher = FedWatcher()
    hype_man = SentimentAnalyst(personality="hype")
    doomsayer = SentimentAnalyst(personality="doom")
    conspiracist = Conspiracist()
    the_boss = TheBoss()

    text_miner = TextMiner(api_key="MOCK_KEY") # Mock key for now

    # 1. Get Data
    print("Mining Data...")
    full_data = get_math_data(start_date=start_date, end_date=end_date)

    if full_data.empty:
        print("No data fetched. Aborting simulation.")
        return

    # Prepare results storage
    results = []

    # Training History for Boss
    boss_X_history = []
    boss_y_history = []

    # Simulation Loop
    current_date = pd.to_datetime(simulation_start_date)
    final_date = pd.to_datetime(end_date)

    print("Entering Time Machine...")

    step_count = 0

    while current_date < final_date:
        date_str = current_date.strftime('%Y-%m-%d')
        # print(f"Processing {date_str}...")

        # 1. Get historical context (training window)
        history = full_data[:current_date]
        if len(history) < 60:
            current_date += pd.Timedelta(days=7)
            continue

        # 2. Train Idiots (Mock/Simplified)
        # In a real full run, we'd retrain here.

        # 3. Predict Tomorrow
        # Prepare input for TrendSetter
        # We need to make sure we use correct columns.
        # full_data has GOOGL, MSFT, T10Y2Y, VIXCLS (4 columns)

        recent_data = history.iloc[-60:].values
        recent_data_reshaped = np.expand_dims(recent_data, axis=0)

        # Fix input shape mismatch dynamically
        try:
             # Rebuild model if shape mismatch
            if trend_setter.model.input_shape[2] != recent_data.shape[1]:
                 trend_setter.model = trend_setter._build_model_dynamic(recent_data.shape[1])

            trend_pred = trend_setter.predict(recent_data_reshaped)[0][0]
        except Exception as e:
            # print(f"TrendSetter failed: {e}")
            trend_pred = history['GOOGL'].iloc[-1] if 'GOOGL' in history else 0 # Fallback


        # Fed Watcher
        macro_cols = ['T10Y2Y', 'VIXCLS']
        # Ensure we only pick existing columns
        available_macro = [c for c in macro_cols if c in history.columns]
        if available_macro:
            current_macro = history[available_macro].iloc[-1:].values
            # Fed Watcher expects trained model.
            # We train it on history first if not trained?
            # Or assume pre-trained. Let's train on history periodically.
            if step_count % 10 == 0: # Train every 10 steps
                # Mock target: next week return of market or just volatility
                # Let's say target is VIX next week
                y_train = history['VIXCLS'].shift(-1).dropna()
                X_train = history[available_macro].iloc[:-1] # Align with y
                # Trim to match
                min_len = min(len(X_train), len(y_train))
                fed_watcher.train(X_train.iloc[:min_len], y_train.iloc[:min_len])

            try:
                fed_pred = fed_watcher.predict(current_macro)[0]
            except:
                fed_pred = 0
        else:
             fed_pred = 0

        # Sentiment
        headlines = text_miner.get_headlines(current_date.year, current_date.month)
        headline_texts = [h['headline'] for h in headlines]

        hype_score = hype_man.analyze_news(headline_texts, dossier)
        doom_score = doomsayer.analyze_news(headline_texts, dossier)

        # Conspiracy
        conspiracy_flag = conspiracist.analyze_insider_activity(100, 125) # Mock

        # 4. Boss Decision
        idiot_outputs = {
            "Conspiracy": conspiracy_flag,
            "Doom": doom_score,
            "Fed": fed_pred,
            "Hype": hype_score,
            "Trend": trend_pred,
        }

        # Train Boss periodically
        # Only train if we have history AND no NaN/Inf values
        if step_count > 10 and step_count % 5 == 0:
             # We need valid X and y history
             # Filter out bad data

             X_clean = []
             y_clean = []

             for i, x in enumerate(boss_X_history):
                 y = boss_y_history[i]
                 if np.isfinite(y): # Check y
                     # Check X
                     vals = list(x.values())
                     if all(np.isfinite(v) for v in vals):
                         X_clean.append(x)
                         y_clean.append(y)

             if len(X_clean) > 5:
                try:
                    the_boss.train(X_clean, y_clean)
                except Exception as e:
                    print(f"Boss training failed: {e}")

        boss_pred, uncertainty = the_boss.make_decision(idiot_outputs)

        # Store for Boss training next time
        # Actual price today (or next week).
        # We are predicting "tomorrow" or "next week".
        # So we store current idiot outputs vs *Future* price.
        # But we can only train on *past* idiot outputs vs *past* price.
        # For this loop, we just store current outputs and we will append the *result* later?
        # No, easier: We append (idiot_outputs, current_price) assuming idiots predict *current* price (nowcasting)
        # OR if they predict *next* price, we must pair outputs[t] with price[t+1].
        # For simplicity, let's assume nowcasting or short term.
        current_price = history['GOOGL'].iloc[-1] if 'GOOGL' in history else 0

        # Check for NaN in current_price
        if pd.isna(current_price):
            current_price = 0

        boss_X_history.append(idiot_outputs)
        boss_y_history.append(current_price)

        # Store Result
        results.append({
            "Date": date_str,
            "Boss_Prediction": boss_pred,
            "Uncertainty": uncertainty,
            **idiot_outputs
        })

        # Step Forward
        current_date += pd.Timedelta(days=7) # Weekly step
        step_count += 1

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("simulation_results.csv", index=False)
    print("Simulation Complete. Results saved to simulation_results.csv")

if __name__ == "__main__":
    run_simulation()

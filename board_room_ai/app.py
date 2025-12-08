# app.py
import streamlit as st
import pandas as pd
import time

st.set_page_config(layout="wide", page_title="The AI Board Room")

st.title("The AI Board Room: 2007 Crash Simulator")

# Load data if available
try:
    results_df = pd.read_csv("simulation_results.csv")
    data_loaded = True
except FileNotFoundError:
    st.warning("No simulation results found. Please run the simulation first.")
    results_df = pd.DataFrame()
    data_loaded = False

if data_loaded:
    # Sidebar for controls
    st.sidebar.header("Controls")

    # Auto-play or Slider
    if 'index' not in st.session_state:
        st.session_state.index = 0

    auto_play = st.sidebar.checkbox("Auto-Play Simulation", value=False)
    speed = st.sidebar.slider("Playback Speed (s)", 0.1, 2.0, 0.5)

    if auto_play:
        if st.session_state.index < len(results_df) - 1:
            st.session_state.index += 1
            time.sleep(speed)
            st.rerun()
    else:
        st.session_state.index = st.sidebar.slider("Timeline", 0, len(results_df)-1, st.session_state.index)

    # Get current step data
    row = results_df.iloc[st.session_state.index]
    current_date = row['Date']

    st.subheader(f"Date: {current_date}")

    # Visualizing the Idiots
    col1, col2, col3, col4 = st.columns(4)

    # Trend Setter
    trend_val = row.get("Trend", 0)
    col1.metric("The Trend Setter", f"${trend_val:.2f}", "Bullish" if trend_val > 500 else "Bearish")

    # Fed Watcher
    fed_val = row.get("Fed", 0)
    col2.metric("The Fed Watcher", f"{fed_val:.2f}", "Warning" if fed_val < 0 else "Stable")

    # Hype Man
    hype_val = row.get("Hype", 0)
    col3.metric("The Hype Man", f"{hype_val:.2f}", "Ecstatic" if hype_val > 0.5 else "Neutral")

    # Doomsayer
    doom_val = row.get("Doom", 0)
    col4.metric("The Doomsayer", f"{doom_val:.2f}", "PANIC" if doom_val < -0.5 else "Calm")

    # The Boss's Verdict
    st.header("The Boss's Decision")
    boss_pred = row.get("Boss_Prediction", 0)
    uncertainty = row.get("Uncertainty", 0)

    st.metric("Boss Prediction", f"${boss_pred:.2f}")

    # Confidence level (Inverse of uncertainty for visualization)
    confidence = max(0, min(100, 100 - (uncertainty * 100))) # Mock logic for confidence
    st.write(f"Confidence: {confidence:.0f}%")
    st.progress(int(confidence))

    if boss_pred < 400: # Mock threshold
         st.error("Verdict: Market Crash Imminent.")
    else:
         st.success("Verdict: Market looks good.")

    st.write("Reasoning: 'Fed Watcher' and 'Doomsayer' signals align with high volatility.")

    # Historical Chart
    st.subheader("Market History & Predictions")
    history_so_far = results_df.iloc[:st.session_state.index+1]
    st.line_chart(history_so_far.set_index("Date")[["Boss_Prediction", "Trend"]])

else:
    st.write("Run `python board_room_ai/main.py` to generate data.")

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

class FedWatcher:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        """
        X_train: Macro indicators (Yield Curve, VIX, etc.)
        y_train: Target (e.g. Future Market Return or Recession Indicator)
        """
        self.model.fit(X_train, y_train)

    def predict(self, current_macro_data):
        """
        current_macro_data: Single row or batch of macro data
        """
        # Returns prediction (e.g., probability of downturn or predicted market level)
        return self.model.predict(current_macro_data)

    def get_personality_quote(self, yield_curve_val):
        if yield_curve_val < 0:
            return "The Yield Curve is inverting! The end is nigh!"
        else:
            return "Economy looks stable... for now."

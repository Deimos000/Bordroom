import xgboost as xgb
import numpy as np
import pandas as pd

class TheBoss:
    def __init__(self):
        self.model = xgb.XGBRegressor(objective='reg:squarederror')
        self.weights = {"Trend": 1, "Fed": 1, "Hype": 1, "Doom": 1} # Dynamic weights
        self.is_trained = False

    def train(self, X_history, y_history):
        # X_history = Matrix of all idiots' past predictions
        # y_history = Actual stock prices

        # Ensure X_history is compatible
        if isinstance(X_history, list):
             # Convert list of dicts to dataframe/array
             # Assuming list of dicts: [{'Trend': ..., 'Fed': ...}, ...]
             if len(X_history) > 0 and isinstance(X_history[0], dict):
                 # Sort keys
                 keys = sorted(X_history[0].keys())
                 X_history = np.array([[x[k] for k in keys] for x in X_history])

        self.model.fit(X_history, y_history)
        self.is_trained = True

    def make_decision(self, idiot_outputs):
        # 1. Get raw prediction

        if isinstance(idiot_outputs, dict):
            # Sort keys to ensure consistent order
            features = [idiot_outputs[k] for k in sorted(idiot_outputs.keys())]
            input_data = np.array([features])
        else:
            input_data = idiot_outputs

        if not self.is_trained:
            # Return mean of inputs or dummy if not trained
            return np.mean(input_data), 0.0

        prediction = self.model.predict(input_data)

        # 2. Calculate Uncertainty (Variance between idiots)
        # Normalize outputs to same scale first!
        if isinstance(idiot_outputs, dict):
            predictions = [val for val in idiot_outputs.values()]
            uncertainty = np.std(predictions)
        else:
            uncertainty = 0 # Cannot calculate from opaque input

        return prediction[0], uncertainty

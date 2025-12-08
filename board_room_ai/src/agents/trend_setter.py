import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class TrendSetter:
    def __init__(self, lookback_period=60):
        self.lookback_period = lookback_period
        self.model = self._build_model()

    def _build_model(self, input_features=5):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.lookback_period, input_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def _build_model_dynamic(self, input_features):
        return self._build_model(input_features=input_features)

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, recent_data):
        # recent_data shape should be (1, 60, 5)
        # Returns float price prediction
        return self.model.predict(recent_data)

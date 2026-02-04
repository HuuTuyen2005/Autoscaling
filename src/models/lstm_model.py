"""
LSTM regression model for time-series forecasting.
Model-only logic (no data loading, no splitting).
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    input_shape,
    lstm_units=64,
    dropout=0.2,
    learning_rate=0.001
):
    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape, return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model


def train_lstm_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=20,
    batch_size=64,
    verbose=1
):
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if X_val is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )
    return model


def predict(model, X_test):
    return model.predict(X_test).flatten()

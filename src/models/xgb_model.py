# src/models/xgb_model.py
"""
XGBoost regression model for time-series forecasting.
This module contains ONLY model-related logic (no data loading, no splitting).
"""

import xgboost as xgb


def train_xgb_regressor(
    X_train,
    y_train,
    params=None,
):
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "objective": "reg:squarederror",
        }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    return model.predict(X_test)

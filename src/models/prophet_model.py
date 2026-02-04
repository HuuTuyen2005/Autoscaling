"""
Facebook Prophet model for time-series forecasting.
Model-only logic (no data loading, no splitting).
"""

import pandas as pd
from prophet import Prophet


def train_prophet_model(
    df_train: pd.DataFrame,
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
):
    """
    df_train must contain columns: ['ds', 'y']
    """
    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality
    )

    model.fit(df_train)
    return model


def predict(model, periods: int, freq: str = "min"):
    """
    Forecast future values.
    """
    future = model.make_future_dataframe(
        periods=periods,
        freq=freq
    )
    forecast = model.predict(future)
    return forecast

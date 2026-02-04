# src/features.py

"""
Reusable feature engineering module for autoscaling forecasting.
- Input: parsed log dataframe with timestamp index
- Output: metrics dataframe with lag, rolling, calendar features
- Safe for time-series forecasting (no data leakage)
"""

import pandas as pd


def resample_base_metrics(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Resample raw logs into base metrics."""
    metrics = pd.DataFrame()
    metrics["requests"] = df.resample(freq).size()
    metrics["bytes"] = df["bytes"].resample(freq).sum()
    metrics["error_rate"] = (
        df["status"].astype(str).str.startswith(("4", "5")).resample(freq).mean()
    )
    return metrics.fillna(0)


def add_lag_features(metrics: pd.DataFrame, lags=(1, 5, 10)) -> pd.DataFrame:
    for lag in lags:
        metrics[f"req_lag_{lag}"] = metrics["requests"].shift(lag)
        metrics[f"bytes_lag_{lag}"] = metrics["bytes"].shift(lag)
        metrics[f"err_lag_{lag}"] = metrics["error_rate"].shift(lag)
    return metrics


def add_rolling_features(metrics: pd.DataFrame, windows=(5, 10)) -> pd.DataFrame:
    metrics["req_roll_5"] = metrics["requests"].rolling(5).mean()
    metrics["req_roll_10"] = metrics["requests"].rolling(10).mean()
    metrics["bytes_roll_5"] = metrics["bytes"].rolling(5).mean()
    metrics["err_roll_5"] = metrics["error_rate"].rolling(5).mean()
    return metrics


def add_calendar_features(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics["hour"] = metrics.index.hour
    metrics["dayofweek"] = metrics.index.dayofweek
    metrics["is_weekend"] = (metrics["dayofweek"] >= 5).astype(int)
    return metrics


def add_future_targets(metrics: pd.DataFrame, targets=("requests", "bytes"), horizons=(1, 5, 15)) -> pd.DataFrame:
    for target in targets:
        for h in horizons:
            metrics[f"{target}_future_{h}"] = metrics[target].shift(-h)
    return metrics


def build_feature_table(df_logs: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Full feature pipeline."""
    metrics = resample_base_metrics(df_logs, freq)
    metrics = add_lag_features(metrics)
    metrics = add_rolling_features(metrics)
    metrics = add_calendar_features(metrics)
    metrics = add_future_targets(metrics)
    metrics = metrics.dropna()
    return metrics



# core/data_loader.py
import pandas as pd

def load_metrics(path="data/processed/metrics_full.csv"):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df

def load_forecast(path="data/processed/xgb_forecast_5m.csv"):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df

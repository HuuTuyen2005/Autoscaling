# src/metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = (abs((y_true - y_pred) / (y_true + 1e-6))).mean() * 100

    return {
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "MAPE (%)": mape
    }

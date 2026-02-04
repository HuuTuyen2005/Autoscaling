# src/split.py
import pandas as pd

def time_series_split(
    df: pd.DataFrame,
    split_date: str,
    label_col: str,
    drop_cols=("scale_action",)
):
    """
    Time-based split for forecasting models.
    """
    X = df.drop(columns=[label_col, *drop_cols], errors="ignore")
    y = df[label_col]

    train_mask = df.index < split_date

    X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
    y_train, y_test = y.loc[train_mask], y.loc[~train_mask]

    return X_train, X_test, y_train, y_test

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE)
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
    return 100 * np.mean(diff)

def compute_tp_window_errors(y_true: pd.Series, y_pred: pd.Series, tp_df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Evaluate model performance in the post-turning-point window.

    Parameters:
        y_true (pd.Series): Ground truth time series (indexed by day).
        y_pred (pd.Series): Forecasted time series (same index as y_true).
        tp_df (pd.DataFrame): Detected turning points, with 'index', 'type', and 'source'.
        window (int): Number of days after each TP to evaluate.

    Returns:
        pd.DataFrame with per-TP metrics: tp_index, type, source, MAE, RMSE, sMAPE.
    """
    results = []

    for _, row in tp_df.iterrows():
        tp_idx = int(row['index'])
        type_ = row['type']
        source = row['source']

        # Slice forecast and truth for the post-TP window
        y_true_slice = y_true.iloc[tp_idx+1 : tp_idx+1+window]
        y_pred_slice = y_pred.iloc[tp_idx+1 : tp_idx+1+window]

        if len(y_true_slice) < window or len(y_pred_slice) < window:
            continue  # skip TPs near the end of the series

        mae = mean_absolute_error(y_true_slice, y_pred_slice)
        rmse = np.sqrt(mean_squared_error(y_true_slice, y_pred_slice))
        smape_val = smape(y_true_slice.values, y_pred_slice.values)

        results.append({
            "tp_index": tp_idx,
            "type": type_,
            "source": source,
            "MAE": mae,
            "RMSE": rmse,
            "sMAPE": smape_val
        })

    return pd.DataFrame(results)

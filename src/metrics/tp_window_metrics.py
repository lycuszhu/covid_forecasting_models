import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_tp_alignment_errors(forecast_df: pd.DataFrame, tp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute TP-aligned forecasting errors (MAE, RMSE, sMAPE) for forecast sequences whose forecast_dates
    overlap with known turning points.

    Returns global average metrics as well as distribution stats across all matched sequences.
    """
    all_true = []
    all_pred = []

    # Store per-sequence metrics
    per_series_mae = []
    per_series_rmse = []
    per_series_smape = []

    def smape(y_true, y_pred):
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
        return 100 * np.mean(diff)

    tp_date_strs = set(tp_df.index.strftime('%Y-%m-%d'))

    for _, f_row in forecast_df.iterrows():
        forecast_dates = [d.split("'")[1][:10] for d in f_row['forecast_dates'].split(",")]

        if any(date in tp_date_strs for date in forecast_dates):
            y_true_seq = json.loads(f_row['true_values'])
            y_pred_seq = json.loads(f_row['predictions'])

            all_true.extend(y_true_seq)
            all_pred.extend(y_pred_seq)

            per_series_mae.append(mean_absolute_error(y_true_seq, y_pred_seq))
            per_series_rmse.append(np.sqrt(mean_squared_error(y_true_seq, y_pred_seq)))
            per_series_smape.append(smape(np.array(y_true_seq), np.array(y_pred_seq)))

    if not all_true:
        return pd.DataFrame([{"MAE": None, "RMSE": None, "sMAPE": None}])

    # Global average metrics
    mae = mean_absolute_error(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    smape_val = smape(np.array(all_true), np.array(all_pred))

    summary = {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape_val,
        "MAE_mean": np.mean(per_series_mae),
        "MAE_median": np.median(per_series_mae),
        "MAE_p25": np.percentile(per_series_mae, 25),
        "MAE_p75": np.percentile(per_series_mae, 75),
        "MAE_std": np.std(per_series_mae),
        "MAE_min": np.min(per_series_mae),
        "MAE_max": np.max(per_series_mae),
        "RMSE_mean": np.mean(per_series_rmse),
        "RMSE_median": np.median(per_series_rmse),
        "RMSE_p25": np.percentile(per_series_rmse, 25),
        "RMSE_p75": np.percentile(per_series_rmse, 75),
        "RMSE_std": np.std(per_series_rmse),
        "RMSE_min": np.min(per_series_rmse),
        "RMSE_max": np.max(per_series_rmse),
        "sMAPE_mean": np.mean(per_series_smape),
        "sMAPE_median": np.median(per_series_smape),
        "sMAPE_p25": np.percentile(per_series_smape, 25),
        "sMAPE_p75": np.percentile(per_series_smape, 75),
        "sMAPE_std": np.std(per_series_smape),
        "sMAPE_min": np.min(per_series_smape),
        "sMAPE_max": np.max(per_series_smape),
    }

    return pd.DataFrame([summary])

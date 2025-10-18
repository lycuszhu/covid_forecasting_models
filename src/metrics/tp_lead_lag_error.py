import pandas as pd
import numpy as np
import json
import os
import sys

# Add the directory containing this file to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from combined_tp_extraction import extract_combined_turning_points


def compute_tp_lead_lag(forecast_df: pd.DataFrame,
                        true_tps: pd.DataFrame,
                        y_true: pd.Series) -> pd.DataFrame:
    """
    Compute the lead-lag error between true and predicted turning points across forecast_df rows.

    Parameters:
        forecast_df (pd.DataFrame): DataFrame containing forecast results (with stringified lists).
        true_tps (pd.DataFrame): Ground-truth TPs, indexed by date, with ['index', 'type', 'source'] columns.
        y_true (pd.Series): Full original ground-truth time series (indexed by datetime).

    Returns:
        pd.DataFrame: One row per matched TP, recording match status and lead-lag error.
    """

    def embed_forecast(y_true: pd.Series, y_pred: list, forecast_start: int) -> pd.Series:
        """Embed predicted values into the true series starting at forecast_start index."""
        y_embedded = y_true.copy()
        y_embedded.iloc[forecast_start:forecast_start + len(y_pred)] = y_pred
        return y_embedded

    all_results = []

    for _, row in forecast_df.iterrows():
        try:
            y_pred = json.loads(row["predictions"])
            forecast_dates = [pd.to_datetime(d.split("'")[1][:10]) for d in row["forecast_dates"].split(",")]
        except Exception as e:
            print(f"⚠️ Skipping row due to parsing error: {e}")
            continue

        # Get start index from y_true
        try:
            forecast_start_date = forecast_dates[0]
            forecast_end_date = forecast_dates[-1]
            forecast_start = y_true.index.get_loc(forecast_start_date)
        except Exception as e:
            print(f"⚠️ Skipping row due to date mismatch: {e}")
            continue

        # Only proceed if at least one TP falls within forecast window
        forecast_window = pd.DatetimeIndex(forecast_dates)
        if not true_tps.index.isin(forecast_window).any():
            continue

        # Embed predictions into true series
        y_embedded = embed_forecast(y_true, y_pred, forecast_start)

        # Extract TPs from embedded series
        pred_tps = extract_combined_turning_points(y_embedded)
        pred_tps["date"] = y_embedded.index[pred_tps["index"]].values
        pred_tps.set_index("date", inplace=True)

        # Match each TP
        for tp_date, tp in true_tps.iterrows():
            tp_type = tp["type"]

            if tp_date not in forecast_window:
                continue

            #match_window = len(y_pred)
            #window_start = tp_date - pd.Timedelta(days=match_window)
            #window_end = tp_date + pd.Timedelta(days=match_window)

            candidates = pred_tps[
                #(pred_tps.index >= window_start) &
                #(pred_tps.index <= window_end) &
                (pred_tps.index >= forecast_start_date) &
                (pred_tps.index <= forecast_end_date) &
                (pred_tps["type"] == tp_type)
            ]

            if not candidates.empty:
                closest_idx = np.abs(candidates.index - tp_date).argmin()
                matched_date = candidates.index[closest_idx]
                lag_error = (matched_date - tp_date).days

                all_results.append({
                    "tp_date": tp_date,
                    "tp_type": tp_type,
                    "matched": True,
                    "matched_date": matched_date,
                    "lead_lag_days": lag_error
                })
            else:
                all_results.append({
                    "tp_date": tp_date,
                    "tp_type": tp_type,
                    "matched": False,
                    "matched_date": None,
                    "lead_lag_days": None
                })


    return pd.DataFrame(all_results)

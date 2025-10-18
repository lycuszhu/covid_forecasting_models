import pandas as pd
import numpy as np
from src.metrics.combined_tp_extraction import extract_combined_turning_points

def embed_forecast(y_true: pd.Series, y_pred: pd.Series, forecast_start: int) -> pd.Series:
    '''
    Replace the true values in the forecast window with predicted values.

    Parameters:
        y_true (pd.Series): Full true time series.
        y_pred (pd.Series): Forecasted values (length H).
        forecast_start (int): Index in y_true where forecast begins.

    Returns:
        pd.Series with forecast embedded into the original time series.
    '''
    y_combined = y_true.copy()
    forecast_end = forecast_start + len(y_pred)
    y_combined.iloc[forecast_start:forecast_end] = y_pred.values
    return y_combined

def compute_tp_lead_lag(true_tps: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series, forecast_start: int, 
                        match_window: int = 14, T: int = 28) -> pd.DataFrame:
    '''
    Compare forecast-based TPs to ground truth TPs using lead-lag error.

    Parameters:
        true_tps (pd.DataFrame): True turning points with 'index' and 'type'.
        y_true (pd.Series): Full true series.
        y_pred (pd.Series): Forecast series (length H).
        forecast_start (int): Index in y_true where forecast begins.
        match_window (int): +/- window around true TP to search for matches.
        T (int): Trend window used in rsd_ttest.

    Returns:
        pd.DataFrame with match status and lead-lag error.
    '''
    def embed_forecast(y_true: pd.Series, y_pred: pd.Series, forecast_start: int) -> pd.Series:
        '''
        Replace the true values in the forecast window with predicted values.

        Parameters:
            y_true (pd.Series): Full true time series.
            y_pred (pd.Series): Forecasted values (length H).
            forecast_start (int): Index in y_true where forecast begins.

        Returns:
            pd.Series with forecast embedded into the original time series.
        '''
        y_combined = y_true.copy()
        forecast_end = forecast_start + len(y_pred)
        y_combined.iloc[forecast_start:forecast_end] = y_pred.values
        return y_combined
    
    # Step 1: Embed forecast into ground truth
    y_with_forecast = embed_forecast(y_true, y_pred, forecast_start)

    # Step 2: Detect TPs in forecast-augmented series
    pred_tps = extract_combined_turning_points(y_with_forecast, T=T)
    pred_tps = pred_tps[pred_tps["index"] >= forecast_start]  # restrict to forecast region

    # Step 3: Match predicted TPs to true TPs
    results = []
    for _, row in true_tps.iterrows():
        tp_idx = int(row["index"])
        tp_type = row["type"]

        # Skip if TP is outside the forecast + buffer range
        if not (forecast_start <= tp_idx <= forecast_start + len(y_pred) + match_window):
            continue

        # Find predicted TPs of the same type within match window
        candidates = pred_tps[pred_tps["type"] == tp_type]
        candidates = candidates[
            (candidates["index"] >= tp_idx - match_window) &
            (candidates["index"] <= tp_idx + match_window)
        ]

        if len(candidates) > 0:
            closest = candidates.iloc[(candidates["index"] - tp_idx).abs().argmin()]
            lead_lag = int(closest["index"]) - tp_idx
            matched = True
        else:
            closest = None
            lead_lag = None
            matched = False

        results.append({
            "tp_index": tp_idx,
            "type": tp_type,
            "matched": matched,
            "matched_index": int(closest["index"]) if matched else None,
            "lead_lag": lead_lag
        })

    return pd.DataFrame(results)

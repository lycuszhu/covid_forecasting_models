import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LinearRegression

def compute_tp_and_global_slope_agreement(forecast_df: pd.DataFrame, tp_df: pd.DataFrame, series: pd.Series) -> pd.DataFrame:
    tp_df = tp_df[tp_df["source"] == "find_peaks"]
    tp_date_strs = set(tp_df.index.strftime('%Y-%m-%d'))

    def compute_slope(series_vals):
        X = np.arange(len(series_vals)).reshape(-1, 1)
        y = np.array(series_vals).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return float(model.coef_[0])
    
    def angle_between_slopes(m1, m2):
        numerator = m1 - m2
        denominator = 1 + m1 * m2
        angle_rad = np.arctan2(numerator, denominator)  # Handles signs correctly
        angle_deg = np.degrees(angle_rad)
        return abs(angle_deg)

    tp_same_sign = tp_diff_sign = 0
    global_same_sign = global_diff_sign = 0
    global_slope_diffs = []
    tp_slope_diffs = []

    for _, row in forecast_df.iterrows():
        forecast_dates = [d.split("'")[1][:10] for d in row['forecast_dates'].split(",")]
        forecast_dates_dt = pd.to_datetime(forecast_dates)

        smoothed = series.rolling(7, center=True, min_periods=1).mean().rolling(7, center=True, min_periods=1).mean()
        y_true = smoothed.loc[forecast_dates_dt].values

        y_pred = json.loads(row["predictions"])
        new_series = series.copy()
        new_series.loc[forecast_dates_dt] = y_pred
        new_smoothed = new_series.rolling(7, center=True, min_periods=1).mean().rolling(7, center=True, min_periods=1).mean()
        y_pred_smoothed = new_smoothed.loc[forecast_dates_dt].values

        true_slope = compute_slope(y_true)
        pred_slope = compute_slope(y_pred_smoothed)
        slope_angle_diff = angle_between_slopes(true_slope, pred_slope)
        global_slope_diffs.append(slope_angle_diff)

        if np.sign(true_slope) == np.sign(pred_slope):
            global_same_sign += 1
        else:
            global_diff_sign += 1

        #if forecast_dates[0] in tp_date_strs:
        if any(date in tp_date_strs for date in forecast_dates):
            tp_slope_diffs.append(slope_angle_diff)
            if np.sign(true_slope) == np.sign(pred_slope):
                tp_same_sign += 1
            else:
                tp_diff_sign += 1

    def summary_stats(arr):
        if len(arr) == 0:
            return [None] * 7
        arr = np.array(arr)
        return [
            np.mean(arr), np.median(arr),
            np.percentile(arr, 25), np.percentile(arr, 75),
            np.std(arr), np.min(arr), np.max(arr)
        ]

    tp_total = tp_same_sign + tp_diff_sign
    global_total = global_same_sign + global_diff_sign
    tp_diff_pct = (tp_diff_sign / tp_total * 100) if tp_total > 0 else None
    global_diff_pct = (global_diff_sign / global_total * 100) if global_total > 0 else None

    g_mean, g_med, g_p25, g_p75, g_std, g_min, g_max = summary_stats(global_slope_diffs)
    tp_mean, tp_med, tp_p25, tp_p75, tp_std, tp_min, tp_max = summary_stats(tp_slope_diffs)

    return pd.DataFrame([{
        "tp_same_sign": tp_same_sign,
        "tp_diff_sign": tp_diff_sign,
        "tp_diff_percentage": round(tp_diff_pct, 2) if tp_diff_pct is not None else None,
        "tp_slope_diff_stats": (tp_mean, tp_med, tp_p25, tp_p75, tp_std, tp_min, tp_max),
        "global_same_sign": global_same_sign,
        "global_diff_sign": global_diff_sign,
        "global_diff_percentage": round(global_diff_pct, 2) if global_diff_pct is not None else None,
        "global_slope_diff_stats": (g_mean, g_med, g_p25, g_p75, g_std, g_min, g_max)
    }])

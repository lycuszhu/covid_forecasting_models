import pandas as pd
import numpy as np
import os
import sys

# Add the directory containing this file to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from find_tp_naive import extract_turning_points
from find_tp_rsd_ttest import rsd_ttest

def extract_combined_turning_points(series, T=28, alpha=0.05, min_separation=7, peak_prominence=50, distance=7,
                                    min_gap_days=14, smooth_window=7, double_smooth=True):
    '''
    Combines RSD t-test (Zuo) and find_peaks methods for robust turning point extraction.

    Parameters:
        series (pd.Series): Raw time series.
        T (int): Trend phase length for RSD t-test.
        alpha (float): Significance level for RSD t-test.
        min_separation (int): Min distance to consider RSD point redundant if close to find_peaks TP.
        peak_prominence (float): Prominence threshold for find_peaks.
        smooth_window (int): Window size for rolling smoothing.
        double_smooth (bool): Whether to apply smoothing twice.

    Returns:
        pd.DataFrame with final TPs: index, type ('peak', 'trough', 'up', 'down'), source ('find_peaks', 'rsd')
    '''
    # Apply smoothing
    smoothed = series.rolling(smooth_window, center=True, min_periods=1).mean()
    if double_smooth:
        smoothed = smoothed.rolling(smooth_window, center=True, min_periods=1).mean()

    smoothed = smoothed.dropna()
    rsd_results = rsd_ttest(smoothed, T=T, alpha=alpha)
    rsd_indices = set(rsd_results["index"].values)

    # Use naive peak/trough detection (already smoothed)
    naive_tps = extract_turning_points(smoothed, peak_prominence=peak_prominence, distance=distance, min_gap_days=min_gap_days)
    naive_peaks = naive_tps[naive_tps['type'] == 'peak']['index'].values
    naive_troughs = naive_tps[naive_tps['type'] == 'trough']['index'].values

    final_tp_indices = []
    final_tp_types = []
    final_tp_sources = []

    for idx in naive_peaks:
        final_tp_indices.append(int(idx))
        final_tp_types.append("peak")
        final_tp_sources.append("find_peaks")

    for idx in naive_troughs:
        final_tp_indices.append(int(idx))
        final_tp_types.append("trough")
        final_tp_sources.append("find_peaks")

    for _, row in rsd_results.iterrows():
        if not any(abs(row["index"] - tp_idx) <= min_separation for tp_idx in np.concatenate((naive_peaks, naive_troughs))):
            final_tp_indices.append(row["index"])
            tp_type = "up" if row["slope_diff"] > 0 else "down"
            final_tp_types.append(tp_type)
            final_tp_sources.append("rsd")

    return pd.DataFrame({
        "index": final_tp_indices,
        "type": final_tp_types,
        "source": final_tp_sources
    }).sort_values("index").reset_index(drop=True).astype({"index": int})

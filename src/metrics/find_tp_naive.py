import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def extract_turning_points(series, peak_prominence=5, distance=7, min_gap_days=14):
    '''
    Detect peaks and troughs using find_peaks, with post-processing to enforce minimum spacing.

    Parameters:
        series (pd.Series): Input series (already smoothed).
        peak_prominence (float): Prominence for find_peaks.
        distance (int): Minimum distance between peaks.
        min_gap_days (int): Minimum distance between any two turning points (peaks or troughs).

    Returns:
        pd.DataFrame with 'index', 'value', and 'type' columns.
    '''
    smoothed = series.dropna()

    peak_indices, peak_props = find_peaks(smoothed, prominence=peak_prominence, distance=distance)
    trough_indices, trough_props = find_peaks(-smoothed, prominence=peak_prominence, distance=distance)

    candidates = []

    for idx, prom in zip(peak_indices, peak_props["prominences"]):
        candidates.append({"index": idx, "value": smoothed.iloc[idx], "type": "peak", "prominence": prom})
    for idx, prom in zip(trough_indices, trough_props["prominences"]):
        candidates.append({"index": idx, "value": smoothed.iloc[idx], "type": "trough", "prominence": prom})

    candidates = sorted(candidates, key=lambda x: x["index"])

    final_tps = []
    for tp in sorted(candidates, key=lambda x: -x["prominence"]):
        if all(abs(tp["index"] - accepted["index"]) >= min_gap_days for accepted in final_tps):
            final_tps.append(tp)

    return pd.DataFrame(final_tps)

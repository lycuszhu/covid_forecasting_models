import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression


def compute_tp_slope_direction(y_true: pd.Series, y_pred: pd.Series, tp_df: pd.DataFrame, T: int = 7) -> pd.DataFrame:
    """
    Compare post-TP trend direction between ground truth and forecast using a slope-alignment t-test.

    Parameters:
        y_true (pd.Series): Ground truth time series (indexed by day).
        y_pred (pd.Series): Forecasted time series (same index as y_true).
        tp_df (pd.DataFrame): Detected turning points with 'index', 'type', and 'source'.
        T (int): Number of days after each TP to compute trend slope.

    Returns:
        pd.DataFrame with TP index, slope values, direction agreement, and significance test result.
    """
    results = []

    def compute_slope(series, start, end):
        """
        Fit a linear regression to the data between indices [start, end)
        and return the slope of the fitted line.
        """
        X = np.arange(start, end).reshape(-1, 1)
        y = series.iloc[start:end].values
        model = LinearRegression().fit(X, y)
        return model.coef_[0]

    for _, row in tp_df.iterrows():
        tp_idx = int(row['index'])
        type_ = row['type']
        source = row['source']

        # Check if post-TP window exceeds bounds
        if tp_idx + T + 1 >= len(y_true) or tp_idx + T + 1 >= len(y_pred):
            continue

        # Extract post-TP windows
        y_true_win = y_true.iloc[tp_idx+1 : tp_idx+1+T]
        y_pred_win = y_pred.iloc[tp_idx+1 : tp_idx+1+T]

        # Compute slopes
        slope_true = compute_slope(y_true, tp_idx+1, tp_idx+1+T)
        slope_pred = compute_slope(y_pred, tp_idx+1, tp_idx+1+T)

        # Approximate slope difference significance via residual-based t-test
        residuals_true = y_true_win - np.poly1d(np.polyfit(np.arange(T), y_true_win.values, 1))(np.arange(T))
        residuals_pred = y_pred_win - np.poly1d(np.polyfit(np.arange(T), y_pred_win.values, 1))(np.arange(T))
        t_stat, p_value = ttest_ind(residuals_true, residuals_pred, equal_var=False)

        results.append({
            "tp_index": tp_idx,
            "type": type_,
            "source": source,
            "actual_slope": slope_true,
            "pred_slope": slope_pred,
            "same_sign": np.sign(slope_true) == np.sign(slope_pred),
            "p_value": p_value,
            "significant_diff": p_value < 0.05
        })

    return pd.DataFrame(results)

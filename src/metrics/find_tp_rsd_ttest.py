import numpy as np
import pandas as pd
from scipy.stats import t

def effective_dof(series1, series2):
    """
    Calculate effective degrees of freedom considering autocorrelation
    using Bartlett’s formula.
    """
    N = len(series1) + len(series2)
    acf = pd.Series(np.concatenate([series1, series2])).autocorr()
    return N / (1 + 2 * acf * (1 - np.arange(1, N) / N).sum())

def compute_slope(series):
    """
    Compute least squares slope of a series.
    """
    x = np.arange(len(series))
    A = np.vstack([x, np.ones(len(series))]).T
    m, _ = np.linalg.lstsq(A, series, rcond=None)[0]
    return m

def slope_diff_t_stat(y, z):
    """
    Compute t-statistic comparing slopes of series y and z.
    """
    n, m = len(y), len(z)
    x_y = np.arange(n)
    x_z = np.arange(m)

    beta_y = compute_slope(y)
    beta_z = compute_slope(z)

    y_hat = beta_y * x_y + np.mean(y) - beta_y * np.mean(x_y)
    z_hat = beta_z * x_z + np.mean(z) - beta_z * np.mean(x_z)

    res_y = y - y_hat
    res_z = z - z_hat

    N = np.sum((x_y - np.mean(x_y)) ** 2)
    M = np.sum((x_z - np.mean(x_z)) ** 2)
    C = N * M / (N + M)

    s2 = (np.sum(res_y ** 2) + np.sum(res_z ** 2)) / (n + m - 4)
    t_stat = (beta_y - beta_z) / np.sqrt(s2 / C)

    # Effective degrees of freedom
    dof = effective_dof(y, z)
    p_val = 2 * (1 - t.cdf(abs(t_stat), df=dof))
    return t_stat, p_val

def rsd_ttest(series, T=50, tau=None, alpha=0.01):
    """
    Full RSD t-test implementation.
    series: input time series (numpy array or pd.Series)
    T: desired minimum trend phase length
    tau: half-window size, default = T-2
    alpha: significance level
    """
    if isinstance(series, pd.Series):
        series = series.values
    series = np.asarray(series)
    k = len(series)
    tau = tau if tau else T - 2

    potential_points = []
    slope_diffs = []

    for a in range(tau + 1, k - tau):
        left = series[a - tau:a]
        right = series[a + 1:a + 1 + tau]

        t_stat, p_val = slope_diff_t_stat(left, right)
        slope_diff = compute_slope(right) - compute_slope(left)

        if p_val < alpha:
            potential_points.append((a, slope_diff, p_val))

    # Identify max slope difference in each contiguous cluster
    tps = []
    if potential_points:
        current_cluster = [potential_points[0]]
        for i in range(1, len(potential_points)):
            if potential_points[i][0] == potential_points[i-1][0] + 1:
                current_cluster.append(potential_points[i])
            else:
                # Save max slope diff in cluster
                max_pt = max(current_cluster, key=lambda x: abs(x[1]))
                tps.append(max_pt)
                current_cluster = [potential_points[i]]
        if current_cluster:
            max_pt = max(current_cluster, key=lambda x: abs(x[1]))
            tps.append(max_pt)

    return pd.DataFrame({
        "index": [i[0] for i in tps],
        "slope_diff": [i[1] for i in tps],
        "p_value": [i[2] for i in tps]
    })

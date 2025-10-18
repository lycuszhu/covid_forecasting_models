import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter1d

def build_curve_db(series_list, match_length=28, forecast_length=28):
    """
    Build a normalized curve database from a list of incidence time series.

    Parameters:
    - series_list: list of 1D np.ndarray, each representing a full incidence series
    - match_length: int, length of the matching window (default 28)
    - forecast_length: int, length of the forecast window (default 28)

    Returns:
    - np.ndarray of shape (n_windows, match_length + forecast_length)
    """
    window_size = match_length + forecast_length
    segments = []
    for series in series_list:
        series = np.asarray(series).astype(float)
        if len(series) < window_size:
            continue
        for i in range(len(series) - window_size + 1):
            window = series[i:i + window_size].copy()
            ref_mean = np.mean(window[:match_length])
            if ref_mean == 0:
                continue
            window /= ref_mean
            segments.append(window)
    return np.stack(segments) if segments else np.empty((0, window_size))

def construct_i_and_seasonality(series, smoothing_sigma=2.0):
    """
    Construct i_original, i_restored, and seasonality from a raw incidence series.

    Parameters:
    - series: np.ndarray, original raw incidence
    - smoothing_sigma: float, standard deviation for Gaussian smoothing

    Returns:
    - i_original: np.ndarray, original input series
    - i_restored: np.ndarray, smoothed/restored incidence
    - seasonality: np.ndarray, normalized 7-day seasonal profile (length 7, mean=1.0)
    """
    i_original = np.asarray(series).astype(float)
    i_restored = gaussian_filter1d(i_original, sigma=smoothing_sigma, mode='nearest')
    
    # Compute 7-day seasonal profile
    seasonality = np.array([np.mean(i_original[i::7]) for i in range(7)])
    
    # Normalize so the average seasonality is 1.0
    mean_seasonality = np.mean(seasonality)
    if mean_seasonality > 0:
        seasonality = seasonality / mean_seasonality
    else:
        seasonality = np.ones(7)  # fallback if division would cause NaN

    return i_original, i_restored, seasonality

def incidence_forecast_by_learning(ir, q, ir_database, lambda_, mu, last_incidence_date, match_length=28, forecast_length=28):
    """
    Replicates the IncidenceForecastByLearning function from EpiLearn in Python.

    Parameters:
    - ir: np.ndarray, restored incidence curve (1D)
    - q: np.ndarray, weekly bias correction factors (same length as ir)
    - ir_database: np.ndarray of shape (n_curves, 56)
    - lambda_: float, weighting decay parameter for distance
    - mu: float, exponential decay parameter for distance computation
    - last_incidence_date: str, in 'YYYY-MM-DD' format

    Returns:
    - ir_forecast: np.ndarray of length 28 (restored forecast)
    - i0_forecast: np.ndarray of length 28 (bias-corrected forecast)
    - dates: list of date strings for each forecast day
    """
    assert len(ir) >= match_length, "Input series too short for matching"

    u = ir[-match_length:].copy()
    u_mean = np.mean(u)
    u0 = u[-1]
    u /= u_mean

    total_length = match_length + forecast_length
    v = np.zeros(total_length)
    for db_curve in ir_database:
        dist = np.sum(np.abs(u - db_curve[:match_length]) * np.exp(-mu * np.arange(match_length - 1, -1, -1))) / match_length
        weight = np.exp(-lambda_ * dist)
        v += weight * db_curve

    v *= u0 / v[match_length - 1]
    ir_forecast = v[match_length:total_length].copy()
    i0_forecast = np.array([ir_forecast[k] / q[-7 + k % 7] for k in range(forecast_length)])

    base_date = datetime.strptime(last_incidence_date, "%Y-%m-%d")
    dates = [(base_date + timedelta(days=k + 1)).strftime("%Y-%m-%d") for k in range(forecast_length)]
    return ir_forecast, i0_forecast, dates

def incidence_forecast_by_learning_median(ir, q, ir_database, mu, last_incidence_date, weighted=False, match_length=28, forecast_length=28):
    """
    Median-based forecast version from EpiLearn.

    Parameters:
    - ir: np.ndarray, restored incidence curve (1D)
    - q: np.ndarray, weekly bias correction factors
    - ir_database: np.ndarray of shape (n_curves, 56)
    - mu: float, exponential decay parameter
    - last_incidence_date: str, in 'YYYY-MM-DD' format
    - weighted: bool, if True, compute weighted median; else simple median

    Returns:
    - ir_forecast: np.ndarray of length 28
    - i0_forecast: np.ndarray of length 28
    - dates: list of forecast dates
    """
    assert len(ir) >= match_length, "Input series too short for matching"

    u = ir[-match_length:].copy()
    u_mean = np.mean(u)
    u0 = u[-1]
    u /= u_mean

    distances = []
    weights = []
    for db_curve in ir_database:
        dist = np.sum(np.abs(u - db_curve[:match_length]) * np.exp(-mu * np.arange(match_length - 1, -1, -1))) / match_length
        distances.append(dist)
        weights.append(np.exp(-dist / 0.0475))

    distances = np.array(distances)
    weights = np.array(weights)

    forecasts = np.stack([db_curve[match_length:match_length + forecast_length] for db_curve in ir_database])

    if weighted:
        def weighted_median(values, weights):
            sorted_idx = np.argsort(values)
            values, weights = values[sorted_idx], weights[sorted_idx]
            cum_weights = np.cumsum(weights)
            cutoff = np.sum(weights) / 2.0
            return values[np.searchsorted(cum_weights, cutoff)]

        ir_forecast = np.array([
            weighted_median(forecasts[:, k], weights) for k in range(forecast_length)
        ])
    else:
        ir_forecast = np.median(forecasts, axis=0)

    ir_forecast *= u0 / ir_forecast[0]
    i0_forecast = np.array([ir_forecast[k] / q[-7 + k % 7] for k in range(forecast_length)])

    base_date = datetime.strptime(last_incidence_date, "%Y-%m-%d")
    dates = [(base_date + timedelta(days=k + 1)).strftime("%Y-%m-%d") for k in range(forecast_length)]
    return ir_forecast, i0_forecast, dates

def epiinvert_forecast(
    raw_series,
    curve_db,
    last_incidence_date,
    forecast_type="median",
    smoothing_sigma=2.0,
    lambda_=108.0,
    mu=0.0675,
    trend_sentiment=0.0,
    match_length=28,
    forecast_length=28
):
    """
    Full wrapper for EpiLearn forecasting in Python.

    Parameters:
    - raw_series: np.ndarray, raw incidence curve
    - curve_db: np.ndarray, shape (n_curves, 56), historical normalized segments
    - last_incidence_date: str, 'YYYY-MM-DD'
    - forecast_type: str, one of ['mean', 'median', 'weightedmedian']
    - smoothing_sigma: float, for Gaussian smoothing
    - lambda_: float, distance weighting decay (used in mean forecast)
    - mu: float, exponential decay in distance calculation
    - trend_sentiment: float, (placeholder, not currently applied)

    Returns:
    - ir_forecast: np.ndarray, 28-day restored forecast
    - i0_forecast: np.ndarray, 28-day bias-corrected forecast
    - dates: list of str, forecast dates
    """
    i_original, i_restored, seasonality = construct_i_and_seasonality(raw_series, smoothing_sigma)

    if forecast_type == "mean":
        return incidence_forecast_by_learning(
            i_restored,
            seasonality,
            curve_db,
            lambda_,
            mu,
            last_incidence_date,
            match_length,
            forecast_length
        )
    elif forecast_type == "median":
        return incidence_forecast_by_learning_median(
            i_restored,
            seasonality,
            curve_db,
            mu,
            last_incidence_date,
            weighted=False,
            match_length=match_length,
            forecast_length=forecast_length
        )
    elif forecast_type == "weightedmedian":
        return incidence_forecast_by_learning_median(
            i_restored,
            seasonality,
            curve_db,
            mu,
            last_incidence_date,
            weighted=True,
            match_length=match_length,
            forecast_length=forecast_length
        )
    else:
        raise ValueError(f"Unsupported forecast_type: {forecast_type}")


def epilearn_pipeline(config, data_dict):
    """
    Run EpiLearn forecast on a specified country using a config dict and shared dataset.

    Parameters:
    - config: dict with keys:
        - country: str
        - target_var: str
        - match_length: int
        - forecast_length: int
        - forecast_type: str ('mean', 'median', 'weightedmedian')
        - smoothing_sigma: float
        - lambda_: float
        - mu: float
        - split_ratio: float
    - data_dict: dict mapping country names to pandas DataFrames with a datetime index and columns including target_var

    Returns:
    - pd.DataFrame with columns [country, forecast_origin, true_values, predictions, forecast_dates]
    """
    country = config['country']
    target_var = config['target_var']
    match_length = config.get('match_length', 28)
    forecast_length = config.get('forecast_length', 28)
    forecast_type = config.get('forecast_type', 'median')
    smoothing_sigma = config.get('smoothing_sigma', 2.0)
    lambda_ = config.get('lambda_', 108.0)
    mu = config.get('mu', 0.0675)
    split_ratio = config.get('split_ratio', 0.7)

    full_series = data_dict[country][target_var].values.astype(float)
    split_idx = int(len(full_series) * split_ratio)
    train_series = full_series[:split_idx]
    test_series = full_series[split_idx:]

    other_series = [df[target_var].values.astype(float) for c, df in data_dict.items() if c != country]
    curve_db = build_curve_db(other_series + [train_series], match_length, forecast_length)

    results = []
    for i in range(0, len(test_series) - (match_length + forecast_length) + 1):
        segment = test_series[i:i + match_length + forecast_length]
        input_series = np.concatenate([train_series, test_series[:i + match_length]])
        last_date = data_dict[country].index[split_idx + i + match_length - 1].strftime('%Y-%m-%d')

        ir_forecast, i0_forecast, forecast_dates = epiinvert_forecast(
            input_series,
            curve_db,
            last_date,
            forecast_type,
            smoothing_sigma,
            lambda_,
            mu,
            trend_sentiment=0.0,
            match_length=match_length,
            forecast_length=forecast_length
        )

        true_values = segment[match_length:match_length + forecast_length]
        results.append({
            'country': country,
            'forecast_origin': data_dict[country].index[split_idx + i + match_length - 1],
            'true_values': true_values.tolist(),
            'predictions': i0_forecast.tolist(),
            'forecast_dates': [pd.to_datetime(d) for d in forecast_dates]
        })

    return pd.DataFrame(results)

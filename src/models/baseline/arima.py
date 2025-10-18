import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


def get_scaler(name, config=None):
    if name == "z_score":
        return StandardScaler()
    elif name == "min_max":
        feature_range = config.get("min_max_range", (0, 1)) if config else (0, 1)
        return MinMaxScaler(feature_range=feature_range)
    else:
        return None


def train_model(train_df, config):
    countries = config.get("countries", [])
    target_var = config.get("target_var", "new_cases")
    use_exog = config.get("use_exogenous", False)
    exog_cols = config.get("exogenous_vars", [])
    scaler_config = config.get("scaler_config", {})
    scaler_name = scaler_config.get("type", None)
    exog_scaler_name = scaler_config.get("exog_type", None)
    seasonal = config.get("seasonal", False)
    seasonal_period = config.get("seasonal_period", 7)

    models = {}
    for country in countries:
        df = train_df[train_df["country"] == country].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.dropna(subset=[target_var])

        y = df[target_var].values.reshape(-1, 1)
        target_scaler = get_scaler(scaler_name, scaler_config)
        if target_scaler:
            y_scaled = target_scaler.fit_transform(y).flatten()
        else:
            y_scaled = y.flatten()

        exog = None
        exog_scaler = get_scaler(exog_scaler_name, scaler_config)
        if use_exog and exog_cols:
            exog_raw = df[exog_cols]
            if exog_scaler:
                exog = exog_scaler.fit_transform(exog_raw)
            else:
                exog = exog_raw.values

        auto_model = auto_arima(y_scaled, exogenous=exog, seasonal=seasonal, m=seasonal_period,
                                stepwise=True, suppress_warnings=True, error_action="ignore")
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order if seasonal else (0, 0, 0, 0)

        print(f"[{country}] Selected ARIMA order: {order}, Seasonal order: {seasonal_order}")

        models[country] = {
            "y": y_scaled,
            "exog": exog,
            "target_scaler": target_scaler,
            "exog_scaler": exog_scaler,
            "exog_cols": exog_cols,
            "order": order,
            "seasonal_order": seasonal_order,
            "seasonal": seasonal,
            "country": country,
            "target_var": target_var,
            "test_scaled_target": None
        }

    return models


def predict_model(model_packages, test_df, config):
    target_var = config.get("target_var", "new_cases")
    forecast_window = config.get("forecast_steps", 7)
    results = []

    for country, model_package in model_packages.items():
        df = test_df[test_df["country"] == country].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if model_package["target_scaler"]:
            df["scaled_target"] = model_package["target_scaler"].transform(df[[target_var]])
        else:
            df["scaled_target"] = df[target_var]

        y_hist = model_package["y"]
        exog_hist = model_package["exog"]
        target_scaler = model_package["target_scaler"]
        exog_scaler = model_package["exog_scaler"]
        exog_cols = model_package["exog_cols"]
        order = model_package["order"]
        seasonal_order = model_package["seasonal_order"]
        seasonal = model_package["seasonal"]

        total_steps = len(df)

        for start in range(total_steps - forecast_window + 1):
            end = start + forecast_window

            if start == 0:
                y_input = y_hist
            else:
                y_input = np.concatenate([y_hist, df["scaled_target"].iloc[:start].values])

            exog_input = None
            exog_forecast = None
            if exog_cols:
                all_exog = df[exog_cols].copy()
                if exog_scaler:
                    all_exog = exog_scaler.transform(all_exog)
                exog_input = np.vstack([exog_hist, all_exog[:start]]) if start > 0 else exog_hist
                exog_forecast = all_exog[start:end]

            model = SARIMAX(endog=y_input, exog=exog_input, order=order,
                            seasonal_order=seasonal_order if seasonal else (0, 0, 0, 0),
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

            forecast = model.forecast(steps=forecast_window, exog=exog_forecast)
            if target_scaler:
                forecast = target_scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

            result = {
                "country": country,
                "forecast_origin": df["date"].iloc[start],
                "true_values": df[target_var].iloc[start:end].values.tolist(),
                "predictions": forecast.tolist(),
                "forecast_dates": df["date"].iloc[start:end].tolist()
            }
            results.append(result)

    return pd.DataFrame(results), order, seasonal_order

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from scipy.optimize import curve_fit


def logistic(x, a, b, c):
    return c / (1 + np.exp(-(x - b) / a))

def gompertz(x, a, b, c):
    return c * np.exp(-np.exp(-a * (x - b)))

def exponential(x, a, b):
    return a * np.exp(b * x)

def get_curve_func(name):
    if name == "logistic":
        return logistic
    elif name == "gompertz":
        return gompertz
    elif name == "exponential":
        return exponential
    else:
        raise ValueError("Unsupported curve function")

def get_scaler(name, config=None):
    if name == "z_score":
        return StandardScaler()
    elif name == "min_max":
        feature_range = config.get("min_max_range", (0, 1)) if config else (0, 1)
        return MinMaxScaler(feature_range=feature_range)
    else:
        return None

def get_columnwise_scalers(columns, df, scaler_type, scaler_config=None):
    scalers = {}
    for col in columns:
        scaler = get_scaler(scaler_type, scaler_config)
        scaler.fit(df[[col]])
        scalers[col] = scaler
    return scalers

def apply_columnwise_scaling(df, scalers):
    scaled_df = df.copy()
    for col, scaler in scalers.items():
        scaled_df[col] = scaler.transform(df[[col]])
    return scaled_df

def prepare_data(df, target_var, exogenous_vars, input_window, forecast_steps, split_ratio, scaler_config):
    df_country = df.sort_values("date").reset_index(drop=True)
    features = [target_var] + exogenous_vars if exogenous_vars else [target_var]
    df_country = df_country[["date"] + features].dropna().reset_index(drop=True)

    split_point = int(len(df_country) * split_ratio)
    train_df = df_country.iloc[:split_point].copy()
    test_df = df_country.iloc[split_point:].copy()

    scalers = get_columnwise_scalers(features, train_df, scaler_config["type"], scaler_config)
    train_scaled = apply_columnwise_scaling(train_df, scalers)
    test_scaled = apply_columnwise_scaling(test_df, scalers)

    full_scaled = pd.concat([train_scaled, test_scaled], axis=0).reset_index(drop=True)
    data = full_scaled[features].values

    X, y = [], []
    for i in range(len(data) - input_window - forecast_steps + 1):
        X.append(data[i:i + input_window])
        y.append(data[i + input_window:i + input_window + forecast_steps, 0])  # only target

    X, y = np.array(X), np.array(y)
    split_idx = len(train_scaled) - input_window - forecast_steps + 1
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    return X_train, y_train, X_test, y_test, scalers, df_country



def curvefit_lstm_pipeline(df, config):
    countries = config["countries"]
    target_var = config["target_var"]
    exogenous_vars = config.get("exogenous_vars", [])
    input_window = config.get("input_window", 14)
    forecast_steps = config.get("forecast_steps", 7)
    split_ratio = config.get("split_ratio", 0.7)
    scaler_config = config.get("scaler_config", {"type": "min_max", "min_max_range": (0, 1)})
    curve_model_name = config.get("curvefit_model", "logistic")

    lstm_param_grid = config.get("lstm_param_grid", {
        "units": [50],
        "dropout": [0.2],
        "learning_rate": [0.001],
        "epochs": [50],
        "batch_size": [32]
    })

    def build_model(input_shape, units, dropout, learning_rate):
        model = Sequential()
        model.add(LSTM(units, activation='tanh', input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Dense(forecast_steps))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model

    curve_func = get_curve_func(curve_model_name)
    results = []

    for country in countries:
        df_country = df[df["country"] == country].sort_values("date").reset_index(drop=True)
        df_country = df_country[["date", target_var] + exogenous_vars].dropna().reset_index(drop=True)

        split_point = int(len(df_country) * split_ratio)
        x_train = np.arange(split_point)
        y_train_raw = df_country[target_var].iloc[:split_point].values

        try:
            popt, _ = curve_fit(curve_func, x_train, y_train_raw, maxfev=10000)
            trend = curve_func(np.arange(len(df_country)), *popt)
        except:
            trend = np.zeros(len(df_country))

        df_country[target_var + "_residuals"] = df_country[target_var] - trend

        X_train, y_train, X_test, y_test, scalers, df_used = prepare_data(
            df_country,
            target_var + "_residuals",
            exogenous_vars,
            input_window,
            forecast_steps,
            split_ratio,
            scaler_config
        )

        # Hyperparameter search
        best_val_loss = float("inf")
        best_model = None
        best_params = {}

        for units, dropout, lr, epochs, batch_size in product(
            lstm_param_grid["units"],
            lstm_param_grid["dropout"],
            lstm_param_grid["learning_rate"],
            lstm_param_grid["epochs"],
            lstm_param_grid["batch_size"]
        ):
            model = build_model((X_train.shape[1], X_train.shape[2]), units, dropout, lr)
            es = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=[es]
            )
            val_loss = min(history.history["val_loss"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_params = {
                    "units": units,
                    "dropout": dropout,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "batch_size": batch_size
                }

        model = best_model

        first_forecast_origin_idx = split_point - forecast_steps
        for i in range(len(X_test)):
            idx = first_forecast_origin_idx + i
            if idx >= len(df_country):
                break

            origin_date = df_country["date"].iloc[idx]
            forecast_dates = pd.date_range(start=origin_date + pd.Timedelta(days=1), periods=forecast_steps)

            y_pred_scaled = model.predict(X_test[i].reshape(1, input_window, -1), verbose=0).flatten()
            scaler = scalers[target_var + "_residuals"]
            y_pred_residual = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

            trend_forecast = trend[idx + 1:idx + 1 + forecast_steps]
            if len(trend_forecast) < forecast_steps:
                pad_values = [trend[-1]] * (forecast_steps - len(trend_forecast))
                trend_forecast = np.concatenate([trend_forecast, pad_values])

            y_pred = y_pred_residual + trend_forecast
            y_true = df_country[target_var].iloc[idx + 1:idx + 1 + forecast_steps].values

            results.append({
                "country": country,
                "forecast_origin": origin_date,
                "true_values": y_true.tolist(),
                "predictions": y_pred.tolist(),
                "forecast_dates": forecast_dates.tolist()
            })

    return pd.DataFrame(results), best_params


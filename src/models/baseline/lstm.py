import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


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


def prepare_data(df, country, target_var, exogenous_vars, input_window, forecast_steps, split_ratio, scaler_config):
    df_country = df[df["country"] == country].sort_values("date").reset_index(drop=True)
    features = [target_var] + exogenous_vars if exogenous_vars else [target_var]
    df_country = df_country[["date"] + features].dropna().reset_index(drop=True)

    split_point = int(len(df_country) * split_ratio)
    train_df = df_country.iloc[:split_point].copy()
    test_df = df_country.iloc[split_point:].copy()

    scaler_type = scaler_config.get("type", "min_max")
    scalers = get_columnwise_scalers(features, train_df, scaler_type, scaler_config)
    train_scaled = apply_columnwise_scaling(train_df, scalers)
    test_scaled = apply_columnwise_scaling(test_df, scalers)
    full_scaled = pd.concat([train_scaled, test_scaled], axis=0).reset_index(drop=True)

    data = full_scaled[features].values
    X, y = [], []
    for i in range(len(data) - input_window - forecast_steps + 1):
        X.append(data[i:i + input_window])
        y.append(data[i + input_window:i + input_window + forecast_steps, 0])

    X, y = np.array(X), np.array(y)
    split_idx = len(train_scaled) - input_window - forecast_steps + 1
    X_train_full, y_train_full = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, shuffle=False)

    return X_train, y_train, X_val, y_val, X_test, y_test, scalers, df_country


def lstm_pipeline(df, config):
    countries = config["countries"]
    target_var = config["target_var"]
    exogenous_vars = config.get("exogenous_vars", [])
    input_window = config.get("input_window", 14)
    forecast_steps = config.get("forecast_steps", 7)
    split_ratio = config.get("split_ratio", 0.7)
    scaler_config = config.get("scaler_config", {"type": "min_max", "min_max_range": (0, 1)})

    param_grid = config.get("lstm_param_grid", {
        "batch_size": [32],
        "hidden_dim": [64],
        "dropout": [0.2],
        "learning_rate": [0.001],
        "epochs": [50]
    })

    results = []

    for country in countries:
        X_train, y_train, X_val, y_val, X_test, y_test, scalers, df_country = prepare_data(
            df, country, target_var, exogenous_vars, input_window, forecast_steps, split_ratio, scaler_config
        )

        n_features = X_train.shape[2]

        best_val_loss = float("inf")
        best_model = None
        best_params = {}

        for batch_size in param_grid["batch_size"]:
            for hidden_dim in param_grid["hidden_dim"]:
                for dropout in param_grid["dropout"]:
                    for lr in param_grid["learning_rate"]:
                        for epochs in param_grid["epochs"]:
                            model = Sequential()
                            model.add(LSTM(hidden_dim, activation='tanh', input_shape=(input_window, n_features)))
                            model.add(Dropout(dropout))
                            model.add(Dense(forecast_steps))
                            model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

                            history = model.fit(
                                X_train, y_train,
                                validation_data=(X_val, y_val),
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=0
                            )

                            val_loss = min(history.history["val_loss"])
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_model = model
                                best_params = {
                                    "batch_size": batch_size,
                                    "hidden_dim": hidden_dim,
                                    "dropout": dropout,
                                    "learning_rate": lr,
                                    "epochs": epochs
                                }

        y_pred = best_model.predict(X_test)
        y_pred = scalers[target_var].inverse_transform(y_pred)
        y_test = scalers[target_var].inverse_transform(y_test)

        for i in range(len(y_pred)):
            origin_date = df_country["date"].iloc[input_window + len(y_train) + len(y_val) + i]
            forecast_dates = pd.date_range(start=origin_date, periods=forecast_steps, freq="D")

            results.append({
                "country": country,
                "forecast_origin": origin_date,
                "true_values": y_test[i].tolist(),
                "predictions": y_pred[i].tolist(),
                "forecast_dates": forecast_dates.tolist()
            })

    return pd.DataFrame(results), best_params

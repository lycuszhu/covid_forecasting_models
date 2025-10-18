import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    return X_train, y_train, X_test, y_test, scalers, df_country

def cnn_lstm_pipeline(df, config):
    from itertools import product
    countries = config["countries"]
    target_var = config["target_var"]
    exogenous_vars = config.get("exogenous_vars", [])
    input_window = config.get("input_window", 14)
    forecast_steps = config.get("forecast_steps", 7)
    split_ratio = config.get("split_ratio", 0.7)
    scaler_config = config.get("scaler_config", {"type": "min_max", "min_max_range": (0, 1)})

    cnn_param_grid = config.get("cnn_param_grid", {
        "filters": [64],
        "kernel_size": [2],
        "activation": ['relu'],
        "pool_size": [2]
    })
    lstm_param_grid = config.get("lstm_param_grid", {
        "units": [50],
        "activation": ['relu']
    })
    training_param_grid = config.get("training_param_grid", {
        "epochs": [50],
        "batch_size": [32],
        "validation_split": [0.2],
        "patience": [10]
    })

    results_all = []
    best_model_config = {}

    for country in countries:
        X_train, y_train, X_test, y_test, scalers, df_country = prepare_data(
            df, country, target_var, exogenous_vars, input_window, forecast_steps, split_ratio, scaler_config
        )
        n_features = X_train.shape[2]

        best_val_loss = float('inf')
        best_model = None

        for (filters, kernel_size, cnn_act, pool_size), (units, lstm_act), (epochs, batch_size, val_split, patience) in product(
            product(cnn_param_grid["filters"], cnn_param_grid["kernel_size"], cnn_param_grid["activation"], cnn_param_grid["pool_size"]),
            product(lstm_param_grid["units"], lstm_param_grid["activation"]),
            product(training_param_grid["epochs"], training_param_grid["batch_size"], training_param_grid["validation_split"], training_param_grid["patience"])
        ):
            model = Sequential()
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=cnn_act, input_shape=(input_window, n_features)))
            model.add(MaxPooling1D(pool_size=pool_size))
            model.add(LSTM(units=units, activation=lstm_act))
            model.add(Dense(forecast_steps))
            model.compile(optimizer='adam', loss='mse')

            es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=val_split,
                                verbose=0,
                                callbacks=[es])
            val_loss = min(history.history["val_loss"])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_model_config[country] = {
                    "cnn": {"filters": filters, "kernel_size": kernel_size, "activation": cnn_act, "pool_size": pool_size},
                    "lstm": {"units": units, "activation": lstm_act},
                    "training": {"epochs": epochs, "batch_size": batch_size, "validation_split": val_split, "patience": patience}
                }

        y_pred = best_model.predict(X_test)
        y_pred = scalers[target_var].inverse_transform(y_pred)
        y_test = scalers[target_var].inverse_transform(y_test)

        for i in range(len(y_pred)):
            origin_date = df_country["date"].iloc[input_window + len(y_train) + i]
            forecast_dates = pd.date_range(start=origin_date, periods=forecast_steps, freq="D")

            results_all.append({
                "country": country,
                "forecast_origin": origin_date,
                "true_values": y_test[i].tolist(),
                "predictions": y_pred[i].tolist(),
                "forecast_dates": forecast_dates.tolist()
            })

    return pd.DataFrame(results_all), best_model_config

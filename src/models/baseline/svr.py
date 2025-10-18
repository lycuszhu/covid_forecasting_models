import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.optimize import differential_evolution

def get_scaler(name, scaler_config=None):
    if name == "z_score":
        return StandardScaler()
    elif name == "min_max":
        feature_range = scaler_config.get("min_max_range", (0, 1)) if scaler_config else (0, 1)
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

def prepare_data(df, country, target_var, exogenous_vars, split_ratio, scaler_config):
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

    return full_scaled[features].values, scalers, df_country, split_point

def prepare_svr_data(data, window_size, forecast_steps):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_steps + 1):
        x_seq = data[i:i + window_size].flatten()
        y_seq = data[i + window_size:i + window_size + forecast_steps, 0]
        X.append(x_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)

def optimize_svr_with_metaheuristic(X_train, y_train_step):
    def svr_loss(params):
        C, epsilon, gamma = params
        model = SVR(C=C, epsilon=epsilon, gamma=gamma)
        model.fit(X_train, y_train_step)
        pred = model.predict(X_train)
        return np.mean((pred - y_train_step) ** 2)

    bounds = [(0.1, 100), (0.01, 1.0), (0.0001, 1.0)]
    result = differential_evolution(svr_loss, bounds)
    best_params = result.x
    return SVR(C=best_params[0], epsilon=best_params[1], gamma=best_params[2])

def svr_pipeline(df, config):
    countries = config["countries"]
    target_var = config["target_var"]
    exogenous_vars = config.get("exogenous_vars", [])
    input_window = config.get("input_window", 14)
    forecast_steps = config.get("forecast_steps", 7)
    split_ratio = config.get("split_ratio", 0.7)
    scaler_config = config.get("scaler_config", {"type": "min_max", "min_max_range": (0, 1)})
    param_grid = config.get("param_grid", {"C": [1.0], "gamma": ["scale"]})
    tuning_strategy = config.get("tuning_strategy", "grid")

    results = []

    for country in countries:
        data, scalers, df_country, split_point = prepare_data(
            df, country, target_var, exogenous_vars,
            split_ratio, scaler_config
        )

        X, y = prepare_svr_data(data, input_window, forecast_steps)

        split_idx = split_point - input_window - forecast_steps + 1
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        best_models = []
        for step in range(forecast_steps):
            if tuning_strategy == "grid":
                svr = SVR(kernel=config.get("kernel", "rbf"))
                grid = GridSearchCV(svr, param_grid, cv=3)
                grid.fit(X_train, y_train[:, step])
                best_models.append(grid.best_estimator_)
                best_params_summary = {
                    f"step_{i}": {
                        "C": model.C,
                        "gamma": model.gamma,
                        "epsilon": model.epsilon
                        } for i, model in enumerate(best_models)
                }
            elif tuning_strategy == "meta":
                best_model = optimize_svr_with_metaheuristic(X_train, y_train[:, step])
                best_models.append(best_model)
            else:
                raise ValueError("Unsupported tuning strategy: choose 'grid' or 'meta'")

        preds = np.column_stack([m.predict(X_test) for m in best_models])
        target_scaler = scalers[target_var]
        preds = target_scaler.inverse_transform(preds)
        y_test = target_scaler.inverse_transform(y_test)

        for i in range(len(preds)):
            origin_date = df_country["date"].iloc[input_window + split_idx + i]
            forecast_dates = pd.date_range(start=origin_date, periods=forecast_steps, freq="D")

            results.append({
                "country": country,
                "forecast_origin": origin_date,
                "true_values": y_test[i].tolist(),
                "predictions": preds[i].tolist(),
                "forecast_dates": forecast_dates.tolist()
            })

    return pd.DataFrame(results), best_params_summary

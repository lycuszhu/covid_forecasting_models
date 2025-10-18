import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import timedelta


def generate_oxcgrt_holidays(df, policy_column="StringencyIndex_Average", threshold=70):
    df = df.copy()
    df = df[df[policy_column] >= threshold]
    df = df.dropna(subset=["date", "country"])
    df["ds"] = pd.to_datetime(df["date"])
    df["holiday"] = "oxcgrt_policy"
    df["lower_window"] = 0
    df["upper_window"] = 0

    holidays_by_country = {
        country: group[["holiday", "ds", "lower_window", "upper_window"]].reset_index(drop=True)
        for country, group in df.groupby("country")
    }

    return holidays_by_country


def generate_fixed_holidays(country_code):
    try:
        import holidays
        cal = holidays.country_holidays(country_code)
        return pd.DataFrame({
            "holiday": ["calendar_holiday"] * len(cal),
            "ds": pd.to_datetime(list(cal.keys())),
            "lower_window": [0] * len(cal),
            "upper_window": [0] * len(cal)
        })
    except Exception as e:
        print(f"Could not generate calendar holidays for {country_code}: {e}")
        return pd.DataFrame(columns=["holiday", "ds", "lower_window", "upper_window"])


def prepare_data(df, country, target_var, exogenous_vars):
    df_country = df[df["country"] == country].sort_values("date").reset_index(drop=True)
    df_country = df_country[["date", target_var] + exogenous_vars].dropna().copy()
    return df_country


def prophet_pipeline(df, config):
    countries = config["countries"]
    target_var = config["target_var"]
    exogenous_vars = config.get("exogenous_vars", [])
    forecast_steps = config.get("forecast_steps", 14)
    split_ratio = config.get("split_ratio", 0.7)
    seasonalities = config.get("seasonalities", [])
    holiday_dict = config.get("holidays", {})

    results = []

    for country in countries:
        df_country = prepare_data(df, country, target_var, exogenous_vars)
        split_point = int(len(df_country) * split_ratio)
        train_df = df_country.iloc[:split_point].copy()
        test_df = df_country.iloc[split_point:].copy()

        df_country = df_country.rename(columns={"date": "ds", target_var: "y"})
        total_steps = len(test_df)
        min_history = len(train_df)

        for step in range(0, total_steps - forecast_steps + 1):
            start = min_history + step
            end = start + forecast_steps

            history = df_country.iloc[:start].copy()
            future = df_country.iloc[start:end].copy()

            if len(history) == 0 or len(future) < forecast_steps:
                continue

            country_holidays = holiday_dict.get(country, None)
            model = Prophet(holidays=country_holidays) if country_holidays is not None else Prophet()

            for season in seasonalities:
                model.add_seasonality(name=season["name"], period=season["period"], fourier_order=season["fourier_order"])

            for var in exogenous_vars:
                model.add_regressor(var)

            model.fit(history)

            future_df = future[["ds"] + exogenous_vars].copy() if exogenous_vars else future[["ds"]].copy()
            forecast = model.predict(future_df)
            pred = forecast["yhat"].values
            true = future["y"].values

            results.append({
                "country": country,
                "forecast_origin": future["ds"].iloc[0],
                "true_values": true.tolist(),
                "predictions": pred.tolist(),
                "forecast_dates": future["ds"].tolist()
            })

    return pd.DataFrame(results)

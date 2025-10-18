import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tqdm import tqdm

# Global error computation
def smape(y_true: List[float], y_pred: List[float]) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return np.mean(np.divide(diff, denominator, out=np.zeros_like(diff), where=denominator != 0)) * 100

def extract_forecast_length(config: dict):
    for key in ["forecast_steps", "forecast_length", "n_days_forecast"]:
        if key in config:
            return config[key]
    return None

def compute_global_errors(all_true: List[List[float]], all_pred: List[List[float]]) -> Dict[str, float]:
    # Flatten all data to compute global metrics
    flat_true = [val for sublist in all_true for val in sublist]
    flat_pred = [val for sublist in all_pred for val in sublist]

    # Global average metrics
    results = {
        "MAE_global": mean_absolute_error(flat_true, flat_pred),
        "RMSE_global": np.sqrt(mean_squared_error(flat_true, flat_pred)),
        "sMAPE_global": smape(flat_true, flat_pred)
    }

    # Per-series metrics
    mae_list = []
    rmse_list = []
    smape_list = []

    for y_t, y_p in zip(all_true, all_pred):
        mae_list.append(mean_absolute_error(y_t, y_p))
        rmse_list.append(np.sqrt(mean_squared_error(y_t, y_p)))
        smape_list.append(smape(y_t, y_p))

    def describe(metric_list, prefix):
        return {
            f"{prefix}_mean": np.mean(metric_list),
            f"{prefix}_median": np.median(metric_list),
            f"{prefix}_p25": np.percentile(metric_list, 25),
            f"{prefix}_p75": np.percentile(metric_list, 75),
            f"{prefix}_std": np.std(metric_list),
            f"{prefix}_min": np.min(metric_list),
            f"{prefix}_max": np.max(metric_list)
        }

    results.update(describe(mae_list, "MAE"))
    results.update(describe(rmse_list, "RMSE"))
    results.update(describe(smape_list, "sMAPE"))

    return results


# Batch processor for a specific model (e.g., SVR)
def compute_global_metrics_for_model(model_name: str, countries: List[str], root_dir: str = None, output_filename: str = None):
    records = []
    script_dir = Path(__file__).resolve().parent
    results_root = script_dir.parent / "results"
    output_path = results_root / (output_filename or f"global_metrics_results_{model_name}.csv")

    for country in countries:
        country_dir = results_root / model_name / country
        if not country_dir.exists():
            print(f"⚠️ Country directory not found: {country_dir}")
            continue

        for dirpath, dirnames, filenames in os.walk(country_dir):
            if "forecast_results.csv" in filenames and "config.json" in filenames:
                try:
                    forecast_path = Path(dirpath) / "forecast_results.csv"
                    config_path = Path(dirpath) / "config.json"

                    df = pd.read_csv(forecast_path, parse_dates=["forecast_origin"])
                    df = df[(df["forecast_origin"] >= "2022-08-20") & (df["forecast_origin"] <= "2023-02-09")].copy()

                    with open(config_path, "r") as f:
                        config = json.load(f)

                    F = extract_forecast_length(config)
                    if F is None:
                        continue

                    all_true, all_pred = [], []
                    for _, row in df.iterrows():
                        y_true = json.loads(row["true_values"])
                        y_pred = json.loads(row["predictions"])
                        all_true.append(y_true)
                        all_pred.append(y_pred)

                    metrics = compute_global_errors(all_true, all_pred)
                    record = {
                        "model": model_name,
                        "country": country,
                        "W": config.get("input_window", "NA"),
                        "F": F,
                        "target_var": config.get("target_var", "new_cases"),
                        **metrics,
                        "folder": str(dirpath)
                    }
                    records.append(record)
                except Exception as e:
                    print(f"❌ Failed to process {dirpath}: {e}")

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_path, index=False)
    print(f"✅ Global metrics saved to: {output_path}")
    return results_df


# generate global metrics for all models
if __name__ == "__main__":
    model_list = [
        "arima",
        "cnn_lstm",
        "curvefit_lstm",
        "delphi",
        "epilearn_fullcurve",
        "lstm",
        "prophet",
        "sarima",
        "svr"
    ]

    countries = ["Australia", "Brazil", "China", "Germany", "India", "United Kingdom", "US"]

    for model_name in tqdm(model_list, desc="Evaluating models"):
        print(f"\n📊 Processing global metrics for model: {model_name}")
        compute_global_metrics_for_model(
            model_name=model_name,
            countries=countries,
            output_filename=f"global_metrics_results_{model_name}.csv"
        )



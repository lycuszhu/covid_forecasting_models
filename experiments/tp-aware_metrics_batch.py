import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import multiprocessing

# === Set up project root ===
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# === Import metric functions ===
from src.metrics.combined_tp_extraction import extract_combined_turning_points
from src.metrics.tp_window_metrics import compute_tp_alignment_errors
from src.metrics.tp_slope_direction import compute_tp_and_global_slope_agreement
from src.metrics.tp_lead_lag_error import compute_tp_lead_lag

# === Load merged data ===
merged_data = pd.read_csv(project_root / "data" / "processed" / "merged_covid_data.csv", parse_dates=["date"])
merged_data.set_index("date", inplace=True)

def extract_forecast_length(config: dict):
    for key in ["forecast_steps", "forecast_length", "n_days_forecast"]:
        if key in config:
            return config[key]
    return None

def compute_tp_metrics_for_model(model_name, countries):
    results = []
    results_root = project_root / "results"
    output_path = results_root / f"tp_aware_metrics_results_{model_name}.csv"

    for country in tqdm(countries, desc=f"{model_name}"):
        data = merged_data[merged_data["country"] == country].copy()
        if data.empty:
            continue

        series = data["new_cases"].fillna(0)
        smoothed = series.rolling(7, center=True, min_periods=1).mean().rolling(7, center=True, min_periods=1).mean()
        tp_df = extract_combined_turning_points(series)
        tp_df["date"] = series.index[tp_df["index"]].values
        tp_df.set_index("date", inplace=True)

        model_root = results_root / model_name / country
        if not model_root.exists():
            continue

        for dirpath, _, filenames in os.walk(model_root):
            if "forecast_results.csv" not in filenames or "config.json" not in filenames:
                continue

            try:
                forecast_df = pd.read_csv(Path(dirpath) / "forecast_results.csv", parse_dates=["forecast_origin"])
                forecast_df = forecast_df[(forecast_df["forecast_origin"] >= "2022-08-20") & (forecast_df["forecast_origin"] <= "2023-02-09")].copy()

                with open(Path(dirpath) / "config.json", "r") as f:
                    config = json.load(f)

                F = extract_forecast_length(config)
                if F is None:
                    continue

                alignment_df = compute_tp_alignment_errors(forecast_df, tp_df)
                slope_df = compute_tp_and_global_slope_agreement(forecast_df, tp_df, smoothed)
                lead_lag_df = compute_tp_lead_lag(forecast_df, tp_df, series)

                row = {
                    "model": model_name,
                    "country": country,
                    "W": config.get("input_window", "NA"),
                    "F": F,
                    "target_var": config.get("target_var", "new_cases"),
                }

                # Directly slice alignment_df values
                row.update({
                    "alignment_MAE": alignment_df["MAE"].iloc[0],
                    "alignment_RMSE": alignment_df["RMSE"].iloc[0],
                    "alignment_sMAPE": alignment_df["sMAPE"].iloc[0],
                })
                for metric in ["MAE", "RMSE", "sMAPE"]:
                    for stat in ["mean", "median", "p25", "p75", "std", "min", "max"]:
                        col = f"{metric}_{stat}"
                        row[f"alignment_{col}"] = alignment_df[col].iloc[0] if col in alignment_df.columns else np.nan

                # Slope metrics (already aggregated)
                row.update({
                    "tp_same_sign": slope_df["tp_same_sign"].iloc[0],
                    "tp_diff_sign": slope_df["tp_diff_sign"].iloc[0],
                    "tp_diff_percentage": slope_df["tp_diff_percentage"].iloc[0],
                    "tp_slope_diff_stats": slope_df["tp_slope_diff_stats"].iloc[0],
                    "global_same_sign": slope_df["global_same_sign"].iloc[0],
                    "global_diff_sign": slope_df["global_diff_sign"].iloc[0],
                    "global_diff_percentage": slope_df["global_diff_percentage"].iloc[0],
                    "global_slope_diff_stats": slope_df["global_slope_diff_stats"].iloc[0],
                })

                # Lead-lag metrics from df
                matched = lead_lag_df["lead_lag_days"].dropna()
                row["lead_lag_unmatched_rate"] = 1 - len(matched) / len(lead_lag_df) if len(lead_lag_df) > 0 else np.nan
                row["lead_lag_within_halfF_ratio"] = (np.sum(np.abs(matched) <= F / 2) / len(lead_lag_df)) if len(lead_lag_df) > 0 else np.nan
                row["lead_lag_mean"] = np.abs(matched).mean() if not matched.empty else np.nan
                row["lead_lag_std"] = np.abs(matched).std() if not matched.empty else np.nan
                row["lead_lag_median"] = np.abs(matched).median() if not matched.empty else np.nan   

                # folder
                row["folder"] = str(dirpath)             


                results.append(row)

            except Exception as e:
                print(f"⚠️ Error in {dirpath}: {e}")

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ TP-aware metrics saved to {output_path}")

def run_model_safe(model_name):
    try:
        countries = [
            "Australia", "Brazil", "China", "Germany",
            "India", "United Kingdom", "US"
        ]
        compute_tp_metrics_for_model(model_name, countries)
        print(f"✅ Finished: {model_name}")
    except Exception as e:
        print(f"❌ Error with model {model_name}: {e}")

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
        "svr",
    ]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(run_model_safe, model_list)

import os
import json
import pandas as pd
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.baseline.svr import svr_pipeline

# Config
COUNTRIES = ['China', 'United Kingdom', 'US', 'Australia', 'India', 'Germany', 'Brazil']
WINDOW_FORECAST_PAIRS = [(14, 7), (21, 7), (28, 14), (56, 28)]
DATA_PATH = './data/processed/merged_covid_data.csv'
RESULTS_DIR = './results/svr'

# Exogenous variable combinations
EXOG_SETTINGS = {
    "no_exog": [],
    "death_recovered": ["new_deaths", "new_recovered"],
    "death_mobility": ["new_deaths", "new_recovered"] + [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline"
    ],
    "death_stringency": ["new_deaths", "new_recovered"] + [
        "C6M_Stay at home requirements",
        "StringencyIndex_Average",
        "GovernmentResponseIndex_Average",
        "ContainmentHealthIndex_Average",
        "EconomicSupportIndex"
    ],
    "death_all": ["new_deaths", "new_recovered"] + [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
        "C6M_Stay at home requirements",
        "StringencyIndex_Average",
        "GovernmentResponseIndex_Average",
        "ContainmentHealthIndex_Average",
        "EconomicSupportIndex"
    ]
}

# Load dataset
df = pd.read_csv(DATA_PATH, parse_dates=['date'])

# Total combinations
total_runs = len(COUNTRIES) * len(WINDOW_FORECAST_PAIRS) * len(EXOG_SETTINGS)

# Progress bar
with tqdm(total=total_runs, desc="SVR Experiments") as pbar:
    for country in COUNTRIES:
        for input_window, forecast_steps in WINDOW_FORECAST_PAIRS:
            for exog_key, exog_vars in EXOG_SETTINGS.items():
                run_label = f"{country} | W={input_window}, F={forecast_steps}, EXOG={exog_key}"
                pbar.set_description(f"Running: {run_label}")

                config = {
                    "countries": [country],
                    "target_var": "new_cases",
                    "exogenous_vars": exog_vars,
                    "input_window": input_window,
                    "forecast_steps": forecast_steps,
                    "split_ratio": 0.8,
                    "scaler_config": {
                        "type": "min_max",
                        "min_max_range": (0, 100)
                    },
                    "kernel": "rbf",
                    "tuning_strategy": "grid",
                    "param_grid": {
                        "C": [0.1, 1, 10],
                        "gamma": ["scale", "auto"]
                    }
                }

                try:
                    results_df, best_params_summary = svr_pipeline(df, config)
                    config["best_model_params"] = best_params_summary
                except Exception as e:
                    print(f"⚠️ Failed: {run_label} — {e}")
                    pbar.update(1)
                    continue

                output_folder = os.path.join(
                    RESULTS_DIR, country, f"{input_window}_{forecast_steps}", exog_key
                )
                os.makedirs(output_folder, exist_ok=True)

                results_df.to_csv(os.path.join(output_folder, 'forecast_results.csv'), index=False)
                with open(os.path.join(output_folder, 'config.json'), 'w') as f:
                    json.dump(config, f, indent=2)

                pbar.update(1)

print("✅ All SVR batch experiments completed.")

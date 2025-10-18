import os
import json
import pandas as pd
from tqdm import tqdm
import sys
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from models.baseline.arima import train_model, predict_model

# Config
COUNTRIES = ['China', 'United Kingdom', 'US', 'Australia', 'India', 'Germany', 'Brazil']
FORECAST_STEPS = [7, 14, 28]
DATA_PATH = './data/processed/merged_covid_data.csv'
RESULTS_DIR = './results/sarima'

SEASONAL = True
SEASONAL_PERIOD = 7

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

# Load dataset once globally
df_global = pd.read_csv(DATA_PATH, parse_dates=['date'])

def run_single_config(args):
    country, forecast_steps, exog_key, exog_vars = args
    run_label = f"{country} | F={forecast_steps}, EXOG={exog_key}"

    config = {
        "countries": [country],
        "target_var": "new_cases",
        "use_exogenous": bool(exog_vars),
        "exogenous_vars": exog_vars,
        "forecast_steps": forecast_steps,
        "split_ratio": 0.8,
        "scaler_config": {
            "type": "min_max",
            "min_max_range": (0, 100),
            "exog_type": "min_max"
        },
        "seasonal": SEASONAL,
        "seasonal_period": SEASONAL_PERIOD
    }

    try:
        model_pkg = train_model(df_global, config)
        results_df, order, seasonal_order = predict_model(model_pkg, df_global, config)
        config["order"] = order
        config["seasonal_order"] = seasonal_order
    except Exception as e:
        return (run_label, None, None, f"⚠️ Failed: {run_label} — {e}")

    output_folder = os.path.join(RESULTS_DIR, country, str(forecast_steps), exog_key)
    os.makedirs(output_folder, exist_ok=True)

    return (run_label, results_df, config, output_folder)


if __name__ == "__main__":
    all_args = [
        (country, forecast_steps, exog_key, exog_vars)
        for country in COUNTRIES
        for forecast_steps in FORECAST_STEPS
        for exog_key, exog_vars in EXOG_SETTINGS.items()
    ]

    with mp.Pool(processes=min(mp.cpu_count(), len(COUNTRIES))) as pool:
        with tqdm(total=len(all_args), desc="ARIMA Experiments (Parallel)") as pbar:
            for run_label, results_df, config, output_folder in pool.imap_unordered(run_single_config, all_args):
                if results_df is None:
                    print(config)  # This will be the error message string if failure occurred
                else:
                    results_df.to_csv(os.path.join(output_folder, 'forecast_results.csv'), index=False)
                    with open(os.path.join(output_folder, 'config.json'), 'w') as f:
                        json.dump(config, f, indent=2)
                pbar.update(1)

    print("✅ All ARIMA batch experiments completed.")

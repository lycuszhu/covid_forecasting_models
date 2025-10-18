import os
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Manager
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.newly_proposed.cnn_lstm import cnn_lstm_pipeline

# Config
COUNTRIES = ['China', 'United Kingdom', 'US', 'Australia', 'India', 'Germany', 'Brazil']
WINDOW_FORECAST_PAIRS = [(14, 7), (21, 7), (28, 14), (56, 28)]
DATA_PATH = './data/processed/merged_covid_data.csv'
RESULTS_DIR = './results/cnn_lstm'

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

# Hyperparameter grids (used internally by cnn_lstm_pipeline)
CNN_PARAM_GRID = {
    "filters": [32, 64],
    "kernel_size": [2],
    "activation": ['relu'],
    "pool_size": [2]
}

LSTM_PARAM_GRID = {
    "units": [32, 64],
    "activation": ['relu']
}

TRAINING_PARAM_GRID = {
    "batch_size": [32],
    "epochs": [50],
    "validation_split": [0.2],
    "patience": [10]
}

# Load data once
df = pd.read_csv(DATA_PATH, parse_dates=['date'])

# Runner function
def run_setting(args):
    country, (input_window, forecast_steps), exog_key, exog_vars, queue = args
    run_label = f"{country} | W={input_window}, F={forecast_steps}, EXOG={exog_key}"

    config = {
        "countries": [country],
        "target_var": "new_cases",
        "exogenous_vars": exog_vars,
        "input_window": input_window,
        "forecast_steps": forecast_steps,
        "split_ratio": 0.8,
        "scaler_config": {"type": "min_max", "min_max_range": (0, 1)},
        "cnn_param_grid": CNN_PARAM_GRID,
        "lstm_param_grid": LSTM_PARAM_GRID,
        "training_param_grid": TRAINING_PARAM_GRID
    }

    try:
        results_df, best_params = cnn_lstm_pipeline(df, config)
        output_folder = os.path.join(RESULTS_DIR, country, f"{input_window}_{forecast_steps}", exog_key)
        os.makedirs(output_folder, exist_ok=True)

        results_df.to_csv(os.path.join(output_folder, 'forecast_results.csv'), index=False)
        config["selected_hyperparams"] = best_params
        with open(os.path.join(output_folder, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    except Exception as e:
        print(f"⚠️ Failed: {run_label} — {e}")

    queue.put(1)

# Multiprocessing wrapper
if __name__ == "__main__":
    manager = Manager()
    queue = manager.Queue()

    args_list = [
        (country, (input_window, forecast_steps), exog_key, exog_vars, queue)
        for country in COUNTRIES
        for (input_window, forecast_steps) in WINDOW_FORECAST_PAIRS
        for exog_key, exog_vars in EXOG_SETTINGS.items()
    ]

    total_runs = len(args_list)

    with Pool(processes=os.cpu_count()) as pool, tqdm(total=total_runs, desc="CNN-LSTM Experiments") as pbar:
        for _ in pool.imap_unordered(run_setting, args_list):
            while not queue.empty():
                queue.get()
                pbar.update(1)

    print("✅ All CNN-LSTM batch experiments completed.")

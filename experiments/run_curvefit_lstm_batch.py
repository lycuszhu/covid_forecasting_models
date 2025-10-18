import os
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Manager
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.newly_proposed.curvefit_lstm import curvefit_lstm_pipeline

# Config
COUNTRIES = ['China', 'United Kingdom', 'US', 'Australia', 'India', 'Germany', 'Brazil']
WINDOW_FORECAST_PAIRS = [(14, 7), (21, 7), (28, 14), (56, 28)]
CURVE_MODELS = ['logistic', 'gompertz']
DATA_PATH = './data/processed/merged_covid_data.csv'
RESULTS_DIR = './results/curvefit_lstm'

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

# LSTM hyperparameter search grid
LSTM_PARAM_GRID = {
    "units": [32, 64, 128],
    "dropout": [0.2],
    "learning_rate": [0.001, 0.0005],
    "epochs": [50],
    "batch_size": [32]
}

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=['date'])

# Runner function
def run_setting(args):
    country, (input_window, forecast_steps), curve_model, exog_key, exog_vars, queue = args

    config = {
        "countries": [country],
        "target_var": "new_cases",
        "exogenous_vars": exog_vars,
        "input_window": input_window,
        "forecast_steps": forecast_steps,
        "split_ratio": 0.7,
        "curvefit_model": curve_model,
        "scaler_config": {
            "type": "min_max",
            "min_max_range": (0, 1)
        },
        "lstm_param_grid": LSTM_PARAM_GRID
    }

    run_label = f"{country} | W={input_window}, F={forecast_steps}, Curve={curve_model}, EXOG={exog_key}"

    try:
        results_df, best_params = curvefit_lstm_pipeline(df, config)
        output_folder = os.path.join(RESULTS_DIR, country, f"{input_window}_{forecast_steps}", exog_key, curve_model)
        os.makedirs(output_folder, exist_ok=True)

        results_df.to_csv(os.path.join(output_folder, 'forecast_results.csv'), index=False)
        config["selected_lstm_params"] = best_params
        config["curvefit_model"] = curve_model
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
        (country, (input_window, forecast_steps), curve_model, exog_key, exog_vars, queue)
        for country in COUNTRIES
        for (input_window, forecast_steps) in WINDOW_FORECAST_PAIRS
        for curve_model in CURVE_MODELS
        for exog_key, exog_vars in EXOG_SETTINGS.items()
    ]

    total_runs = len(args_list)

    with Pool(processes=os.cpu_count()) as pool, tqdm(total=total_runs, desc="CurveFit-LSTM Experiments") as pbar:
        for _ in pool.imap_unordered(run_setting, args_list):
            while not queue.empty():
                queue.get()
                pbar.update(1)

    print("✅ All CurveFit-LSTM batch experiments completed.")

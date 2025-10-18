import os
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Manager
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.newly_proposed.delphi import fit_and_forecast

# Config
COUNTRIES = ['China', 'United Kingdom', 'US', 'Australia', 'India', 'Germany', 'Brazil']
DATA_PATH = './data/processed/merged_covid_data.csv'
POP_PATH = './data/processed/population_global.csv'
RESULTS_DIR = './results/delphi'

# Forecasting settings
N_DAYS_FIT = 30
N_DAYS_FORECAST = 28
SPLIT_RATIO = 0.8

# p0 and bounds to be applied for all countries
P0 = [0.2, 0.5, 0.1, 0.01, 0.5, 0.1, 15]
BOUNDS = [
    (0.01, 1.0),
    (0.01, 2.0),
    (0.01, 1.0),
    (0.001, 0.1),
    (0.0, 1.0),
    (0.01, 1.0),
    (0, 100)
]

# Runner function
def run_setting(args):
    country, queue = args

    config = {
        "file_path": DATA_PATH,
        "population_path": POP_PATH,
        "countries": [country],
        "n_days_fit": N_DAYS_FIT,
        "n_days_forecast": N_DAYS_FORECAST,
        "split_ratio": SPLIT_RATIO,
        "p0": P0,
        "bounds": BOUNDS
    }

    try:
        results_df = fit_and_forecast(config)
        output_folder = os.path.join(RESULTS_DIR, country)
        os.makedirs(output_folder, exist_ok=True)

        results_df.to_csv(os.path.join(output_folder, 'forecast_results.csv'), index=False)
        with open(os.path.join(output_folder, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    except Exception as e:
        print(f"⚠️ Failed: {country} — {e}")
    
    queue.put(1)

# Multiprocessing wrapper
if __name__ == "__main__":
    manager = Manager()
    queue = manager.Queue()
    total_runs = len(COUNTRIES)

    args_list = [(country, queue) for country in COUNTRIES]

    with Pool(processes=os.cpu_count()) as pool, tqdm(total=total_runs, desc="Delphi Experiments") as pbar:
        for _ in pool.imap_unordered(run_setting, args_list):
            while not queue.empty():
                queue.get()
                pbar.update(1)

    print("✅ All Delphi batch experiments completed.")

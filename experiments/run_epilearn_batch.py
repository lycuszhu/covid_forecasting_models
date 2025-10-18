import os
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Manager
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.newly_proposed.epilearn import epilearn_pipeline

# Config
DATA_PATH = './data/processed/merged_covid_data.csv'
RESULTS_DIR = './results/epilearn_fullcurve'
TARGET_VAR = 'new_cases'
MATCH_LENGTH = 28
FORECAST_LENGTH = 28
SPLIT_RATIO = 0.8
FORECAST_TYPES = ['mean', 'median', 'weightedmedian']
SMOOTHING_SIGMA = 2.0
LAMBDA = 108.0
MU = 0.0675

# Load full data and build global data_dict
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
ALL_COUNTRIES = df["country"].unique().tolist()

data_dict_full = {
    country: df[df["country"] == country].copy().set_index("date").sort_index()
    for country in ALL_COUNTRIES
}

# Target countries (forecasted)
TARGET_COUNTRIES = ['China', 'United Kingdom', 'US', 'Australia', 'India', 'Germany', 'Brazil']

# Runner function
def run_setting(args):
    country, forecast_type, queue = args

    config = {
        "country": country,
        "target_var": TARGET_VAR,
        "match_length": MATCH_LENGTH,
        "forecast_length": FORECAST_LENGTH,
        "forecast_type": forecast_type,
        "smoothing_sigma": SMOOTHING_SIGMA,
        "lambda_": LAMBDA,
        "mu": MU,
        "split_ratio": SPLIT_RATIO
    }

    try:
        results_df = epilearn_pipeline(config, data_dict_full)
        output_folder = os.path.join(RESULTS_DIR, country, forecast_type)
        os.makedirs(output_folder, exist_ok=True)

        results_df.to_csv(os.path.join(output_folder, 'forecast_results.csv'), index=False)
        with open(os.path.join(output_folder, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    except Exception as e:
        print(f"⚠️ Failed: {country} ({forecast_type}) — {e}")

    queue.put(1)

# Multiprocessing wrapper
if __name__ == "__main__":
    manager = Manager()
    queue = manager.Queue()
    total_runs = len(TARGET_COUNTRIES) * len(FORECAST_TYPES)

    args_list = [(country, forecast_type, queue) for country in TARGET_COUNTRIES for forecast_type in FORECAST_TYPES]

    with Pool(processes=os.cpu_count()) as pool, tqdm(total=total_runs, desc="EpiLearn (Full Curve DB)") as pbar:
        for _ in pool.imap_unordered(run_setting, args_list):
            while not queue.empty():
                queue.get()
                pbar.update(1)

    print("✅ All EpiLearn (Full Curve DB) batch experiments completed.")

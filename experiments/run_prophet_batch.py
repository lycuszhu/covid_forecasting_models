import os
import json
import copy
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.baseline.prophet import prophet_pipeline, generate_oxcgrt_holidays, generate_fixed_holidays

# Config
COUNTRIES = ['China', 'United Kingdom', 'US', 'Australia', 'India', 'Germany', 'Brazil']
FORECAST_STEPS = [7, 14, 28]
DATA_PATH = './data/processed/merged_covid_data.csv'
RESULTS_DIR = './results/prophet'

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

# Generate holiday dictionaries once
oxcgrt_holidays = generate_oxcgrt_holidays(df)
calendar_holidays = {country: generate_fixed_holidays(country) for country in COUNTRIES}
combined_holidays = {
    country: pd.concat([oxcgrt_holidays.get(country, pd.DataFrame()), calendar_holidays[country]])
    for country in COUNTRIES
}

# Optional: Define seasonalities (can be left empty or customized)
SEASONALITIES = [
    {"name": "weekly", "period": 7, "fourier_order": 3},
    {"name": "monthly", "period": 30.5, "fourier_order": 5}
]

# Total combinations
total_runs = len(COUNTRIES) * len(FORECAST_STEPS) * len(EXOG_SETTINGS)

# Progress bar
with tqdm(total=total_runs, desc="Prophet Experiments") as pbar:
    for country in COUNTRIES:
        for forecast_steps in FORECAST_STEPS:
            for exog_key, exog_vars in EXOG_SETTINGS.items():
                run_label = f"{country} | F={forecast_steps}, EXOG={exog_key}"
                pbar.set_description(f"Running: {run_label}")

                config = {
                    "countries": [country],
                    "target_var": "new_cases",
                    "exogenous_vars": exog_vars,
                    "forecast_steps": forecast_steps,
                    "split_ratio": 0.8,
                    #"holidays": combined_holidays,
                    "seasonalities": SEASONALITIES
                }

                try:
                    results_df = prophet_pipeline(df, config)
                except Exception as e:
                    print(f"⚠️ Failed: {run_label} — {e}")
                    pbar.update(1)
                    continue

                output_folder = os.path.join(
                    RESULTS_DIR, country, f"{forecast_steps}", exog_key
                )
                os.makedirs(output_folder, exist_ok=True)

                results_df.to_csv(os.path.join(output_folder, 'forecast_results.csv'), index=False)
                with open(os.path.join(output_folder, 'config.json'), 'w') as f:
                    json.dump(config, f, indent=2)

                pbar.update(1)

print("✅ All Prophet batch experiments completed.")

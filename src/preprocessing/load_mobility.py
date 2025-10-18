import pandas as pd
from pathlib import Path

def load_mobility_data(file_path):
    df = pd.read_csv(Path(file_path), parse_dates=["date"])
    df = df[df["sub_region_1"].isna() & df["sub_region_2"].isna()]
    cols_to_keep = ["country_region", "date"] + [col for col in df.columns if "percent_change" in col]
    df = df[cols_to_keep].rename(columns={"country_region": "country"})
    return df

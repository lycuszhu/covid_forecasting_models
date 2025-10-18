import pandas as pd
from pathlib import Path

def load_oxcgrt_data(file_path):
    df = pd.read_csv(Path(file_path), parse_dates=["Date"])
    
    # Format date and rename for consistency
    df["date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df = df.rename(columns={"CountryName": "country"})

    # Keep only national-level rows (drop subnational)
    df = df[df["RegionName"].isna() & (df["Jurisdiction"] == "NAT_TOTAL")]

    # Select relevant columns
    cols_to_keep = [
        "country", "date",
        "C6M_Stay at home requirements",
        "StringencyIndex_Average",
        "GovernmentResponseIndex_Average",
        "ContainmentHealthIndex_Average",
        "EconomicSupportIndex"
    ]
    df = df[cols_to_keep]

    # Optional: fill missing values forward
    df = df.sort_values(["country", "date"])
    df = df.fillna(method="ffill")

    return df

import pandas as pd
from pathlib import Path

def load_jhu_data(data_dir):
    data_dir = Path(data_dir)
    
    def load_and_melt(file_name, value_name):
        df = pd.read_csv(data_dir / file_name)
        df = df.drop(columns=["Lat", "Long", "Province/State"], errors="ignore")
        df = df.groupby("Country/Region").sum().reset_index()
        df = df.melt(id_vars=["Country/Region"], var_name="date", value_name=value_name)
        df["date"] = pd.to_datetime(df["date"])
        return df

    confirmed = load_and_melt("time_series_covid19_confirmed_global.csv", "confirmed")
    deaths = load_and_melt("time_series_covid19_deaths_global.csv", "deaths")

    try:
        recovered = load_and_melt("time_series_covid19_recovered_global.csv", "recovered")
    except FileNotFoundError:
        recovered = pd.DataFrame(columns=["Country/Region", "date", "recovered"])

    # Merge cumulative values
    df = confirmed.merge(deaths, on=["Country/Region", "date"], how="outer")
    if not recovered.empty:
        df = df.merge(recovered, on=["Country/Region", "date"], how="outer")

    # Rename and sort
    df = df.rename(columns={"Country/Region": "country"})
    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    # Compute daily new values
    df["new_cases"] = df.groupby("country")["confirmed"].diff().fillna(0).clip(lower=0)
    df["new_deaths"] = df.groupby("country")["deaths"].diff().fillna(0).clip(lower=0)
    if "recovered" in df.columns and not df["recovered"].isna().all():
        df["new_recovered"] = df.groupby("country")["recovered"].diff().fillna(0).clip(lower=0)
    else:
        df["new_recovered"] = pd.NA

    return df

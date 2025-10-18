import pandas as pd

def merge_datasets(jhu_df, mobility_df, oxcgrt_df):
    df = jhu_df.merge(mobility_df, on=["country", "date"], how="left")
    df = df.merge(oxcgrt_df, on=["country", "date"], how="left")
    df = df.sort_values(["country", "date"]).fillna(method="ffill")
    return df

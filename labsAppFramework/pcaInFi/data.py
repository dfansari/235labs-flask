"""Prepare data for Plotly Dash."""
import numpy as np
import pandas as pd


def create_dataframe_ust():
    df = pd.read_csv("data/ust_data.csv", parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df.replace("N/A", np.nan, inplace=True)
    df.set_index('date', inplace=True)
    return df


def convert_ust_tenor_to_years(ust_tenor):
    if ust_tenor[-2:] == 'mo':
        return float(ust_tenor[:-2]) / 12
    elif ust_tenor[-2:] == 'yr':
        return float(ust_tenor[:-2])


def get_ust_correlation_frame(df_ust_raw):
    df_ust = df_ust_raw.loc[:, df_ust_raw.columns != '2mo']
    df_ust_diff = df_ust.diff()

    # Calculate Correlation
    df_ust_corr_matrix = df_ust_diff.corr().round(2)
    for r in range(len(df_ust_corr_matrix.index)):
        for c in range(len(df_ust_corr_matrix.columns)):
            if c >= r:
                df_ust_corr_matrix.iloc[r, c] = np.nan

    return df_ust_corr_matrix
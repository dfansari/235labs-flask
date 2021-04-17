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

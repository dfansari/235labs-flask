"""Prepare data for Plotly Dash."""
import numpy as np
import pandas as pd


def create_dataframe():
    """Create Pandas DataFrame from local CSV."""
    df = pd.read_csv("data/311-calls.csv", parse_dates=["created"])
    df["created"] = df["created"].dt.date
    df.drop(columns=["incident_zip"], inplace=True)
    num_complaints = df["complaint_type"].value_counts()
    to_remove = num_complaints[num_complaints <= 30].index
    df.replace(to_remove, np.nan, inplace=True)
    return df

def create_dataframe_ust():
    df = pd.read_csv("data/ust_data.csv", parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df.replace("N/A", np.nan, inplace=True)
    df.set_index('date', inplace=True)
    return df

def convert_ust_tenor_to_years(ust_tenor):
    if ust_tenor[-2:] == 'mo':
        return float(ust_tenor[:-2])/12
    elif ust_tenor[-2:] == 'yr':
        return float(ust_tenor[:-2])
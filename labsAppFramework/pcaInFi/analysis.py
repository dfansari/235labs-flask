import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from plotly import graph_objects as go
from plotly.offline import plot
from labsAppFramework.pcaInFi.data import *

# Load DataFrame
df_ust_raw = create_dataframe_ust()

def get_ust_correlation_frame(df_ust_raw):
    df_ust = df_ust_raw.loc[:, df_ust_raw.columns != '2mo']
    df_ust_diff = df_ust.diff()

    # Calculate Correlation
    df_ust_corr_matrix = df_ust_diff.corr().round(1)
    for r in range(len(df_ust_corr_matrix.index)):
        for c in range(len(df_ust_corr_matrix.columns)):
            if c >= r:
                df_ust_corr_matrix.iloc[r, c] = np.nan

    return df_ust_corr_matrix

df_ust_corr_matrix = get_ust_correlation_frame(df_ust_raw)

# Generate Correlation Plot

fig = go.Figure(data=go.Heatmap(
    z=df_ust_corr_matrix.T.values,
    x=df_ust_corr_matrix.T.index,
    y=df_ust_corr_matrix.T.columns,
    xgap=1, ygap=1,
))
plot(fig)

fig, ax = plt.subplots(figsize=(7,5))
mask = np.triu(np.ones_like(df_ust_corr_matrix, dtype=bool))
sns.heatmap(df_ust_corr_matrix, mask=mask, annot=True)
ax.set_title(label='US Treasury Tenor Correlation Matrix\nDaily Change in Basis Points, 2008-2021')
fig.savefig("ust_chg_correlation_plot.png")

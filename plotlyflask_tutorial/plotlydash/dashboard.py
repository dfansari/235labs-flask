"""Instantiate a Dash app."""
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from .data import create_dataframe, create_dataframe_ust, convert_ust_tenor_to_years
from .layout import html_layout


def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix="/dashapp/",
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )

    # Load DataFrame
    df = create_dataframe()
    df_ust = create_dataframe_ust()
    last_date = df_ust.index.max()

    df_ust_plot = pd.DataFrame(df_ust.loc[last_date])
    df_ust_plot.index = [convert_ust_tenor_to_years(ust_tenor) for ust_tenor in df_ust_plot.index]

    # Custom HTML layout
    dash_app.index_string = html_layout

    # Create Layout
    dash_app.layout = html.Div(
        children=[
            html.H1("Understanding Yield Curve Dynamics through Principle Component Analysis"),
            html.P("""
            Understanding the performance of a portfolio of US Treasuries can be a complex task, that is well suited for the fundamental
            data analysis of Principle Component Analysis (PCA).
            """),

            dcc.Graph(
                id="current_ust_curve",
                figure=dict(
                    data=[
                        go.Scatter(
                            x=df_ust_plot.index,
                            y=df_ust_plot[last_date]
                        )
                    ],
                    layout=dict(
                        title=f"US Treasury Yield Curve: {last_date}",
                        height=500,
                        padding=150,
                    )
                ),

            ),
            create_data_table(df_ust.reset_index()),
            html.A('Data Source',
                   href="https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldYear&year=2008"),
        ],
        id="dash-container",
    )
    return dash_app.server


def create_data_table(df):
    """Create Dash datatable from Pandas DataFrame."""
    table = dash_table.DataTable(
        id="database-table",
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("records"),
        sort_action="native",
        sort_mode="native",
        style_table=dict(
            height='400px',
            overflowY='auto'
        ),
        fixed_rows=dict(headers=True),
        style_cell=dict(
            whiteSpace='normal'
        )
    )
    return table

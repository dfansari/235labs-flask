"""Instantiate a Dash app."""
from datetime import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.graph_objs as go

from .data import YieldData, convert_ust_tenor_to_years
from .layout import html_layout

text_intro = """
Understanding price changes in the US Government's Notes and Bonds (UST) can be a complex, but important task, as it drives hundreds of thousands of investment decisions each day. 
In modern portfolio management, this typically manifests in analysis of conceptual plot of the yields of US Treasury securities by year (tenor), dubbed "the Yield Curve."
As we'll discuss, Principle Component Analysis (PCA) is an especially suited data analysis for turning quantitative models into actionable insights. 
The relative stability and interpretability of PCA's output has driven decades of bond traders to intuitively employ the first three eigenvectors, 
without any conceptual understanding of the statistics behind the scenes. 
Principle Component Analysis is a favorite algorithm among Data Science and Engineering communities alike, due to it's clear interpretability and ability to reduce data dimensionality.
In the case of yield curves, it can reduce the daily changes in a multitude of securities to a few understandable patterns.
"""
text_body_1a = """
If you're looking for a deeper conceptual background than what's provided here, there are ample longer form explanations of PCA applications in Fixed Income.
"""
links_body = [
    html.A("Credit Suisse: PCA Unleashed", href="https://research-doc.credit-suisse.com/docView?language=ENG&source=emfromsendlink&format=PDF&document_id=1001969281&extdocid=1001969281_1_eng_pdf&serialid=Coz8ZUCgL92gmMydSBULHIsgm%2b9q2TPfBu%2bX1XhViIs%3d"),
    html.Br(),
    html.A("NY Federal Reserve: Deconstructing the Yield Curve",
           href="https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr884.pdf")
]
text_body_1b = """
To begin with any PCA analysis, we first must establish a high degree of covariance in our data. Simply put -- if our different tenors' yields (our features here) 
had no relationship to one another, we wouldn't be able to reduce our data. If a move in the 2yr yield can't be approximated by a movement in the 3yr, 
the we'd need both features to understand how the yield curve moves.
A quick skim of a correlation matrix confirms moves in individual tenors are extremely correlated with one another. 
Additionally correlation appears to decrease as the tenor difference increases as a percentage. i.e. 
3month - 1month = 2month = 200% of 1month = lower correlation,
30year - 20year = 10years = 50% of 20year = higher correlation
"""

def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix="/pca-for-yield-curves/",
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )

    # Load Data Object
    yield_data = YieldData()

    # Format Plot Data
    last_date = yield_data.data.index.max()
    df_ust_plot = pd.DataFrame(yield_data.data.loc[last_date])
    df_ust_plot.index = [convert_ust_tenor_to_years(ust_tenor) for ust_tenor in df_ust_plot.index]

    # Custom HTML layout
    dash_app.index_string = html_layout

    graph_yield_curve = dcc.Graph(
        id="graph-ust-curve",
        figure=dict(
            data=[
                go.Scatter(
                    x=df_ust_plot.index,
                    y=df_ust_plot[last_date] / 100
                )
            ],
            layout=dict(
                title=f"US Treasury Yield Curve: {datetime.strftime(last_date, '%Y.%m.%d')}",
                height=400,
                width=800,
                padding=50,
                xaxis=dict(title='Security Duration (years)'),
                yaxis=dict(title='Yield', tickformat=',.1%', )
            )
        ),
    )

    graph_ust_correlation = dcc.Graph(
        id="graph-ust-correlation",
        figure=dict(
            data=[
                get_heatmap_plot(yield_data.correlation_matrix,
                                 hovertemplate='Tenor 1: %{x}<br>Tenor 2: %{y}<br>Correlation: %{z}<extra></extra>')
            ],
            layout=dict(
                title=f"UST Tenor Correlation Matrix, 1day bp changes",
                height=500,
                padding=150,
            )
        ),
    )

    data = yield_data.gaussian_test_by_year
    graph_ust_normality = dcc.Graph(
        id="graph-ust-normality",
        figure=dict(
            data=[
                get_heatmap_plot(yield_data.gaussian_test_by_year,
                                 colorscale=[[0, "rgb(255,255,255)"],
                                             [0.10, "rgb(90,220,200)"],
                                             [1, "rgb(43, 204, 162)"]],
                                 hovertemplate='Year: %{x:.0f}<br>Tenor: %{y}<br>P-value: %{z}<extra></extra>'
                                 )
            ],
            layout=dict(
                title=f"UST D'Agostino Gaussian Distribution Test, 1day bp changes",
                height=500,
                padding=150,
            )
        ),
    )


    # Create Layout
    dash_app.layout = html.Div(
        children=[
                     html.H1("Understanding Yield Curve Dynamics through Principle Component Analysis"),
                     graph_yield_curve,
                     html.P(text_intro),
                     html.P(text_body_1a)
                 ] +
                 links_body +
                 [
                     html.P(text_body_1b),
                     graph_ust_correlation,
                     graph_ust_normality,
                     html.H3("Raw Data"),
                     create_data_table(yield_data.data.reset_index()),
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


def get_heatmap_plot(data, colorscale=[[0, "rgb(255,255,255)"], [1, "rgb(43, 204, 162)"]],
                     hovertemplate=None):
    return go.Heatmap(
        z=data.values,
        y=data.index,
        x=data.columns,
        xgap=1, ygap=1,
        colorscale=colorscale,
        hovertemplate=hovertemplate
    )
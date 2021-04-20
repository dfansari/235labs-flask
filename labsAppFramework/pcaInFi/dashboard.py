"""Instantiate a Dash app."""
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.graph_objs as go

from .data import create_dataframe_ust, convert_ust_tenor_to_years, get_ust_correlation_frame
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

    # Load DataFrame
    df_ust = create_dataframe_ust()
    last_date = df_ust.index.max()

    df_ust_plot = pd.DataFrame(df_ust.loc[last_date])
    df_ust_plot.index = [convert_ust_tenor_to_years(ust_tenor) for ust_tenor in df_ust_plot.index]

    df_ust_corr_matrix = get_ust_correlation_frame(df_ust)

    # Custom HTML layout
    dash_app.index_string = html_layout

    # Create Layout
    dash_app.layout = html.Div(
        children=[
                     html.H1("Understanding Yield Curve Dynamics through Principle Component Analysis"),
                     html.P(text_intro),
                     html.P(text_body_1a)
        ] +
        links_body +
        [
             dcc.Graph(
                 id="graph-ust-curve",
                 figure=dict(
                     data=[
                         go.Scatter(
                             x=df_ust_plot.index,
                             y=df_ust_plot[last_date]/100
                         )
                     ],
                     layout=dict(
                         title=f"US Treasury Yield Curve: {last_date}",
                         height=400,
                         width=800,
                         padding=50,
                         xaxis=dict(title='Security Duration (years)'),
                         yaxis=dict(title='Yield', tickformat=',.1%',)
                     )
                 ),
             ),
             html.P(text_body_1b),
             dcc.Graph(
                 id="graph-ust-correlation",
                 figure=dict(
                     data=[
                         go.Heatmap(
                             z=df_ust_corr_matrix.T.values,
                             x=df_ust_corr_matrix.T.index,
                             y=df_ust_corr_matrix.T.columns,
                             xgap=1, ygap=1,
                             colorscale=[[0, "rgb(255,255,255)"], [1, "rgb(43, 204, 162)"]]
                         )
                     ],
                     layout=dict(
                         title=f"UST Tenor Correlation Matrix",
                         height=500,
                         padding=150,
                     )
                 ),
             ),
             html.H3("Raw Data"),
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

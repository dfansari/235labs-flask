"""Instantiate a Dash app."""
from datetime import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.graph_objs as go

from .data import YieldData, convert_ust_tenor_to_years, PcaAnalysis
from .layout import html_layout

color_palette = ['#420c30', '#b41039', '#d6423b', '#e58637', '#ffb400']

text_intro_p1 = """Understanding price changes of US Government Notes and Bonds (UST) is a complex task driving thousands of investment decisions each day. This typically manifests in examining the conceptual plot of the yields of US Treasury securities by year (tenor), dubbed "the Yield Curve". Yield changes are measured in “basis points”, or 1/100th of 1% in yield (i.e. 30Yr yield decreasing from 2% to 1.90% = -10bp)."""
text_intro_p2 = """Principal Component Analysis (PCA) is an especially suited method for this task. Due to it's clear interpretability and ability to reduce data dimensionality, PCA is a favorite algorithm among Data Science and Engineering communities alike. This is much the case when applying PCA to yield curves, as the result is so stable and easily interpreted that decades of bond traders have intuitively employed PCA’s findings, without any regard for the statistics behind the scenes."""
text_intro_pca_links = """If you're looking for a deeper conceptual background than what's provided here, there are ample longer form explanations of PCA applications in Fixed Income."""
text_correlation = """To begin with PCA analysis, we first must establish a high degree of covariance in our data (daily yield changes). If our different tenors' yields (our features here) had no relationship to one another, we wouldn't be able to reduce our data — if 2yr Yield variance is unrelated to 3yr, then we'd need both to understand how the yield curve moves as a whole. A quick skim of a correlation matrix confirms moves in individual tenors are highly correlated with one another. Additionally, correlation appears to increase as tenors are closer together."""
text_normalization = """Traditionally data would be normalized before applying PCA. Testing yield changes for normal (Gaussian) distribution, however, reveals a high degree of non-stationarity. While annual datasets notably fail testing for non-normal distribution (p-value > 0.05), all tenors produce significant p-values for non-normality tests when looking across multiple years. This means analysis could be done with rolling normalization, but for simplicity, and to mirror real-life risk management using yield-based metrics, I’ve run the PCA using unnormalized yield changes."""
text_maths = """PCA identifies the orthogonal axes (“Eigenvectors”) onto which your data can be transformed to produce the greatest variance. Here I use a partial example of only [x, y, z] = [2yr, 7yr, 30yr] to allow for 3D visual representation. The original x-axis of 1D change in 2yr yields varies from -45.0 bps to +38bps. The transformed x-axis of the first Eigenvector however varies from -101 to + 103 by accounting for the additional correlated variances of other Tenors. Subsequent Eigenvectors are then calculated orthogonally to the prior by the same process of variance maximization along the new axis. As the data is recast onto these axes, it creates the “Principal Components” of our data as they explain its variance."""
text_results = """Treasury traders typically only reference the first three Principal components, as they explain a cumulative 82% of Yield Curve variance. Normalizing the Eigenvectors with respect to a +1bp change in the 30year yield shows us their intuitive shapes: (1) A parallel shift in yields, (2) a steepening/flattening of the curve pivoting around the 5yr tenor, and (3) an increase/decrease in curve convexity."""
text_implications = """Looking at the Yield Curve through the lens of Principal Components can help traders manage their risk, as well as position their portfolio precisely for the impacts of macroeconomic events. Unscientifically, the first Principal component can be understood as broad economic events driving investors to buy or sell US Treasuries regardless of tenor. Given the second Principal Component’s slope is steepest in the shorter tenors, it can be more closely associated with actions of the Federal Reserve controlling short-term interest rates. The last major Component can be thought of as a shifting forecasted position in the economic cycle (i.e. reduced yields in the middle of the curve suggest possible mid-term recession and normalization in the long run). Understandably, these three fundamental drivers of US Treasury pricing are predictable in that order, and correspond to decreasing explained variance by PCA."""

links_intro = [
    html.A("Credit Suisse: PCA Unleashed", href="https://research-doc.credit-suisse.com/docView?language=ENG&source=emfromsendlink&format=PDF&document_id=1001969281&extdocid=1001969281_1_eng_pdf&serialid=Coz8ZUCgL92gmMydSBULHIsgm%2b9q2TPfBu%2bX1XhViIs%3d"),
    html.Br(),
    html.A("NY Federal Reserve: Deconstructing the Yield Curve",
           href="https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr884.pdf")
]

def init_dashboard_pca(server):
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

    df_eigen_vectors = yield_data.pca.eigen_vectors
    df_eigen_vectors = df_eigen_vectors.divide(df_eigen_vectors['30yr'], axis=0).round(2)

    df_eigen_table = df_eigen_vectors[:]
    df_eigen_table.index += 1
    df_eigen_table = df_eigen_table.reset_index().rename(columns=dict(index='Eigen Vector'))
    df_eigen_table['Expl. Var.'] = yield_data.pca.explained_variance.explained_variance[:len(df_eigen_table)].round(0)

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

    table_eigen_vectors = create_data_table(df_eigen_table, id='tbl-eigen-vectors')

    graph_eigen_vectors = dcc.Graph(
        id='graph-eigen-vectors',
        figure=get_eigen_line_plot(df_eigen_vectors, yield_data)
    )

    graph_eigen_scatter = dcc.Graph(
        id='graph-eigen-scatter',
        figure=get_eigen_scatter_plot(yield_data)
    )

    # Create Layout
    dash_app.layout = html.Div(
        children=[
                     html.H1("Understanding Yield Curves through Principle Component Analysis"),
                     html.H3("Context"),
                     html.P(text_intro_p1),
                     graph_yield_curve,
                     html.P(text_intro_p2)
                 ] +
                 links_intro +
                 [
                     html.H3("Correlation"),
                     html.P(text_correlation),
                     graph_ust_correlation,
                     html.H3("Normalization"),
                     html.P(text_normalization),
                     graph_ust_normality,
                     html.H3("Mathematical Foundation"),
                     html.P(text_maths),
                     graph_eigen_scatter,
                     html.H3("Principal Components"),
                     html.P(text_results),
                     table_eigen_vectors,
                     graph_eigen_vectors,
                     html.H3("Implications and Interpretations"),
                     html.P(text_implications),
                     html.H3("Raw Data"),
                     create_data_table(yield_data.data.reset_index(), id='tbl-raw-data'),
                     html.A('Data Source',
                            href="https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldYear&year=2008"),
                 ],
        id="dash-container",
    )
    return dash_app.server


def create_data_table(df, id):
    """Create Dash datatable from Pandas DataFrame."""
    table = dash_table.DataTable(
        id=id,
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict("records"),
        sort_action="native",
        sort_mode="native",
        style_data=dict(
            whiteSpace='normal'
        ),
        style_table=dict(
            height='400px',
            #overflowY='auto'
        ),
        #fixed_rows=dict(headers=True),
        # style_cell=dict(
        #     whiteSpace='normal'
        # )
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


def get_eigen_scatter_plot(yield_data, x='2yr', y='7yr', z='30yr'):
    data = yield_data.data_chg
    pca = PcaAnalysis(data, tenors=[x,y,z])
    df_eigen_1 = pca.get_eigen_plot_data(x, y, z, eigen_value=0)
    df_eigen_2 = pca.get_eigen_plot_data(x, y, z, eigen_value=1)
    df_eigen_3 = pca.get_eigen_plot_data(x, y, z, eigen_value=2)

    hover_template = 'i: %{x:.0f}<br>i: %{y}<br>i: %{z}<extra></extra>' \
        .replace("i", x, 1).replace("i", y, 1).replace("i", z, 1)

    fig = go.Figure(
        data=[
                 go.Scatter3d(x=data[x], y=data[y], z=data[z],
                              mode='markers',
                              marker=dict(
                                  color=color_palette[0],
                                  size=5,
                                  opacity=0.5),
                              name='Historic 1D Yield Changes',
                              hovertemplate=hover_template
                              ),
             ] + [go.Scatter3d(x=dataframe[x], y=dataframe[y], z=dataframe[z],
                               mode='lines',
                               line=dict(
                                   dash='dash',
                                   color=color_palette[eigen+1]),
                               name=f'Eigen Vector {eigen + 1}: '
                                    f'{pca.explained_variance.explained_variance[eigen]:.0f}% variance explained',
                               hovertemplate=hover_template
                               ) for eigen, dataframe in enumerate([df_eigen_1, df_eigen_2, df_eigen_3])],
        layout=dict(
            title='Daily UST Yield Change PCA',
            height=1000,
            scene=dict(
                xaxis=dict(title=x),
                yaxis=dict(title=y),
                zaxis=dict(title=z)
            )
        )
    )
    return fig

def get_eigen_line_plot(df_eigen_table, yield_data):
    fig = go.Figure(
        data=[
                 go.Scatter(x=df_eigen_table.loc[i].index, y=df_eigen_table.loc[i].values,
                            mode='lines',
                            line=dict(color=color_palette[i]),
                            name=f"Component {i+1}: Explained Variance={yield_data.pca.explained_variance.explained_variance[i]:.0f}%"
                            )
                 for i in range(3)]+
             [go.Scatter(x=df_eigen_table.loc[0].index, y=[0 for i in df_eigen_table.loc[0].index],
                         mode='lines',
                         line=dict(color='gray', dash='dash'),
                         opacity=0.5,
                         showlegend=False
                         )],
        layout=dict(
            title='UST Principal Components, Normalized to 1bp 30yr Change',
            xaxis=dict(title='Tenor'),
            yaxis=dict(title='Change in Bps', zeroline=True)
        )
    )
    return fig
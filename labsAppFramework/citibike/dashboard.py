"""Instantiate a Dash app."""

# dash
from dash import dcc
from dash import html
from dash import Dash, dcc, html, Input, Output, callback, dash_table
from .layout import html_layout
# visualization
import plotly.graph_objs as go
from plotly.subplots import make_subplots
# computation
import pandas as pd
import numpy as np
# data i/o and data types
import json
from urllib.request import urlopen
import pickle
import datetime
from pytz import timezone
# ML
from sklearn.metrics import r2_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def init_dashboard_citibike(server):
    """Create a Plotly Dash dashboard."""
    dash_app = Dash(
        server=server,
        routes_pathname_prefix="/citibike/",
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )

    # Custom HTML layout
    dash_app.index_string = html_layout

    # create station selector
    df_station_flows = pd.read_csv("data/citibike/2023_station_hourly_trends.csv")
    stations_ids = [col for col in df_station_flows.columns if col not in ["estimator", "metric", "hour"] and "Unnamed" not in col]

    df_station_metadata = pd.read_csv("data/citibike/2023_station_summary.csv")
    df_station_id_name_map = df_station_metadata[["station_id", "station_name"]].loc[df_station_metadata.station_id.isin(stations_ids)]
    df_station_id_name_map = df_station_id_name_map.groupby(by="station_name")[["station_id"]].first().reset_index()
    station_name_id_map = {value:name for [name, value] in df_station_id_name_map.values}

    init_station_id = "4419.03"
    station_selector = dcc.Dropdown(
        id="station-selector", options=station_name_id_map, value=init_station_id, multi=False, clearable=False
    )
    
    text_intro = """
        Using publically available data on CitiBike trips and stations status, I have greated the below 
        interactive visualizations and tools to understand both the New York CitiBike system as a whole, as well 
        as understand the traffic dynamics of an individual stations. Throughout the analysis, you'll see a few metrics include:
        Inbound Rides = Rides arriving at the station, Outbound Rides= Rides departing the station, 
        Total Rides = Outbound + Inbound Rides, Net Rides = Inbound - Outbound Rides
    """
    text_map = """
        The below map contains all of the stations which have had rides in 2023. The stations are scaled based on the
        total traffic that they receive and colored based on their clustered neighbhorhood. Hovering over a station
        will show it's station name, which can be used below to investigate the exact dynamics of the station.
    """
    text_neighborhoods = """
        As colored in the map, I've classified CitiBike stations using K-means clustering on their latitude and longitude coordinates.
        These clusters effectively identify 30 'neighborhoods' of similarly located Citibikes
    """
    text_station = """
        The below visualizes the current status, historic trends and predicted demand of a selected station.
    """
    text_station_status = """
        This table represents a real-time call to the CitiBike status API.
    """
    text_station_history = """
        The below visualizes the historic total rides (inbound + outbound), and net flows (inbound - outbound) for the station over 2023
    """
    text_station_prediction = """
        The below visualizes historic hourly outflows from the station, as well as a regression-predicted outflow for this current hour. 
        Not all station models are equally predictive based on data availability, so the right hand side shows the R Squared of the 
        station-specific regression vs. the R Squared's of other stations. Regressions have been done on a per-station basis, as
        much of the outflows are attributed to relatively unique properties. Within a station regression, day of week, month and hour of day are features.
    """

    # Create Layout
    dash_app.layout = html.Div(
        children=[
            html.H1("Citibike Data Analysis"),
            html.P(text_intro),
            
            html.H2("City-wide Visualizations"),
            html.H3("Station Map"),
            html.P(text_map),
            get_map_chart(),
            html.H3("Neighborhood Flows"),
            html.P(text_neighborhoods),
            get_total_flows_heatmap_by_neighborhood(),
            get_net_flows_heatmap_by_neighborhood(),
            get_monthly_total_flows_by_neighborhood(),
            
            html.H2("Station Visualizations"),
            html.P(text_station),
            html.Div([html.P("Select a Station:"), station_selector]),
            html.H3("Station Current Status"),
            html.P(text_station_status),
            html.Div([get_station_metadata_and_status(station_id=init_station_id)], id="station-status-div"),
            html.H3("Station Historic Flows"),
            html.P(text_station_history),
            html.Div([graph_station_hourly_trends(station_id=init_station_id)], id="station-hourly-div"),
            html.H3("Predicted Station Flows"),
            html.P(text_station_prediction),
            html.Div([get_model_prediction_graph(station_id=init_station_id, now=datetime.datetime.now(timezone('US/Eastern')))], id="station-model-div"),
        ],
        id="dash-container",
    )

    @dash_app.callback(
        Output(component_id='station-status-div', component_property='children'),
        Input(component_id='station-selector', component_property='value')
    )
    def update_station_status(input_station_id):
        return get_station_metadata_and_status(station_id=input_station_id)

    @dash_app.callback(
        Output(component_id='station-hourly-div', component_property='children'),
        Input(component_id='station-selector', component_property='value')
    )
    def update_station_hourly(input_station_id):
        return graph_station_hourly_trends(station_id=input_station_id)

    @dash_app.callback(
        Output(component_id='station-model-div', component_property='children'),
        Input(component_id='station-selector', component_property='value'),
    )
    def update_station_model(input_station_id, now):
        graph = get_model_prediction_graph(station_id=input_station_id, now=datetime.datetime.now(timezone('US/Eastern')))
        if graph is not None:
            return graph
        else:
            return html.P("Regression unavailable for station")

    return dash_app.server


def get_map_chart():
    geojson_nyc = json.load(open("data/citibike/nyc_neighborhoods.geojson"))
    df_nyc_population = pd.read_csv("data/citibike/New_York_City_Population_By_Neighborhood_Tabulation_Areas.csv")
    df_station_total = pd.read_csv("data/citibike/2023_station_summary.csv")
    df_station_total["label"] = df_station_total.apply(
        lambda row:
        f"{row['station_name']}<br>Neighborhood {row['neighborhood']}<br>Station ID {row['station_id']}",
        axis=1
    )


    def calibrate_size(x, scale_factor=0.25, max_size=15):
        return (x**scale_factor / max(x**scale_factor)) * max_size

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=geojson_nyc,
            locations=df_nyc_population["NTA Code"],
            featureidkey="properties.ntacode",
            z=df_nyc_population["Population"],
            colorscale="gray",
            colorbar=dict(title="population"),
            reversescale=True,
            text=df_nyc_population["NTA Name"],
            marker=dict(opacity=0.3),
        )
    )

    fig.add_scattermapbox(
        lat = df_station_total["lat"],
        lon = df_station_total["lon"],
        text = df_station_total["label"],
        marker = dict(
            color=df_station_total['neighborhood'],
            colorscale='agsunset',
            opacity = 0.3,
            size = calibrate_size(df_station_total["total_rides"]),
        )
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10,
        mapbox_center = {"lat": df_station_total.lat.mean(), "lon": df_station_total.lon.mean(),}
    )
    fig.update_layout(margin={"r":10,"t":10,"l":10,"b":10})

    graph_map = dcc.Graph(
        id="graph-citibike-map",
        figure=fig
    )
    return graph_map

def get_total_flows_heatmap_by_neighborhood():
    df_total_flows = pd.read_csv("data/citibike/2023_neighborhood_hourly_total_flows.csv").set_index("neighborhood")
    df_total_flows = df_total_flows.divide(df_total_flows.abs().max(axis=1).max(), axis=0)

    colorscale=[[0, "rgb(255,255,255)"], [1, "rgb(0, 0, 0)"]]
    hovertemplate=None

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=df_total_flows.values,
            y=df_total_flows.index.astype(str),
            x=df_total_flows.columns.astype(str),
            xgap=1, ygap=1,
            colorscale=colorscale,
            hovertemplate=hovertemplate
        )
    )

    fig.update_layout(
        title="Hourly Total Rides By Neighbhorhood",
        xaxis=dict(title="Hour"),
        yaxis=dict(title="Neighbhorhood Number")
    )

    graph_heatmap = dcc.Graph(
        id="graph-heamap-neighborhood-hourly-total",
        figure=fig
    )
    return graph_heatmap

def get_net_flows_heatmap_by_neighborhood():
    df_net_flows = pd.read_csv("data/citibike/2023_neighborhood_hourly_net_flows.csv").set_index("neighborhood")
    df_net_flows = df_net_flows.divide(df_net_flows.abs().max(axis=1).max(), axis=0)

    colorscale=[[0, "rgb(232, 63, 91)"], [0.5, "rgb(255, 255, 255)"], [1, "rgb(81, 174, 245)"]]
    hovertemplate=None

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=df_net_flows.values,
            y=df_net_flows.index.astype(str),
            x=df_net_flows.columns.astype(str),
            xgap=1, ygap=1,
            colorscale=colorscale,
            hovertemplate=hovertemplate
        )
    )

    fig.update_layout(
        title="Hourly Net Rides By Neighbhorhood",
        xaxis=dict(title="Hour"),
        yaxis=dict(title="Neighbhorhood Number")
    )

    graph_heatmap = dcc.Graph(
        id="graph-heamap-neighborhood-hourly-net",
        figure=fig
    )
    return graph_heatmap

def get_monthly_total_flows_by_neighborhood():
    df_total_flows = pd.read_csv("data/citibike/2023_neighborhood_monthly_total_flows.csv").set_index("neighborhood")
    df_total_flows = df_total_flows.divide(df_total_flows.abs().max(axis=1), axis=0).T

    fig = go.Figure()

    for neighborhood in df_total_flows.columns:
        fig.add_trace(
            go.Scatter(
                x=df_total_flows.index,
                y=df_total_flows[neighborhood],
                name=f"Neighborhood {neighborhood}",
            )
        )

    fig.update_layout(
        title="Monthly Rides By Neighbhorhood",
        xaxis=dict(title="Month"),
        yaxis=dict(title="Rides vs. Peak Month")
    )
    
    graph_monthly_neighborhood = dcc.Graph(
        id="graph-neighborhood-monthly-total",
        figure=fig
    )
    return graph_monthly_neighborhood

def graph_station_hourly_trends(station_id):
    # get station data
    df_station_hourly_trends = pd.read_csv("data/citibike/2023_station_hourly_trends.csv")
    df_station_metadata = pd.read_csv("data/citibike/2023_station_summary.csv")
    df_station_metadata["station_id"] = df_station_metadata["station_id"].astype(str)

    # subset station data
    station_metadata = df_station_metadata.loc[df_station_metadata.station_id == station_id].iloc[0]
    df_station_hourly = df_station_hourly_trends[["estimator", "metric", "hour", station_id]]
    df_station_hourly = pd.pivot(
        df_station_hourly,
        index="hour",
        columns=["metric", "estimator"],
        values=station_id
    )

    # plot
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[
            f"{station_metadata['station_name']}<br>Hourly Total Rides", 
            f"{station_metadata['station_name']}<br>Hourly Net Rides", 
        ]
    )

    fig.add_trace(
        go.Bar(
            x=df_station_hourly.index,
            y=df_station_hourly["total_rides"]["mean"],
            error_y=dict(type="data", array=df_station_hourly["total_rides"]["std"]),
            name="Avg Total Rides / Hour",
            marker=dict(color="gray")
        ),
        row=1, col=1
    )
    fig.update_xaxes(title="Hour", row=1, col=1)
    fig.update_yaxes(title="Avg Total Rides / Hour", row=1, col=1)

    for is_positive in [True, False]:
        fig.add_trace(
            go.Bar(
                x=df_station_hourly.loc[(df_station_hourly.net_rides["mean"] >= 0) == is_positive].index,
                y=df_station_hourly.loc[(df_station_hourly.net_rides["mean"] >= 0) == is_positive]["net_rides"]["mean"],
                error_y=dict(type="data", array=df_station_hourly["net_rides"]["std"]),
                name="Avg Net Flows / Hour",
                showlegend=is_positive,
                marker=dict(color="rgb(81, 174, 245)" if is_positive else "rgb(232, 63, 91)")
            ),
            row=1, col=2
        )

    fig.update_xaxes(title="Hour", row=1, col=2)
    fig.update_yaxes(title="Avg Net Rides / Hour", row=1, col=2)

    graph_station_hourly = dcc.Graph(
        id="graph-station-hourly",
        figure=fig
    )
    return graph_station_hourly

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

def get_station_metadata_and_status(station_id):
    URL_STATION_STATUS = "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_status.json"
    with urlopen(URL_STATION_STATUS) as url:
        data_station_status = json.load(url)

    columns = ['station_id', 'num_docks_available', 'num_ebikes_available', 'num_bikes_available', 'is_renting']
    # other_columns = ['num_bikes_disabled', 'is_installed', 'vehicle_types_available', 'last_reported', 'num_docks_disabled', 'is_returning']
    station_data = {column:[] for column in columns}
    for station in data_station_status["data"]["stations"]:
        for column in columns:
            station_data[column].append(station[column])

    df_station_status = pd.DataFrame(station_data)

    URL_STATION_METADATA = "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_information.json"
    with urlopen(URL_STATION_METADATA) as url:
        data_station_metadata = json.load(url)

    columns = ['station_id', 'short_name', 'capacity', 'lat', 'lon', 'region_id', 'name']
    station_data = {column:[] for column in columns}
    for station in data_station_metadata["data"]["stations"]:
        for column in columns:
            if column in station.keys():
                station_data[column].append(station[column])
            else:
                station_data[column].append(np.nan)

    df_station_metadata = pd.DataFrame(station_data)
    for col in ["lat", "lon", "capacity"]:
        df_station_metadata[col] = pd.to_numeric(df_station_metadata[col], errors='coerce')


    df_station_metadata = pd.merge(
        df_station_metadata, df_station_status,
        left_on="station_id", right_on="station_id",
        how="inner"
    )

    df_station_info = df_station_metadata \
        .loc[df_station_metadata.short_name == station_id] \
        [["name", "short_name", "region_id", "is_renting", "capacity", "num_docks_available", "num_bikes_available", "num_ebikes_available"]] \
        .T \
        .reset_index()
    df_station_info.columns = ["metric", "value"]

    table = create_data_table(df_station_info, "station-status")

    return table

def get_model_prediction_graph(station_id, now):
    # load model data
    filehandler = open("data/citibike/models_dict.pkl",'rb')
    station_models_dict = pickle.load(filehandler)

    if station_id not in station_models_dict.keys():
        return None

    # get distribution of model r2s
    df_r_2s = pd.DataFrame(np.array([[k, v["r_2"]] for k, v in station_models_dict.items()]))
    df_r_2s.columns = ["station_id", "r_2"]
    df_r_2s["r_2"] = df_r_2s["r_2"].astype(float)

    # # get staion model
    model = station_models_dict[station_id]["model"]
    r_2 = station_models_dict[station_id]["r_2"]

    # get current prediction
    df_curr_data = pd.DataFrame(dict(
        month=[now.month], weekday=[now.weekday], hour=[now.hour], 
    ))
    pred_outbound_traffic = model.predict(df_curr_data)

    # get distribution of outbound rides
    df_station_hourly_trends = pd.read_csv("data/citibike/2023_station_hourly_trends.csv")

    df_station_hourly = df_station_hourly_trends[["estimator", "metric", "hour", station_id]]
    df_station_hourly = pd.pivot(
        df_station_hourly,
        index="hour",
        columns=["metric", "estimator"],
        values=station_id
    )

    data = []
    for hour in df_station_hourly.index:
        mean, std = df_station_hourly["outbound_rides"]["mean"][hour], df_station_hourly["outbound_rides"]["std"][hour]
        data += np.random.normal(loc=mean, scale=std, size=100).tolist()

    data = np.array(data)
    data = data[data >= 0]

    # visual
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=[
            f"{station_id}<br>Hourly Outbound Rides", 
            f"R2 of Station Models"
        ]
    )

    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=20,
            histnorm="percent",
            name="Outbound Rides / Hour",
            marker=dict(color="rgb(232, 63, 91)")
        ),
        row=1, col=1
    )
    fig.add_vline(
        x=pred_outbound_traffic[0], 
        line_width=5,
        fillcolor="darkblue",
        name=f"Predicted Traffic This Hour={pred_outbound_traffic[0]:,.1f}",
        annotation_text="    Predicted Traffic this hour",
        row=1, col=1, 
    )
    fig.update_xaxes(title="Outbound Rides / Hour", row=1, col=1)
    fig.update_yaxes(title="% of Data", row=1, col=1)

    fig.add_trace(
        go.Histogram(
            x=df_r_2s["r_2"],
            nbinsx=20,
            histnorm="percent",
            name="Station Model R2",
            marker=dict(color="gray")
        ),
        row=1, col=2
    )
    fig.add_vline(
        x=r_2, 
        line_width=5,
        fillcolor="darkblue",
        name=f"Predicted Traffic This Hour={r_2:,.0%}",
        annotation_text="    Model R2",
        row=1, col=2, 
    )
    fig.update_xaxes(title="Station Model R2", row=1, col=2)
    fig.update_yaxes(title="% of Data", row=1, col=2)

    graph_station_model = dcc.Graph(
        id="graph-station-model",
        figure=fig
    )
    return graph_station_model
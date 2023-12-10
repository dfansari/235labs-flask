"""Instantiate a Dash app."""
from datetime import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.graph_objs as go
import json

from .layout import html_layout

def init_dashboard_citibike(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix="/citibike/",
        external_stylesheets=[
            "/static/dist/css/styles.css",
            "https://fonts.googleapis.com/css?family=Lato",
        ],
    )

    # Custom HTML layout
    dash_app.index_string = html_layout

    # Create Layout
    dash_app.layout = html.Div(
        children=[
            html.H1("City Wide Data"),
            get_map_chart(),
            get_total_flows_heatmap_by_neighborhood(),
            get_net_flows_heatmap_by_neighborhood(),
            get_monthly_total_flows_by_neighborhood(),
            html.H1("Station Data"),
        ],
        id="dash-container",
    )
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


    def calibrate_size(x, scale_factor=0.4, max_size=5):
        return (x**scale_factor / max(x**scale_factor)) * max_size

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=geojson_nyc,
            locations=df_nyc_population["NTA Code"],
            featureidkey="properties.ntacode",
            z=df_nyc_population["Population"],
            colorscale="gray",
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
            size = calibrate_size(df_station_total["total_rides"], max_size=10),
        )
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10,
        mapbox_center = {"lat": df_station_total.lat.mean(), "lon": df_station_total.lon.mean(),}
    )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

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


from __future__ import annotations
from dash import html, dcc, register_page
import dash_bootstrap_components as dbc

register_page(__name__, path="/browse")

layout = dbc.Container([
    html.H3("Oplo – Local Browser"),
    dbc.Alert(id="browse-alert", is_open=False, color="warning", className="mb-2"),
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText("Folder"),
                dcc.Dropdown(id="dir-select", placeholder="Select a subfolder..."),
                dbc.Button("Rescan", id="dir-rescan", n_clicks=0, color="secondary"),
            ], className="mb-2"),
            dcc.Loading(id="files-loading", type="default", children=html.Div(id="files-table")),
        ], width=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Preview"),
                dbc.CardBody([
                    html.Div(id="file-meta", className="mb-2"),
                    dcc.Graph(id="file-preview", style={"height": "45vh"}),
                    dbc.Button("Open in Viewer", id="open-in-viewer", color="primary"),
                    dcc.Location(id="browse-nav", refresh=True),
                ])
            ])
        ], width=5)
    ], className="mt-3"),
    dcc.Store(id="selected-file"),
], fluid=True)

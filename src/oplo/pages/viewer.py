# src/oplo/pages/viewer.py
from __future__ import annotations
import dash_bootstrap_components as dbc
from dash import html, dcc, register_page

register_page(__name__, path="/")

layout = dbc.Container([
    html.H3("Oplo – Viewer"),

    dcc.Upload(
        id="upload-image",
        children=html.Div(["Drag & drop or ", html.A("select files")]),
        multiple=False,
        style={
            "border": "1px dashed #888",
            "padding": "20px",
            "borderRadius": "10px"
        }
    ),

    dbc.Row([
        # Main image display
        dbc.Col(
            dcc.Graph(id="image-view", style={"height": "70vh"}),
            width=9
        ),

        # Right sidebar
        dbc.Col([

            # === Display Controls ===
            dbc.Card([
                dbc.CardHeader("Display"),
                dbc.CardBody([

                    # Gamma
                    html.Div(className="mb-2", children=[
                        html.Label("Gamma", className="me-2"),
                        dcc.Slider(
                            min=0.1, max=4, step=0.1, value=1.0,
                            id="gamma",
                            tooltip={"always_visible": True},
                            marks=None
                        )
                    ]),

                    # Auto scale checkbox
                    dcc.Checklist(
                        options=[{"label": "Auto Percentile (0.5–99.5%)", "value": "auto"}],
                        value=["auto"],
                        id="auto-scale",
                        className="mb-2"
                    ),

                    # Colormap
                    dcc.Dropdown(
                        id="colormap",
                        options=[
                            {"label": "Gray (default)", "value": "gray"},
                            {"label": "Cividis", "value": "cividis"},
                            {"label": "Viridis", "value": "viridis"},
                            {"label": "Plasma", "value": "plasma"},
                            {"label": "Magma", "value": "magma"},
                            {"label": "Inferno", "value": "inferno"},
                            {"label": "Turbo", "value": "turbo"},
                        ],
                        value="gray",
                        clearable=False,
                        className="mb-2"
                    ),

                    # Colorbar toggle
                    dcc.Checklist(
                        id="show-colorbar",
                        options=[{"label": "Show colorbar", "value": "on"}],
                        value=[],
                        className="mb-2"
                    ),

                    html.Div(
                        id="downsample-note",
                        className="text-muted",
                        style={"fontSize": "12px"}
                    ),
                ])
            ], className="mb-3"),

            # === Metadata ===
            dbc.Card([
                dbc.CardHeader("Image Info"),
                dbc.CardBody([html.Div(id="meta")])
            ], className="mb-3"),

            # === Histogram ===
            dbc.Card([
                dbc.CardHeader("Histogram"),
                dbc.CardBody([
                    dcc.Graph(id="hist-view", style={"height": "25vh"})
                ])
            ], className="mb-3"),

            # === Hover Probe ===
            dbc.Card([
                dbc.CardHeader("Hover Probe"),
                dbc.CardBody([
                    html.Pre(id="hover-readout", style={"fontSize": "12px"})
                ])
            ]),

        ], width=3)
    ], className="mt-3"),
], fluid=True)

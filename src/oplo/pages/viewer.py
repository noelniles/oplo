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

        # Right sidebar (collapsible sections)
        dbc.Col([
            # pipeline store for client-side persistence of pipeline graph
            dcc.Store(id="pipeline-store", data=[]),
            # selected node index for editing
            dcc.Store(id="selected-node", data=None),
            # intermediate store for node updates
            dcc.Store(id="node-update", data=None),

            # Processing controls accordion will be rendered here
            dbc.Accordion([
                dbc.AccordionItem([
                    # Display controls
                    html.Div(className="mb-2", children=[
                        html.Label("Gamma", className="me-2"),
                        dcc.Slider(
                            min=0.1, max=4, step=0.1, value=1.0,
                            id="gamma",
                            tooltip={"always_visible": True},
                            marks=None
                        )
                    ]),

                    dcc.Checklist(
                        options=[{"label": "Auto Percentile (0.5–99.5%)", "value": "auto"}],
                        value=["auto"],
                        id="auto-scale",
                        className="mb-2"
                    ),

                    dbc.Switch(id="roi-mode", value=False, label="ROI mode", className="mb-2"),
                    html.Div(
                        "Enable ROI mode, then drag a box on the image to compute stats.",
                        id="roi-hint",
                        className="text-muted",
                        style={"FontSize": "12px"},
                    ),

                        # Slice slider (hidden unless stack)
                        html.Div(id="slice-container", children=[
                            html.Label("Slice", className="me-2"),
                            dcc.Slider(id="slice-index", min=0, max=0, step=1, value=0),
                            html.Div(id="slice-info", className="text-muted", style={"fontSize": "12px"}),
                        ], style={"display": "none"}),
                    html.Div(id="roi-hint", className="text-muted", style={"fontSize": "12px"}),

                    # ROI stats inside the display accordion
                    dbc.Card([
                        dbc.CardHeader("ROI Stats"),
                        dbc.CardBody([
                            html.Pre(id="roi-stats", style={"fontSize": "12px", "whiteSpace": "pre-wrap"}),
                            dcc.Graph(id="roi-hist", style={"height": "2-vh"}),
                        ])
                    ], className="mb-3"),

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

                    dcc.Checklist(
                        id="show-colorbar",
                        options=[{"label": "Show colorbar", "value": "on"}],
                        value=[],
                        className="mb-2"
                    ),

                    html.Div(id="downsample-note", className="text-muted", style={"fontSize": "12px"}),
                ], title="Display"),

                dbc.AccordionItem([html.Div(id="meta")], title="Image Info"),

                dbc.AccordionItem([dcc.Graph(id="hist-view", style={"height": "25vh"})], title="Histogram"),

                dbc.AccordionItem([html.Pre(id="hover-readout", style={"fontSize": "12px"})], title="Hover Probe"),
                dbc.AccordionItem([
                    html.Div([
                        html.Label("Search Processors"),
                        dcc.Input(id="proc-search", type="text", placeholder="Search by name or tag...", style={"width": "100%"}),
                        html.Br(),
                        html.Label("Category Filter"),
                        dcc.Dropdown(
                            id="proc-category",
                            options=[],  # Will be populated by callback
                            placeholder="All categories",
                            clearable=True,
                        ),
                        html.Br(),
                        html.Label("Processor"),
                        dcc.Dropdown(
                            id="proc-select",
                            options=[],  # Will be populated by callback
                            placeholder="Select a processor...",
                            clearable=False,
                        ),
                        html.Div(id="proc-description", className="text-muted", style={"fontSize": "11px", "fontStyle": "italic"}),
                        html.Br(),
                        html.Div(id="proc-params"),  # Dynamic parameter inputs
                        html.Br(),
                        html.Div([
                            html.Label("Tile size (H x W)"),
                            dcc.Input(id="tile-h", type="number", value=512, style={"width": "100px"}),
                            dcc.Input(id="tile-w", type="number", value=512, style={"width": "100px", "marginLeft": "8px"}),
                        ]),
                        html.Br(),
                        html.Div([
                            html.Label("Overlap (H x W)"),
                            dcc.Input(id="overlap-h", type="number", value=16, style={"width": "100px"}),
                            dcc.Input(id="overlap-w", type="number", value=16, style={"width": "100px", "marginLeft": "8px"}),
                        ]),
                        html.Br(),
                        html.Div([
                            html.Label("Workers"),
                            dcc.Input(id="workers", type="number", value=4, style={"width": "80px"}),
                        ]),
                        html.Br(),
                        dbc.Button("Add Node", id="add-node", color="secondary", size="sm", className="me-2"),
                        dbc.Button("Run Pipeline", id="run-pipeline", color="primary", size="sm"),
                        dbc.Button("Clear Pipeline", id="clear-pipeline", color="warning", size="sm", className="ms-2"),
                        html.Hr(),
                        html.Div(id="pipeline-nodes", style={"fontSize": "12px"}),
                        html.Br(),
                        html.Div(id="node-editor", children=[]),
                        html.Br(),
                        html.Label("History Navigation"),
                        html.Div([
                            dbc.Button("◀", id="history-prev", size="sm", className="me-1"),
                            dbc.Button("▶", id="history-next", size="sm", className="me-2"),
                            html.Span(id="history-info", className="text-muted", style={"fontSize": "12px"}),
                        ]),
                        dcc.Dropdown(id="history-select", placeholder="Jump to stage...", style={"marginTop": "8px"}),
                        html.Br(),
                        dbc.Progress(id="process-progress", value=0, style={"height": "20px"}),
                        html.Div(id="process-status", className="text-muted", style={"fontSize": "12px"}),
                        html.Hr(),
                        html.Label("Save Results"),
                        dcc.Input(id="save-filename", type="text", placeholder="output.tif", style={"width": "100%"}),
                        html.Br(),
                        dbc.Button("Save Image", id="save-btn", color="success", size="sm", className="mt-2"),
                        html.Div(id="save-status", className="text-muted", style={"fontSize": "12px", "marginTop": "8px"}),
                        dcc.Download(id="download-data"),
                        dcc.Interval(id="process-interval", interval=1000, n_intervals=0, disabled=True),
                    ])
                ], title="Processing"),
            ], start_collapsed=True),

        ], width=3)
    ], className="mt-3"),
], fluid=True)

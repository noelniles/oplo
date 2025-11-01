# src/oplo/pages/viewer.py
from __future__ import annotations

from dataclasses import dataclass
import math
import tempfile
import base64
from typing import Tuple, Optional

import dash
from dash import html, dcc, Input, Output, State, callback, register_page
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import cv2

from ..io.image_io import load_image
from ..utils.convert import percentile_scale
from ..state import STORE


register_page(__name__, path="/")

# ---- Display policy ---------------------------------------------------------
# We keep full-precision data in memory (STORE["img"]). For visualization,
# we decimate very large images to keep interactive latency low.
# Target maximum display dimension (longest side) in pixels.
DISPLAY_MAX = 2048  # adjustable via future settings panel


@dataclass
class DisplayImage:
    disp: np.ndarray  # float32 [0,1], possibly downsampled
    scale: float      # scale factor relative to original (<=1)


def _maybe_downsample_for_display(img: np.ndarray, max_dim: int = DISPLAY_MAX) -> DisplayImage:
    """Return an image suitable for display with a scale <= 1.

    Uses OpenCV INTER_AREA downsampling when needed. Preserves channels.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return DisplayImage(disp=img, scale=1.0)
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return DisplayImage(disp=resized.astype(np.float32, copy=False), scale=scale)


layout = dbc.Container([
    html.H3("Oplo – Viewer"),
    dcc.Upload(
        id="upload-image",
        children=html.Div(["Drag & drop or ", html.A("select files")]),
        multiple=False,
        style={"border":"1px dashed #888","padding":"20px","borderRadius":"10px"}
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id="image-view", style={"height": "70vh"}), width=9),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Display"),
                dbc.CardBody([
                    html.Div(className="mb-2", children=[
                        html.Label("Gamma", className="me-2"),
                        dcc.Slider(0.1, 4, 0.1, value=1.0, id="gamma", tooltip={"always_visible": True}, marks=None)
                    ]),
                    dcc.Checklist(
                        options=[{"label":"Auto Percentile (0.5–99.5%)","value":"auto"}],
                        value=["auto"], id="auto-scale"
                    ),
                    html.Div(id="downsample-note", className="text-muted", style={"fontSize":"12px"}),
                ])
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("Image Info"),
                dbc.CardBody([
                    html.Div(id="meta"),
                ])
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("Histogram"),
                dbc.CardBody([dcc.Graph(id="hist-view", style={"height":"25vh"})])
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("Hover Probe"),
                dbc.CardBody([html.Pre(id="hover-readout", style={"fontSize":"12px"})])
            ])
        ], width=3)
    ], className="mt-3"),
], fluid=True)


@callback(
    Output("image-view", "figure"),
    Output("meta", "children"),
    Output("hist-view", "figure"),
    Output("downsample-note", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    Input("gamma", "value"),
    Input("auto-scale", "value"),
    prevent_initial_call=True
)
def update_image(contents, filename, gamma, autos):
    ctx = [p["prop_id"] for p in dash.callback_context.triggered][0]

    # 1) Handle a new upload
    if "upload-image" in ctx:
        if contents is None:
            raise dash.exceptions.PreventUpdate
        header, b64 = contents.split(",", 1)
        data = base64.b64decode(b64)
        suffix = "." + filename.split(".")[-1] if "." in filename else ""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data); tmp.flush(); tmp.close()
        img, meta = load_image(tmp.name)
        STORE.set(img=img, meta=meta)

    # 2) Build a display layer from stored full-precision
    img = STORE.get("img")
    if img is None:
        raise dash.exceptions.PreventUpdate

    # Tonemap / percentile scale on full-res first, then downsample for view
    tonemapped = percentile_scale(img, 0.5, 99.5, gamma=gamma) if "auto" in autos else np.clip(img,0,1)

    disp = _maybe_downsample_for_display(tonemapped, DISPLAY_MAX)
    STORE.set(disp=disp.disp)

    # 3) Plotly figure for display
    fig = px.imshow(disp.disp, origin="upper")
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    fig.update_xaxes(showspikes=True); fig.update_yaxes(showspikes=True)

    # 4) Meta panel
    meta = STORE.get("meta", {})
    h, w = img.shape[:2]
    ch = 1 if img.ndim == 2 else img.shape[2]
    meta_ul = html.Ul([
        html.Li(f"File: {filename}"),
        html.Li(f"Original dtype: {meta.get('orig_dtype')}") ,
        html.Li(f"Bit depth: {meta.get('bit_depth')}") ,
        html.Li(f"Colorspace: {meta.get('colorspace')}") ,
        html.Li(f"Size: {w} × {h} × {ch}"),
    ])

    # 5) Histogram on displayed image (fast)
    hist = np.clip(disp.disp,0,1).ravel()
    hfig = px.histogram(hist, nbins=256)
    hfig.update_layout(margin=dict(l=0,r=0,t=0,b=0), bargap=0)

    # 6) Downsample note
    note = None
    if disp.scale < 1.0:
        ds = f"Downsampled for view: scale={disp.scale:.3f} (max dimension {DISPLAY_MAX}px)"
        note = html.Div(ds)

    return fig, meta_ul, hfig, note


@callback(
    Output("hover-readout", "children"),
    Input("image-view", "hoverData"),
    prevent_initial_call=True
)
def show_hover(hover):
    if not hover or STORE.get("img") is None or STORE.get("disp") is None:
        raise dash.exceptions.PreventUpdate

    disp = STORE.get("disp")
    img = STORE.get("img")

    disp_h, disp_w = disp.shape[:2]
    img_h, img_w = img.shape[:2]

    scale_x = disp_w / img_w
    scale_y = disp_h / img_h

    pt = hover["points"][0]
    x_disp, y_disp = float(pt["x"]), float(pt["y"])  # display coordinates

    x = int(round(x_disp / scale_x))
    y = int(round(y_disp / scale_y))

    x = max(0, min(img_w - 1, x))
    y = max(0, min(img_h - 1, y))

    val = img[y, x] if img.ndim == 2 else img[y, x, :]
    return f"(x={x}, y={y})  value={np.array2string(val, precision=6, floatmode='maxprec')}"

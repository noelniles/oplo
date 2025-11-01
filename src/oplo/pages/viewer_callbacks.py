from __future__ import annotations

import base64, tempfile
from typing import Dict, NamedTuple

import numpy as np
import cv2
import plotly.express as px
from dash import Input, Output, State, no_update

from oplo.io.image_io import load_image, ImageBundle
from oplo.state import STORE

DISPLAY_MAX = 2048


class DisplayImage(NamedTuple):
    disp: np.ndarray
    scale: float

def _maybe_downsample_for_display(img: np.ndarray, max_dim: int = DISPLAY_MAX) -> DisplayImage:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return DisplayImage(img, 1.0)
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return DisplayImage(resized.astype(np.float32, copy=False), scale)

def register(app):
    # Main image update (bind to this app instance => no duplicate global registration)
    @app.callback(
        Output("image-view", "figure"),
        Output("meta", "children"),
        Output("hist-view", "figure"),
        Output("downsample-note", "children"),
        Input("upload-image", "contents"),
        State("upload-image", "filename"),
        Input("gamma", "value"),
        Input("auto-scale", "value"),
        Input("colormap", "value"),
        Input("show-colorbar", "value"),
        prevent_initial_call=True,
    )
    def update_image(contents, filename, gamma, autos, cmap, showbar):
        # New upload?
        if contents is not None and filename:
            header, b64 = contents.split(",", 1)
            data = base64.b64decode(b64)
            suffix = "." + filename.split(".")[-1] if "." in filename else ""
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data); tmp.flush(); tmp.close()
            bundle = load_image(tmp.name)
            STORE.set(bundle=bundle)
        else:
            bundle: ImageBundle = STORE.get("bundle")
            if bundle is None:
                return no_update, no_update, no_update, no_update

        bundle: ImageBundle = STORE.get("bundle")
        policy = "percentile" if ("auto" in (autos or [])) else "dtype_range"
        tonemapped = bundle.view(policy=policy, lo=0.5, hi=99.5, gamma=gamma or 1.0)

        disp = _maybe_downsample_for_display(tonemapped, DISPLAY_MAX)
        STORE.set(disp=disp.disp)

        img_disp = disp.disp

        # Single-channel if 2D OR 3D with a singleton channel
        is_single = (img_disp.ndim == 2) or (img_disp.ndim == 3 and img_disp.shape[2] == 1)
        if img_disp.ndim == 3 and img_disp.shape[2] == 1:
            img_disp = np.squeeze(img_disp, axis=2)  # make it 2D for Plotly

        is_rgb = (disp.disp.ndim == 3 and disp.disp.shape[2] in (3, 4))

        cmap_name = cmap or "gray"

        if is_single:
            # Apply chosen colorscale to single-channe
            fig = px.imshow(
                img_disp, origin="upper",
                zmin=0.0, zmax=1.0,
                color_continuous_scale=cmap_name,
            )
            fig.update_layout(coloraxis_showscale=bool(showbar))
        elif is_rgb:
            # True-color RGB/RGBA (ignore colorscale)
            fig = px.imshow(img_disp, origin="upper", zmin=0.0, zmax=1.0)
        else:
            # Fallback (shouldn't happen, but keep behavior predictable)
            fig = px.imshow(img_disp, origin="upper", zmin=0.0, zmax=1.0,
                            color_continuous_scale=cmap_name) 

            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            fig.update_xaxes(showspikes=True); fig.update_yaxes(showspikes=True)

            fig = px.imshow(disp.disp, origin="upper")
            fig.update_layout(coloraxis_showscale=bool(showbar)) if not is_rgb else None

        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(showspikes=True); fig.update_yaxes(showspikes=True)

        meta = bundle.meta
        h, w = bundle.data.shape[:2]
        ch = 1 if bundle.data.ndim == 2 else bundle.data.shape[2]
        from dash import html
        meta_ul = html.Ul([
            html.Li(f"File: {meta.get('path') or filename}"),
            html.Li(f"Original dtype: {meta.get('orig_dtype')}"),
            html.Li(f"Bit depth: {meta.get('bit_depth')}"),
            html.Li(f"Colorspace: {meta.get('colorspace')}"),
            html.Li(f"Size: {w} × {h} × {ch}"),
            html.Li(f"Reader: {meta.get('reader')}"),
        ])

        hist = np.clip(disp.disp, 0, 1).ravel()
        hfig = px.histogram(hist, nbins=256)
        hfig.update_layout(margin=dict(l=0, r=0, t=0, b=0), bargap=0)

        note = None
        if disp.scale < 1.0:
            from dash import html
            note = html.Div(f"Downsampled for view: scale={disp.scale:.3f} (max dimension {DISPLAY_MAX}px)")

        return fig, meta_ul, hfig, note

    # Hover probe
    @app.callback(
        Output("hover-readout", "children"),
        Input("image-view", "hoverData"),
        prevent_initial_call=True,
    )
    def show_hover(hover):
        bundle: ImageBundle = STORE.get("bundle")
        disp = STORE.get("disp")
        if not hover or bundle is None or disp is None:
            return no_update

        img = bundle.data
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

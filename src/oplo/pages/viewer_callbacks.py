# src/oplo/pages/viewer_callbacks.py
from __future__ import annotations

import base64, os, tempfile
from typing import NamedTuple, Dict

import numpy as np
import cv2
import plotly.express as px
from dash import Input, Output, State, no_update, html, ctx  # ctx to detect trigger

from oplo.io.image_io import load_image, ImageBundle
from oplo.state import STORE

DISPLAY_MAX = 2048

# Allowed UI → Plotly colormap names
_COLORMAPS: Dict[str, str] = {
    "gray": "gray",
    "cividis": "cividis",
    "viridis": "viridis",
    "plasma": "plasma",
    "magma": "magma",
    "inferno": "inferno",
    "turbo": "turbo",
}

class DisplayImage(NamedTuple):
    disp: np.ndarray
    scale: float

def _maybe_downsample_for_display(img: np.ndarray, max_dim: int = DISPLAY_MAX) -> DisplayImage:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return DisplayImage(img.astype(np.float32, copy=False), 1.0)
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return DisplayImage(resized.astype(np.float32, copy=False), scale)


def register(app):

    @app.callback(
        Output("image-view", "figure"),
        Output("meta", "children"),
        Output("hist-view", "figure"),
        Output("downsample-note", "children"),
        Output("image-view", "selectedData"),      # NEW: clear ROI on new image
        Input("upload-image", "contents"),
        State("upload-image", "filename"),
        Input("gamma", "value"),
        Input("auto-scale", "value"),
        Input("colormap", "value"),
        Input("show-colorbar", "value"),
        Input("roi-mode", "value"),               # ensure dragmode is preserved
        prevent_initial_call=True,
    )
    def update_image(contents, filename, gamma, autos, cmap, showbar, roi_on):
        triggered = ctx.triggered_id  # which input triggered this callback

        # -- Load uploaded image into ImageBundle if a new upload happened --
        is_new_upload = (triggered == "upload-image") and (contents is not None) and bool(filename)
        if is_new_upload:
            STORE.clear()
            _, b64 = contents.split(",", 1)
            data = base64.b64decode(b64)
            suffix = "." + filename.split(".")[-1] if "." in filename else ""
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                tmp.write(data)
                tmp.flush()
                tmp.close()
                bundle = load_image(tmp.name)
            except ValueError as err:
                STORE.clear()
                err_note = html.Div(str(err), className="text-danger")
                return no_update, err_note, no_update, err_note, no_update
            except Exception as err:
                STORE.clear()
                err_note = html.Div(f"Failed to load image: {err}", className="text-danger")
                return no_update, err_note, no_update, err_note, no_update
            finally:
                try:
                    tmp.close()
                except Exception:
                    pass
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
            STORE.set(bundle=bundle)
        else:
            if STORE.get("bundle") is None:
                return no_update, no_update, no_update, no_update, no_update

        bundle: ImageBundle = STORE.get("bundle")

        # Tonemap original to float32 [0,1]
        policy = "percentile" if ("auto" in (autos or [])) else "dtype_range"
        gamma = gamma or 1.0
        tonemapped = bundle.view(policy=policy, lo=0.5, hi=99.5, gamma=gamma)

        # Downsample for display
        disp = _maybe_downsample_for_display(tonemapped, DISPLAY_MAX)
        STORE.set(disp=disp.disp, scale=disp.scale)

        # Handle single-channel vs RGB logic
        img_disp = disp.disp  # display version (float32)
        if img_disp.ndim == 3 and img_disp.shape[2] == 1:
            img_disp = np.squeeze(img_disp, axis=2)

        is_single = img_disp.ndim == 2
        is_rgb = (img_disp.ndim == 3 and img_disp.shape[2] in (3, 4))
        cmap_name = _COLORMAPS.get(cmap or "gray", "gray")

        if is_single:
            fig = px.imshow(
                img_disp, origin="upper",
                zmin=0.0, zmax=1.0,
                color_continuous_scale=cmap_name,
            )
            fig.update_layout(coloraxis_showscale=bool(showbar))
        elif is_rgb:
            fig = px.imshow(img_disp, origin="upper", zmin=0.0, zmax=1.0)
        else:
            fig = px.imshow(
                img_disp, origin="upper",
                zmin=0.0, zmax=1.0,
                color_continuous_scale=cmap_name,
            )

        # Keep margins tidy and spikes on; most importantly, preserve ROI dragmode
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(showspikes=True); fig.update_yaxes(showspikes=True)
        fig.update_layout(dragmode=("select" if roi_on else "zoom"))

        # Metadata
        meta = bundle.meta
        h, w = bundle.data.shape[:2]
        ch = 1 if bundle.data.ndim == 2 else bundle.data.shape[2]
        meta_ul = html.Ul([
            html.Li(f"File: {meta.get('path') or filename}"),
            html.Li(f"Original dtype: {meta.get('orig_dtype')}"),
            html.Li(f"Bit depth: {meta.get('bit_depth')}"),
            html.Li(f"Colorspace: {meta.get('colorspace')}"),
            html.Li(f"Size: {w} × {h} × {ch}"),
            html.Li(f"Reader: {meta.get('reader')}"),
        ])

        # Histogram
        hist_src = img_disp if is_single else np.clip(img_disp, 0, 1).astype(np.float32)
        hfig = px.histogram(hist_src.ravel(), nbins=256)
        hfig.update_layout(margin=dict(l=0, r=0, t=0, b=0), bargap=0)

        # Downsample note
        note = None
        if disp.scale < 1.0:
            note = html.Div(f"Downsampled for view: scale={disp.scale:.3f} (max {DISPLAY_MAX}px)")

        # Clear selectedData only on NEW upload; otherwise leave selection intact
        selected_reset = None if is_new_upload else no_update

        return fig, meta_ul, hfig, note, selected_reset


    @app.callback(
        Output("roi-stats", "children"),
        Output("roi-hist", "figure"),
        Output("roi-hint", "children"),
        Input("image-view", "selectedData"),
        prevent_initial_call=True,
    )
    def roi_stats(selected):
        bundle: ImageBundle = STORE.get("bundle")
        disp = STORE.get("disp")
        scale = STORE.get("scale", 1.0)
        if bundle is None or disp is None or not selected:
            return no_update, no_update, "Enable ROI mode and drag a selection on the image."

        rng = selected.get("range")
        if not rng or "x" not in rng or "y" not in rng:
            return no_update, no_update, "Use box select to get ROI stats."

        x0, x1 = rng["x"][0], rng["x"][1]
        y0, y1 = rng["y"][0], rng["y"][1]

        disp_h, disp_w = disp.shape[:2]
        img = bundle.data
        img_h, img_w = img.shape[:2]
        if scale <= 0:
            scale_x = disp_w / img_w
            scale_y = disp_h / img_h
        else:
            scale_x = scale_y = scale

        x0_i = max(0, min(img_w - 1, int(np.floor(min(x0, x1) / scale_x))))
        x1_i = max(0, min(img_w,     int(np.ceil (max(x0, x1) / scale_x))))
        y0_i = max(0, min(img_h - 1, int(np.floor(min(y0, y1) / scale_y))))
        y1_i = max(0, min(img_h,     int(np.ceil (max(y0, y1) / scale_y))))

        if x1_i <= x0_i or y1_i <= y0_i:
            return "Empty selection.", no_update, no_update

        roi = img[y0_i:y1_i, x0_i:x1_i] if img.ndim == 2 else img[y0_i:y1_i, x0_i:x1_i, :]

        # If RGB, convert to luminance for stats
        if roi.ndim == 3 and roi.shape[2] in (3, 4):
            r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
            roi_scalar = 0.2126*r + 0.7152*g + 0.0722*b
        else:
            roi_scalar = roi

        roi_flat = roi_scalar.ravel().astype(np.float64, copy=False)
        if roi_flat.size == 0:
            return "Empty selection.", no_update, ""
        if np.all(np.isnan(roi_flat)):
            return "ROI contains only NaN values.", no_update, ""

        count = int(roi_flat.size)
        vmin  = float(np.nanmin(roi_flat))
        vmax  = float(np.nanmax(roi_flat))
        mean  = float(np.nanmean(roi_flat))
        std   = float(np.nanstd(roi_flat))
        p1, p50, p99 = [float(np.nanpercentile(roi_flat, q)) for q in (1, 50, 99)]

        stats_text = (
            f"Raw ROI: x[{x0_i}:{x1_i}), y[{y0_i}:{y1_i})  "
            f"w={x1_i-x0_i}, h={y1_i-y0_i}, n={count}\n"
            f"min={vmin:.6g}, max={vmax:.6g}\n"
            f"mean={mean:.6g}, std={std:.6g}\n"
            f"p1={p1:.6g}, p50={p50:.6g}, p99={p99:.6g}"
        )

        hist_fig = px.histogram(roi_flat, nbins=128)
        hist_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), bargap=0)

        return stats_text, hist_fig, ""


    @app.callback(
        Output("hover-readout", "children", allow_duplicate=True),
        Input("image-view", "hoverData"),
        prevent_initial_call=True,
    )
    def hover(hover):
        bundle: ImageBundle = STORE.get("bundle")
        disp = STORE.get("disp")
        scale = STORE.get("scale", 1.0)
        if not hover or bundle is None or disp is None:
            return no_update

        img = bundle.data
        disp_h, disp_w = disp.shape[:2]
        img_h, img_w = img.shape[:2]

        if scale <= 0:
            scale_x = disp_w / img_w
            scale_y = disp_h / img_h
        else:
            scale_x = scale_y = scale

        pt = hover["points"][0]
        x = int(round(pt["x"] / scale_x))
        y = int(round(pt["y"] / scale_y))
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))

        val = img[y, x] if img.ndim == 2 else img[y, x, :]
        return f"(x={x}, y={y}) value={np.array2string(val, precision=6, floatmode='maxprec')}"

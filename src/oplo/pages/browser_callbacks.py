from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.express as px
from dash import Input, Output, State, html, no_update
from dash import dash_table

from oplo.config import get_base_dir, safe_join, ALLOWED_EXTS
from oplo.io.image_io import load_image, ImageBundle
from oplo.state import STORE

def _list_subdirs(base: Path) -> List[str]:
    subs = ["."]
    for root, dirs, _files in os.walk(base):
        rel = Path(root).relative_to(base)
        if len(rel.parts) <= 2:  # keep dropdown short
            for d in dirs:
                subs.append(str(rel / d))
    return sorted(set(subs))

def _list_files(dirpath: Path) -> List[Tuple[str, int]]:
    items = []
    for name in sorted(os.listdir(dirpath)):
        p = dirpath / name
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            try:
                size = p.stat().st_size
            except Exception:
                size = -1
            items.append((name, size))
    return items

def _fmt_bytes(n: int) -> str:
    if n < 0: return "?"
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0: return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def register(app):
    @app.callback(
        Output("dir-select", "options"),
        Output("dir-select", "value"),
        Output("browse-alert", "children"),
        Output("browse-alert", "is_open"),
        Input("dir-rescan", "n_clicks"),
        prevent_initial_call=False,
    )
    def scan_dirs(_n):
        base = get_base_dir()
        opts = [{"label": s, "value": s} for s in _list_subdirs(base)]
        alert = html.Span(["Base dir: ", html.Code(str(base))])
        return opts, ".", alert, True

    @app.callback(
        Output("files-table", "children"),
        Input("dir-select", "value"),
        prevent_initial_call=False,
    )
    def list_files(rel_subdir):
        base = get_base_dir()
        try:
            target = safe_join(base, rel_subdir or ".")
        except Exception as e:
            return html.Div(f"Invalid path: {e}")
        files = _list_files(target)
        rows = [{
            "name": name,
            "size": _fmt_bytes(size),
            "path": str(Path(rel_subdir or ".") / name),
        } for name, size in files]
        table = dash_table.DataTable(
            id="files-dt",
            columns=[
                {"name": "File", "id": "name"},
                {"name": "Size", "id": "size"},
                {"name": "RelPath", "id": "path"},
            ],
            data=rows,
            sort_action="native",
            filter_action="native",
            row_selectable="single",
            page_size=12,
            style_table={"height": "60vh", "overflowY": "auto"},
            style_cell={"fontSize": 13},
            style_as_list_view=True,
        )
        return table

    @app.callback(
        Output("selected-file", "data"),
        Input("files-dt", "selected_rows"),
        State("files-dt", "data"),
        prevent_initial_call=True,
    )
    def remember_selection(sel, data):
        if not sel: return no_update
        return data[sel[0]]["path"]

    @app.callback(
        Output("file-meta", "children"),
        Output("file-preview", "figure"),
        Input("selected-file", "data"),
        State("dir-select", "value"),
        prevent_initial_call=True,
    )
    def preview_file(relpath, _subdir):
        if not relpath:
            return no_update, no_update
        base = get_base_dir()
        p = safe_join(base, relpath)
        bundle: ImageBundle = load_image(str(p))
        img = bundle.view(policy="percentile", lo=1.0, hi=99.0, gamma=1.0)

        # downscale preview
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > 768:
            import cv2
            scale = 768 / float(longest)
            img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

        # grayscale vs rgb
        if img.ndim == 3 and img.shape[2] == 1:
            img = np.squeeze(img, 2)
        if img.ndim == 2:
            fig = px.imshow(img, origin="upper", zmin=0.0, zmax=1.0, color_continuous_scale="gray")
            fig.update_layout(coloraxis_showscale=False)
        else:
            fig = px.imshow(img, origin="upper", zmin=0.0, zmax=1.0)
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))

        meta = bundle.meta
        h0, w0 = bundle.data.shape[:2]
        ch = 1 if bundle.data.ndim == 2 else bundle.data.shape[2]
        meta_div = html.Div([
            html.Div([html.B("Path: "), html.Code(str(p))]),
            html.Div(f"Size: {w0} × {h0} × {ch}"),
            html.Div(f"Dtype: {meta.get('orig_dtype')}  Bit depth: {meta.get('bit_depth')}"),
        ])
        return meta_div, fig

    @app.callback(
        Output("browse-nav", "pathname"),
        Input("open-in-viewer", "n_clicks"),
        State("selected-file", "data"),
        prevent_initial_call=True,
    )
    def open_in_viewer(n, relpath):
        if not n or not relpath:
            return no_update
        base = get_base_dir()
        p = safe_join(base, relpath)
        bundle = load_image(str(p))
        STORE.set(bundle=bundle, disp=None)
        return "/"

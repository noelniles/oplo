# src/oplo/pages/viewer_callbacks.py
from __future__ import annotations

import base64, os, tempfile
from typing import NamedTuple, Dict

import numpy as np
import cv2
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Input, Output, State, no_update, html, ctx, dcc, ALL, MATCH  # ctx to detect trigger

from oplo.io.image_io import load_image, ImageBundle
from oplo.state import STORE
from oplo.pipeline import process_bundle
from oplo.processor_registry import processor_registry
import oplo.processors  # Import to trigger registration

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

    def _render_bundle_view(bundle: ImageBundle, slice_idx: int, policy: str, gamma: float, cmap: str, showbar, roi_on, crosshair_on, filename=None):
        """Render view-related artifacts for a bundle and a given slice index.

        Returns: (fig, meta_ul, hist_fig, note, disp_array)
        """
        # choose slice or whole image for viewing
        if bundle.is_stack():
            n = bundle.num_slices()
            si = max(0, min(n - 1, int(slice_idx or 0)))
            img_view = bundle.view_slice(si, policy=policy, gamma=gamma)
        else:
            img_view = bundle.view(policy=policy, lo=0.5, hi=99.5, gamma=gamma)

        # Downsample for display
        disp = _maybe_downsample_for_display(img_view, DISPLAY_MAX)
        img_disp = disp.disp
        if img_disp.ndim == 3 and img_disp.shape[2] == 1:
            img_disp = np.squeeze(img_disp, axis=2)

        is_single = img_disp.ndim == 2
        is_rgb = (img_disp.ndim == 3 and img_disp.shape[2] in (3, 4))
        cmap_name = _COLORMAPS.get(cmap or "gray", "gray")

        if is_single:
            fig = px.imshow(img_disp, origin="upper", zmin=0.0, zmax=1.0, color_continuous_scale=cmap_name)
            fig.update_layout(coloraxis_showscale=bool(showbar))
        elif is_rgb:
            fig = px.imshow(img_disp, origin="upper", zmin=0.0, zmax=1.0)
        else:
            fig = px.imshow(img_disp, origin="upper", zmin=0.0, zmax=1.0, color_continuous_scale=cmap_name)

        # spikes / crosshair styling
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        show_spikes = bool(crosshair_on)
        spike_style = dict(spikecolor="rgba(150,150,150,0.55)", spikethickness=1, spikemode="across")
        fig.update_xaxes(showspikes=show_spikes, **spike_style)
        fig.update_yaxes(showspikes=show_spikes, **spike_style)
        fig.update_layout(dragmode=("select" if roi_on else "zoom"))

        # metadata summary
        meta = bundle.meta
        if bundle.is_stack():
            # use slice 0 for size reporting
            slice0 = bundle.get_slice(0)
            h, w = slice0.shape[:2]
            ch = 1 if slice0.ndim == 2 else slice0.shape[2]
        else:
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

        hist_src = img_disp if is_single else np.clip(img_disp, 0, 1).astype(np.float32)
        hfig = px.histogram(hist_src.ravel(), nbins=256)
        hfig.update_layout(margin=dict(l=0, r=0, t=0, b=0), bargap=0)

        note = None
        if disp.scale < 1.0:
            note = html.Div(f"Downsampled for view: scale={disp.scale:.3f} (max {DISPLAY_MAX}px)")

        return fig, meta_ul, hfig, note, disp.disp

    # ----------------- Processing / pipeline callbacks -----------------

    @app.callback(
        Output("proc-category", "options"),
        Input("process-interval", "n_intervals"),  # Just to trigger once
    )
    def populate_categories(n):
        """Populate category dropdown from registry."""
        categories = processor_registry.get_categories()
        return [{"label": cat, "value": cat} for cat in categories]


    @app.callback(
        Output("proc-select", "options"),
        Input("proc-search", "value"),
        Input("proc-category", "value"),
    )
    def filter_processors(search, category):
        """Filter processors by search query and category."""
        if search:
            infos = processor_registry.search(search)
        else:
            infos = processor_registry.list_all()
        
        if category:
            infos = [info for info in infos if info.category == category]
        
        return [
            {"label": f"{info.name} - {info.description}", "value": info.name}
            for info in infos
        ]


    @app.callback(
        Output("proc-description", "children"),
        Output("proc-params", "children"),
        Input("proc-select", "value"),
    )
    def show_processor_info(proc_name):
        """Show processor description and generate parameter inputs."""
        if not proc_name:
            return "", []
        
        info = processor_registry.get(proc_name)
        if not info:
            return "Processor not found", []
        
        desc = f"{info.category} > {info.description}"
        
        # Generate parameter inputs dynamically
        param_inputs = []
        for param in info.params:
            if param.type == "float":
                inp = dcc.Input(
                    id={"type": "proc-param", "name": param.name},
                    type="number",
                    value=param.default,
                    step=0.1,
                    min=param.min,
                    max=param.max,
                    style={"width": "100px"}
                )
            elif param.type == "int":
                inp = dcc.Input(
                    id={"type": "proc-param", "name": param.name},
                    type="number",
                    value=param.default,
                    step=1,
                    min=param.min,
                    max=param.max,
                    style={"width": "100px"}
                )
            elif param.type == "bool":
                inp = dcc.Checklist(
                    id={"type": "proc-param", "name": param.name},
                    options=[{"label": "", "value": "true"}],
                    value=["true"] if param.default else []
                )
            else:
                inp = dcc.Input(
                    id={"type": "proc-param", "name": param.name},
                    type="text",
                    value=str(param.default),
                    style={"width": "150px"}
                )
            
            param_inputs.append(html.Div([
                html.Label(f"{param.name}: {param.description}"),
                inp
            ], style={"marginBottom": "8px"}))
        
        return desc, param_inputs


    @app.callback(
        Output("pipeline-store", "data"),
        Input("add-node", "n_clicks"),
        State("proc-select", "value"),
        State({"type": "proc-param", "name": ALL}, "value"),
        State("tile-h", "value"),
        State("tile-w", "value"),
        State("overlap-h", "value"),
        State("overlap-w", "value"),
        State("workers", "value"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def add_node(n_clicks, proc, param_values, tile_h, tile_w, overlap_h, overlap_w, workers, pipeline_data):
        # Append a node to the JSON-serializable pipeline store
        if not proc:
            return pipeline_data or []
        
        pipeline = list(pipeline_data or [])
        
        # Build params dict from dynamic inputs
        info = processor_registry.get(proc)
        params = {}
        if info and param_values:
            for i, param in enumerate(info.params):
                if i < len(param_values):
                    val = param_values[i]
                    if param.type == "float":
                        params[param.name] = float(val) if val is not None else param.default
                    elif param.type == "int":
                        params[param.name] = int(val) if val is not None else param.default
                    elif param.type == "bool":
                        params[param.name] = bool(val and len(val) > 0)
                    else:
                        params[param.name] = val if val is not None else param.default
        
        node = {
            "id": str(uuid.uuid4()),
            "op": proc,
            "params": params,
            "tile_size": [int(tile_h or 512), int(tile_w or 512)],
            "overlap": [int(overlap_h or 0), int(overlap_w or 0)],
            "workers": int(workers or 1),
            "input": "source",
        }
        pipeline.append(node)
        return pipeline


    @app.callback(
        Output("pipeline-nodes", "children"),
        Input("pipeline-store", "data"),
        Input("selected-node", "data"),
    )
    def render_pipeline_nodes(pipeline_data, selected_idx):
        if not pipeline_data:
            return "(no processing nodes)"
        items = []
        for i, n in enumerate(pipeline_data):
            is_selected = (selected_idx == i)
            card_style = {
                "cursor": "pointer",
                "marginBottom": "8px",
                "border": "2px solid #007bff" if is_selected else "1px solid #ddd",
                "backgroundColor": "#e7f3ff" if is_selected else "white",
            }
            items.append(
                dbc.Card([
                    dbc.CardBody([
                        html.H6(f"#{i+1}: {n.get('op')}", className="mb-1"),
                        html.Small(f"Tile: {n.get('tile_size')}, Overlap: {n.get('overlap')}", className="text-muted d-block"),
                        html.Small(f"Params: {n.get('params')}", className="text-muted"),
                    ], style={"padding": "8px"})
                ], id={"type": "pipeline-node-card", "index": i}, style=card_style)
            )
        return items


    @app.callback(
        Output("selected-node", "data"),
        Input({"type": "pipeline-node-card", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def select_pipeline_node(n_clicks_list):
        if not ctx.triggered:
            return no_update
        # Get the index of the clicked card
        triggered_id = ctx.triggered_id
        if triggered_id and isinstance(triggered_id, dict):
            return triggered_id["index"]
        return no_update


    @app.callback(
        Output("node-editor", "children"),
        Input("selected-node", "data"),
        State("pipeline-store", "data"),
    )
    def show_node_editor(selected_idx, pipeline_data):
        if selected_idx is None or not pipeline_data:
            return []
        
        if selected_idx >= len(pipeline_data):
            return []
        
        node = pipeline_data[selected_idx]
        op_name = node.get("op")
        
        # Get processor info
        try:
            proc_info = processor_registry.get_info(op_name)
        except Exception:
            return html.Div(f"Unknown processor: {op_name}", className="text-danger")
        
        # Build parameter inputs
        param_inputs = []
        current_params = node.get("params", {})
        
        for param in proc_info.params:
            param_id = {"type": "edit-param", "node": selected_idx, "param": param.name}
            current_value = current_params.get(param.name, param.default)
            
            if param.type == "int":
                input_el = dcc.Input(
                    id=param_id,
                    type="number",
                    value=current_value,
                    style={"width": "100px"}
                )
            elif param.type == "float":
                input_el = dcc.Input(
                    id=param_id,
                    type="number",
                    value=current_value,
                    step=0.1,
                    style={"width": "100px"}
                )
            elif param.type == "bool":
                input_el = dcc.Checklist(
                    id=param_id,
                    options=[{"label": "", "value": "true"}],
                    value=["true"] if current_value else []
                )
            else:  # str or other
                input_el = dcc.Input(
                    id=param_id,
                    type="text",
                    value=str(current_value),
                    style={"width": "100px"}
                )
            
            param_inputs.append(
                html.Div([
                    html.Label(f"{param.name}:", style={"marginRight": "8px", "fontWeight": "bold"}),
                    input_el,
                    html.Small(f" ({param.description})", className="text-muted"),
                ], style={"marginBottom": "8px"})
            )
        
        return dbc.Card([
            dbc.CardHeader(f"Edit Node #{selected_idx + 1}: {op_name}"),
            dbc.CardBody([
                html.Div(param_inputs),
                html.Div([
                    html.Label("Tile Size:", style={"marginRight": "8px"}),
                    dcc.Input(
                        id={"type": "edit-tile-size", "node": selected_idx},
                        type="number",
                        value=node.get("tile_size", 512),
                        style={"width": "100px"}
                    ),
                ], style={"marginTop": "12px"}),
                html.Div([
                    html.Label("Overlap:", style={"marginRight": "8px"}),
                    dcc.Input(
                        id={"type": "edit-overlap", "node": selected_idx},
                        type="number",
                        value=node.get("overlap", 32),
                        style={"width": "100px"}
                    ),
                ], style={"marginTop": "8px"}),
                html.Div([
                    dbc.Button("Delete Node", id={"type": "delete-node", "node": selected_idx}, 
                              color="danger", size="sm", className="mt-3"),
                ]),
            ])
        ], className="mt-2")


    def _lookup_processor(name: str):
        try:
            return processor_registry.get_func(name)
        except Exception:
            return None


    # Use ALL to capture all edits, then determine which triggered
    @app.callback(
        Output("node-update", "data"),
        Input({"type": "edit-param", "node": ALL, "param": ALL}, "value"),
        Input({"type": "edit-tile-size", "node": ALL}, "value"),
        Input({"type": "edit-overlap", "node": ALL}, "value"),
        Input({"type": "delete-node", "node": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def capture_node_edit(param_values, tile_sizes, overlaps, delete_clicks):
        triggered = ctx.triggered_id
        if not triggered or not isinstance(triggered, dict):
            return no_update
        
        node_idx = triggered.get("node")
        edit_type = triggered.get("type")
        
        update = {
            "node_idx": node_idx,
            "type": edit_type,
            "timestamp": time.time(),
        }
        
        if edit_type == "edit-param":
            param_name = triggered.get("param")
            # Find the index of this param in the list
            triggered_list = [d for d in ctx.inputs_list[0] if d['id'].get('node') == node_idx and d['id'].get('param') == param_name]
            if triggered_list:
                value = triggered_list[0]['value']
                update["param_name"] = param_name
                update["param_value"] = value
        elif edit_type == "edit-tile-size":
            # Find which tile_size changed
            triggered_list = [d for d in ctx.inputs_list[1] if d['id'].get('node') == node_idx]
            if triggered_list:
                update["tile_size"] = triggered_list[0]['value']
        elif edit_type == "edit-overlap":
            # Find which overlap changed
            triggered_list = [d for d in ctx.inputs_list[2] if d['id'].get('node') == node_idx]
            if triggered_list:
                update["overlap"] = triggered_list[0]['value']
        elif edit_type == "delete-node":
            update["delete"] = True
        
        return update


    # Non-pattern-matched callback to apply the update
    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Output("selected-node", "data", allow_duplicate=True),
        Input("node-update", "data"),
        State("pipeline-store", "data"),
        State("selected-node", "data"),
        prevent_initial_call=True,
    )
    def apply_node_update(update, pipeline_data, selected_idx):
        if not update or not pipeline_data:
            return no_update, no_update
        
        node_idx = update.get("node_idx")
        edit_type = update.get("type")
        
        # Handle delete
        if edit_type == "delete-node":
            new_pipeline = [n for i, n in enumerate(pipeline_data) if i != node_idx]
            return new_pipeline, None
        
        if node_idx >= len(pipeline_data):
            return no_update, no_update
        
        node = pipeline_data[node_idx].copy()
        
        # Apply the update
        if edit_type == "edit-param":
            param_name = update.get("param_name")
            value = update.get("param_value")
            
            # Get processor info to convert type
            op_name = node.get("op")
            try:
                proc_info = processor_registry.get_info(op_name)
                param_info = next((p for p in proc_info.params if p.name == param_name), None)
                
                if param_info:
                    # Convert value based on type
                    if param_info.type == "bool":
                        value = bool(value and "true" in value) if isinstance(value, list) else bool(value)
                    elif param_info.type == "int":
                        value = int(value) if value is not None else param_info.default
                    elif param_info.type == "float":
                        value = float(value) if value is not None else param_info.default
                    
                    params = node.get("params", {}).copy()
                    params[param_name] = value
                    node["params"] = params
            except Exception:
                pass
        elif edit_type == "edit-tile-size":
            tile_size = update.get("tile_size")
            if tile_size is not None:
                node["tile_size"] = int(tile_size)
        elif edit_type == "edit-overlap":
            overlap = update.get("overlap")
            if overlap is not None:
                node["overlap"] = int(overlap)
        
        # Update pipeline
        new_pipeline = pipeline_data.copy()
        new_pipeline[node_idx] = node
        
        return new_pipeline, node_idx


    @app.callback(
        Output("process-status", "children"),
        Output("process-interval", "disabled", allow_duplicate=True),
        Input("run-pipeline", "n_clicks"),
        State("pipeline-store", "data"),
        prevent_initial_call=True,
    )
    def run_pipeline(n_clicks, pipeline_data):
        if not pipeline_data:
            return "No pipeline defined", True  # Keep interval disabled

        def worker(pipeline):
            # Evaluate nodes sequentially, updating STORE and progress.
            bundle: ImageBundle = STORE.get("bundle")
            if bundle is None:
                STORE.set(process_status="No image loaded")
                return

            total_nodes = len(pipeline)
            for idx, node in enumerate(pipeline):
                op_name = node.get("op")
                func = _lookup_processor(op_name)
                if func is None:
                    # List available processors for debugging
                    available = [info.name for info in processor_registry.list_all()]
                    STORE.set(process_status=f"Unknown processor '{op_name}'. Available: {', '.join(available)}")
                    return
                if not callable(func):
                    STORE.set(process_status=f"Processor '{op_name}' is not callable (type: {type(func).__name__})")
                    return

                # progress per-node will be coarse; process_bundle accepts a progress_cb
                def progress_cb(done, total):
                    # compute overall percent: node index + fractional
                    try:
                        frac = (idx + (done / float(total))) / float(total_nodes)
                    except Exception:
                        frac = float(idx) / float(total_nodes)
                    STORE.set(process_progress=float(frac))

                STORE.set(process_status=f"Running node {idx+1}/{total_nodes}: {op_name}")
                try:
                    # Validate parameters before calling
                    tile_size_val = node.get("tile_size", (512, 512))
                    if not tile_size_val:
                        tile_size_val = (512, 512)
                    overlap_val = node.get("overlap", (0, 0))
                    if not overlap_val:
                        overlap_val = (0, 0)
                    
                    processed = process_bundle(
                        bundle,
                        func,
                        tile_size=tuple(tile_size_val),
                        overlap=tuple(overlap_val),
                        workers=node.get("workers", None),
                        progress_cb=progress_cb,
                        executor_class=None,  # let process_bundle choose default
                        func_kwargs=node.get("params") or {},
                    )
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    STORE.set(process_status=f"Node failed: {e}\n{tb}")
                    return

                # update bundle and add to history
                bundle = processed
                STORE.push_history(bundle, f"Node {idx+1}: {op_name}")
                # short delay to ensure UI poll sees bundle change
                time.sleep(0.05)

            # finished
            STORE.set(process_progress=1.0)
            STORE.set(process_status="Processing complete")

        # Initialize progress tracking
        STORE.set(process_progress=0.0)
        STORE.set(process_status="Starting...")
        
        th = threading.Thread(target=worker, args=(list(pipeline_data),), daemon=True)
        th.start()
        return "Processing started", False  # Enable interval while processing


    @app.callback(
        Output("process-status", "children", allow_duplicate=True),
        Output("process-interval", "disabled", allow_duplicate=True),
        Input("pipeline-store", "data"),
        State("selected-node", "data"),
        prevent_initial_call=True,
    )
    def instant_preview(pipeline_data, selected_idx):
        """Auto re-run pipeline from edited node onwards for instant preview"""
        # Only trigger if the change came from editing (not from add/clear)
        if not ctx.triggered or not pipeline_data:
            return no_update, no_update
        
        # Check if triggered by node editing (has selected node)
        triggered_prop = ctx.triggered[0]["prop_id"]
        if "pipeline-store" not in triggered_prop:
            return no_update, no_update
        
        # Start from the edited node if available, otherwise from beginning
        start_from = selected_idx if selected_idx is not None else 0
        
        def preview_worker(pipeline, start_idx):
            bundle: ImageBundle = STORE.get("bundle")
            if bundle is None:
                return
            
            # If starting from middle, get the last history state before this node
            if start_idx > 0:
                hist_len = STORE.get_history_length()
                if hist_len > start_idx:
                    STORE.set_history_index(start_idx)
                    bundle = STORE.get("bundle")
                    if bundle is None:
                        return
            
            total_nodes = len(pipeline)
            for idx in range(start_idx, total_nodes):
                node = pipeline[idx]
                op_name = node.get("op")
                func = _lookup_processor(op_name)
                if func is None:
                    STORE.set(process_status=f"Unknown processor: {op_name}")
                    return
                
                frac = float(idx) / float(total_nodes) if total_nodes > 0 else 0.0
                STORE.set(process_progress=frac)
                STORE.set(process_status=f"Preview: {idx+1}/{total_nodes} {op_name}")
                
                try:
                    processed = process_bundle(
                        bundle,
                        func,
                        tile_size=tuple(node.get("tile_size", (512, 512))),
                        overlap=tuple(node.get("overlap", (0, 0))),
                        workers=node.get("workers", None),
                        progress_cb=None,  # No sub-progress for instant preview
                        executor_class=None,
                        func_kwargs=node.get("params") or {},
                    )
                    bundle = processed
                    STORE.set(bundle=bundle)
                    STORE.push_history(bundle, f"Preview {op_name}")
                    time.sleep(0.05)
                except Exception as e:
                    STORE.set(process_status=f"Preview failed: {e}")
                    return
            
            STORE.set(process_progress=1.0)
            STORE.set(process_status="Preview complete")
        
        STORE.set(process_progress=0.0)
        STORE.set(process_status="Generating preview...")
        
        th = threading.Thread(target=preview_worker, args=(list(pipeline_data), start_from), daemon=True)
        th.start()
        return "Preview started", False  # Enable interval


    @app.callback(
        Output("process-progress", "value"),
        Output("process-status", "children", allow_duplicate=True),
        Output("process-interval", "disabled"),
        Output("image-view", "figure", allow_duplicate=True),
        Output("meta", "children", allow_duplicate=True),
        Output("hist-view", "figure", allow_duplicate=True),
        Output("downsample-note", "children", allow_duplicate=True),
        Output("image-view", "selectedData", allow_duplicate=True),
        Output("slice-index", "min", allow_duplicate=True),
        Output("slice-index", "max", allow_duplicate=True),
        Output("slice-index", "value", allow_duplicate=True),
        Output("slice-index", "marks", allow_duplicate=True),
        Output("slice-container", "style", allow_duplicate=True),
        Output("slice-info", "children", allow_duplicate=True),
        Input("process-interval", "n_intervals"),
        State("gamma", "value"),
        State("auto-scale", "value"),
        State("colormap", "value"),
        State("show-colorbar", "value"),
        State("roi-mode", "value"),
        State("crosshair", "value"),
        prevent_initial_call=True,
    )
    def poll_progress(n, gamma, autos, cmap, showbar, roi_on, crosshair_on):
        # read progress and status from STORE and optionally render new bundle
        prog = STORE.get("process_progress")
        status = STORE.get("process_status") or ""
        if prog is None:
            prog_val = 0
        else:
            try:
                prog_val = int(max(0.0, min(1.0, float(prog))) * 100.0)
            except Exception:
                prog_val = 0

        # default no_update for figure outputs
        fig_out = no_update
        meta_out = no_update
        hist_out = no_update
        note_out = no_update
        sel_reset = no_update
        smin = no_update
        smax = no_update
        svalue = no_update
        marks = no_update
        sstyle = no_update
        sinfo = no_update

        bundle: ImageBundle = STORE.get("bundle")
        last = STORE.get("_last_broadcast")
        if bundle is not None and id(bundle) != last:
            # render first slice / full image
            policy = "percentile" if ("auto" in (autos or [])) else "dtype_range"
            gamma = gamma or 1.0
            try:
                fig, meta_ul, hfig, note, disp_arr = _render_bundle_view(
                    bundle, 0, policy, gamma, cmap, showbar, roi_on, crosshair_on
                )
                STORE.set(disp=disp_arr)
                STORE.set(_last_broadcast=id(bundle))
                fig_out = fig
                meta_out = meta_ul
                hist_out = hfig
                note_out = note
                sel_reset = None

                if bundle.is_stack():
                    n = bundle.num_slices()
                    smin, smax, svalue = 0, max(0, n - 1), 0
                    if n <= 10:
                        marks = {i: str(i) for i in range(n)}
                    else:
                        marks = {0: "0", smax: str(smax)}
                    sstyle = {"display": "block"}
                    sinfo = f"Stack slices: {n}"
                else:
                    smin, smax, svalue, marks = 0, 0, 0, {}
                    sstyle = {"display": "none"}
                    sinfo = ""
            except Exception:
                # ignore rendering errors during polling
                pass

        # Disable interval when processing is complete
        interval_disabled = ("complete" in status.lower()) or (prog_val >= 100)
        
        return prog_val, status or "", interval_disabled, fig_out, meta_out, hist_out, note_out, sel_reset, smin, smax, svalue, marks, sstyle, sinfo


    @app.callback(
        Output("image-view", "figure", allow_duplicate=True),
        Output("meta", "children"),
        Output("hist-view", "figure", allow_duplicate=True),
        Output("downsample-note", "children", allow_duplicate=True),
        Output("image-view", "selectedData"),      # NEW: clear ROI on new image
        # slice slider configuration outputs
        Output("slice-index", "min"),
        Output("slice-index", "max"),
        Output("slice-index", "value"),
        Output("slice-index", "marks"),
        Output("slice-container", "style"),
        Output("slice-info", "children"),
        Input("upload-image", "contents"),
        State("upload-image", "filename"),
        Input("gamma", "value"),
        Input("auto-scale", "value"),
        Input("colormap", "value"),
        Input("show-colorbar", "value"),
        Input("roi-mode", "value"),               # ensure dragmode is preserved
        Input("crosshair", "value"),              # toggle cursor crosshair
        prevent_initial_call=True,
    )
    def update_image(contents, filename, gamma, autos, cmap, showbar, roi_on, crosshair_on):
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

        # Use helper to render first slice / full image and configure slider
        policy = "percentile" if ("auto" in (autos or [])) else "dtype_range"
        gamma = gamma or 1.0

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

        # Configure slice slider UI based on bundle
        if bundle.is_stack():
            n = bundle.num_slices()
            smin, smax, svalue = 0, max(0, n - 1), 0
            if n <= 10:
                marks = {i: str(i) for i in range(n)}
            else:
                marks = {0: "0", smax: str(smax)}
            slice_style = {"display": "block"}
            slice_info = f"Stack slices: {n}"
        else:
            smin, smax, svalue, marks = 0, 0, 0, {}
            slice_style = {"display": "none"}
            slice_info = ""

        return fig, meta_ul, hfig, note, selected_reset, smin, smax, svalue, marks, slice_style, slice_info


    @app.callback(
        Output("image-view", "figure", allow_duplicate=True),
        Output("roi-stats", "children"),
        Output("roi-hist", "figure"),
        Output("roi-hint", "children"),
        Input("image-view", "selectedData"),
        State("image-view", "figure"),
        prevent_initial_call=True,
    )
    def roi_stats(selected, fig_state):
        bundle: ImageBundle = STORE.get("bundle")
        disp = STORE.get("disp")
        scale = STORE.get("scale", 1.0)
        if bundle is None or disp is None or not selected:
            return no_update, no_update, "Enable ROI mode and drag a selection on the image."

        rng = selected.get("range")
        if not rng or "x" not in rng or "y" not in rng:
            return no_update, no_update, no_update, "Use box select to get ROI stats."

        x0, x1 = rng["x"][0], rng["x"][1]
        y0, y1 = rng["y"][0], rng["y"][1]

        disp_h, disp_w = disp.shape[:2]
        # Choose the current image (slice) for ROI calculations
        cur_slice = STORE.get("slice_index") or 0
        if bundle.is_stack():
            img = bundle.get_slice(int(cur_slice))
        else:
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
            return no_update, "Empty selection.", no_update, no_update

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

        # Build a clear rectangle overlay to highlight the ROI on the displayed
        # figure. Use a thick red outline and transparent fill so it stands out.
        rect = dict(
            type="rect",
            xref="x",
            yref="y",
            x0=min(x0, x1),
            x1=max(x0, x1),
            y0=min(y0, y1),
            y1=max(y0, y1),
            line=dict(color="#00FF66", width=1),
            fillcolor="rgba(0,255,102,0.12)",
        )

        out_fig = no_update
        try:
            if fig_state is not None:
                # fig_state is a dict-like figure; clone and replace shapes
                fig = fig_state.copy()
                # ensure layout exists
                fig.setdefault("layout", {})
                fig["layout"]["shapes"] = [rect]
                out_fig = fig
        except Exception:
            out_fig = no_update

        return out_fig, stats_text, hist_fig, ""


    # Optional: reflect any shapes created/updated in the client relayout events.
    # Some Plotly interactions emit `relayoutData` during user gestures; if the
    # client provides shape info we can mirror it back so the selection rectangle
    # appears while dragging in some environments. This is best-effort and will
    # not break if `relayoutData` is empty or non-shape-related.
    @app.callback(
        Output("image-view", "figure", allow_duplicate=True),
        Input("image-view", "relayoutData"),
        State("image-view", "figure"),
        prevent_initial_call=True,
    )
    def _mirror_relayout(relayout, fig_state):
        if not relayout or fig_state is None:
            return no_update
        # If relayout contains shapes, try to adopt them. Otherwise ignore.
        shapes = relayout.get("shapes") if isinstance(relayout, dict) else None
        if not shapes:
            return no_update
        try:
            fig = fig_state.copy()
            fig.setdefault("layout", {})
            fig["layout"]["shapes"] = shapes
            return fig
        except Exception:
            return no_update


    @app.callback(
        Output("image-view", "figure", allow_duplicate=True),
        Output("hist-view", "figure", allow_duplicate=True),
        Output("downsample-note", "children", allow_duplicate=True),
        Input("slice-index", "value"),
        State("gamma", "value"),
        State("auto-scale", "value"),
        State("colormap", "value"),
        State("show-colorbar", "value"),
        State("roi-mode", "value"),
        State("crosshair", "value"),
        prevent_initial_call=True,
    )
    def _on_slice_change(slice_idx, gamma, autos, cmap, showbar, roi_on, crosshair_on):
        bundle: ImageBundle = STORE.get("bundle")
        if bundle is None:
            return no_update

        policy = "percentile" if ("auto" in (autos or [])) else "dtype_range"
        gamma = gamma or 1.0
        si = int(slice_idx or 0)
        fig, meta_ul, hfig, note, disp_arr = _render_bundle_view(
            bundle, si, policy, gamma, cmap, showbar, roi_on, crosshair_on
        )
        STORE.set(disp=disp_arr)
        STORE.set(slice_index=si)
        return fig, hfig, note


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

        # Choose the current image (slice) for hover readout
        cur_slice = STORE.get("slice_index") or 0
        if bundle.is_stack():
            img = bundle.get_slice(int(cur_slice))
        else:
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


    @app.callback(
        Output("save-status", "children"),
        Input("save-btn", "n_clicks"),
        State("save-filename", "value"),
        prevent_initial_call=True,
    )
    def save_image(n_clicks, filename):
        bundle: ImageBundle = STORE.get("bundle")
        if bundle is None:
            return "No image to save"
        
        if not filename:
            filename = "output.tif"
        
        # Ensure it has an extension
        from pathlib import Path
        p = Path(filename)
        if not p.suffix:
            filename = filename + ".tif"
        
        try:
            # Try to write TIFF if tifffile is available
            try:
                import tifffile as tiff
                tiff.imwrite(filename, bundle.data)
                return f"Saved to {filename}"
            except ImportError:
                # Fallback to numpy save
                import numpy as np
                np.save(filename + ".npy", bundle.data)
                return f"Saved to {filename}.npy (tifffile not available)"
        except Exception as e:
            return f"Save failed: {e}"


    @app.callback(
        Output("pipeline-store", "data", allow_duplicate=True),
        Input("clear-pipeline", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_pipeline(n_clicks):
        return []


    @app.callback(
        Output("history-info", "children"),
        Output("history-select", "options"),
        Input("process-interval", "n_intervals"),
    )
    def update_history_info(n):
        idx = STORE.get_history_index()
        total = STORE.get_history_length()
        labels = STORE.get_history_labels()
        
        if total == 0:
            return "No history", []
        
        info = f"Stage {idx + 1} of {total}"
        options = [{"label": f"{i+1}. {label}", "value": i} for i, label in enumerate(labels)]
        return info, options


    @app.callback(
        Output("image-view", "figure", allow_duplicate=True),
        Output("meta", "children", allow_duplicate=True),
        Output("hist-view", "figure", allow_duplicate=True),
        Output("downsample-note", "children", allow_duplicate=True),
        Output("image-view", "selectedData", allow_duplicate=True),
        Output("slice-index", "min", allow_duplicate=True),
        Output("slice-index", "max", allow_duplicate=True),
        Output("slice-index", "value", allow_duplicate=True),
        Output("slice-index", "marks", allow_duplicate=True),
        Output("slice-container", "style", allow_duplicate=True),
        Output("slice-info", "children", allow_duplicate=True),
        Input("history-prev", "n_clicks"),
        Input("history-next", "n_clicks"),
        Input("history-select", "value"),
        State("gamma", "value"),
        State("auto-scale", "value"),
        State("colormap", "value"),
        State("show-colorbar", "value"),
        State("roi-mode", "value"),
        State("crosshair", "value"),
        prevent_initial_call=True,
    )
    def navigate_history(prev_clicks, next_clicks, select_value, gamma, autos, cmap, showbar, roi_on, crosshair_on):
        triggered = ctx.triggered_id
        
        current_idx = STORE.get_history_index()
        total = STORE.get_history_length()
        
        if total == 0:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        new_idx = current_idx
        if triggered == "history-prev" and current_idx > 0:
            new_idx = current_idx - 1
        elif triggered == "history-next" and current_idx < total - 1:
            new_idx = current_idx + 1
        elif triggered == "history-select" and select_value is not None:
            new_idx = int(select_value)
        
        if new_idx != current_idx:
            STORE.set_history_index(new_idx)
        
        bundle = STORE.get("bundle")
        if bundle is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        # Render the bundle at the new history position
        policy = "percentile" if ("auto" in (autos or [])) else "dtype_range"
        gamma = gamma or 1.0
        
        fig, meta_ul, hfig, note, disp_arr = _render_bundle_view(
            bundle, 0, policy, gamma, cmap, showbar, roi_on, crosshair_on
        )
        STORE.set(disp=disp_arr)
        
        # Configure slice slider
        if bundle.is_stack():
            n = bundle.num_slices()
            smin, smax, svalue = 0, max(0, n - 1), 0
            if n <= 10:
                marks = {i: str(i) for i in range(n)}
            else:
                marks = {0: "0", smax: str(smax)}
            slice_style = {"display": "block"}
            slice_info = f"Stack slices: {n}"
        else:
            smin, smax, svalue, marks = 0, 0, 0, {}
            slice_style = {"display": "none"}
            slice_info = ""
        
        return fig, meta_ul, hfig, note, None, smin, smax, svalue, marks, slice_style, slice_info


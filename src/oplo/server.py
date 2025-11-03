import os
from pathlib import Path
import dash
import dash_bootstrap_components as dbc
from .router import layout
from oplo.pages.viewer_callbacks import register as register_viewer_callbacks
from oplo.pages.browser_callbacks import register as register_browser_callbacks


def build_app():
    app = dash.Dash(
        __name__,
        use_pages=True,
        pages_folder=str(Path(__file__).parent / "pages"),
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        serve_locally=True,
        title="Oplo",
        assets_folder="assets",
    )
    # Upload size limit (in bytes) -- affect dccUpload payload size.
    # Increase for large images or multi-page stacks.
    max_mb = int(os.getenv("OPLO_MAX_UPLOAD_MB", "4096"))
    app.server.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024

    # Optional: limit in-memory form parsing (Werkzeug). If unset, Werkzeug defaults apply.
    # app.server.config["MAX_FORM_MEMORY_SIZE"] = app.server.config["MAX_CONTENT_LENGTH"]
    app.layout = layout()

    register_viewer_callbacks(app)
    register_browser_callbacks(app)

    return app

def run_dev():
    app = build_app()
    app.run(debug=True, port=8050)

def run_prod():
    bind = os.getenv("OPLO_BIND", "127.0.0.1:8050") # e.g. 0.0.0.0:8050 for lab-wide.
    host, port = bind.split(":")
    app = build_app()
    app.run(host=host, port=int(port), debug=False)

app = build_app()
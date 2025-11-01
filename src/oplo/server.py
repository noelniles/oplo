import os
import dash
import dash_bootstrap_components as dbc
from .router import layout


def build_app():
    app = dash.Dash(
        __name__,
        use_pages=True,
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
    app.layout = layout
    return app

def dev():
    app = build_app()
    app.run_server(debug=True, port=8050)

def prod():
    bind = os.getenv("OPLO_BIND", "127.0.0.1:8050") # e.g. 0.0.0.0:8050 for lab-wide.
    host, port = bind.split(":")
    app = build_app()
    app.run_server(host=host, port=int(port), debug=False)

app = build_app()
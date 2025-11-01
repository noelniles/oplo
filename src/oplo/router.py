"""Dash page router for Oplo.

This file wires pages together and provides a top‑level layout for multi‑page navigation.
Right now we only have the viewer page, but this sets up the structure to grow.

Dash will discover pages in the `pages/` package automatically.
"""

from dash import html, dcc
import dash
import dash_bootstrap_components as dbc

# Import all Dash pages under src/oplo/pages via dash.register_page
# The viewer page is already registered automatically by its module.

def layout():  # called by server
    return dbc.Container([
        # Top navigation bar
        dbc.Navbar([
            dbc.NavbarBrand("Oplo", className="ms-2"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Viewer", href="/")),
                # Future pages:
                # dbc.NavItem(dbc.NavLink("Batch", href="/batch")),
                # dbc.NavItem(dbc.NavLink("Metadata", href="/meta")),
                # dbc.NavItem(dbc.NavLink("About", href="/about")),
            ], className="ms-auto", navbar=True),
        ], color="dark", dark=True, className="mb-3"),

        # Page content from Dash's built‑in router
        dash.page_container
    ], fluid=True)

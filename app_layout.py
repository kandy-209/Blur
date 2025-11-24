"""
Multi-Page Application Layout
==============================
Clean, organized multi-page structure for the trading dashboard.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

def create_navbar():
    """Create navigation bar for multi-page app."""
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    [
                        html.Span("📈", style={"fontSize": "24px", "marginRight": "10px"}),
                        "Futures Trading Terminal",
                    ],
                    href="/",
                    className="ms-2",
                    style={"color": "#7dd87d", "fontFamily": "'Share Tech Mono', monospace", "fontSize": "18px", "fontWeight": "600"},
                ),
                dbc.Nav(
                    [
                        dbc.NavLink("Dashboard", href="/", active="exact", style={"color": "#7dd87d", "marginRight": "15px"}),
                        dbc.NavLink("Paper Trading", href="/paper-trading", active="exact", style={"color": "#7dd87d", "marginRight": "15px"}),
                        dbc.NavLink("Analytics", href="/analytics", active="exact", style={"color": "#7dd87d", "marginRight": "15px"}),
                        dbc.NavLink("Settings", href="/settings", active="exact", style={"color": "#7dd87d"}),
                    ],
                    navbar=True,
                    className="ms-auto",
                ),
            ],
            fluid=True,
        ),
        dark=True,
        color="dark",
        style={
            "backgroundColor": "#0a0a0a",
            "borderBottom": "1px solid rgba(125, 216, 125, 0.3)",
            "padding": "10px 0",
        },
    )

def create_sidebar():
    """Create sidebar navigation (alternative to navbar)."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("📈", style={"fontSize": "28px", "marginRight": "10px"}),
                            html.Div(
                                [
                                    html.Div("FUTURES", style={"fontSize": "14px", "color": "#7dd87d", "fontWeight": "600"}),
                                    html.Div("TRADING", style={"fontSize": "14px", "color": "#7dd87d", "fontWeight": "600"}),
                                ],
                                style={"display": "flex", "flexDirection": "column"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "padding": "20px",
                            "borderBottom": "1px solid rgba(125, 216, 125, 0.2)",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Link(
                                [
                                    html.Span("📊", style={"marginRight": "10px"}),
                                    "Dashboard",
                                ],
                                href="/",
                                style={
                                    "display": "block",
                                    "padding": "15px 20px",
                                    "color": "#7dd87d",
                                    "textDecoration": "none",
                                    "borderLeft": "3px solid transparent",
                                    "transition": "all 0.3s",
                                },
                                className="nav-link",
                            ),
                            dcc.Link(
                                [
                                    html.Span("💰", style={"marginRight": "10px"}),
                                    "Paper Trading",
                                ],
                                href="/paper-trading",
                                style={
                                    "display": "block",
                                    "padding": "15px 20px",
                                    "color": "#7dd87d",
                                    "textDecoration": "none",
                                    "borderLeft": "3px solid transparent",
                                    "transition": "all 0.3s",
                                },
                                className="nav-link",
                            ),
                            dcc.Link(
                                [
                                    html.Span("📈", style={"marginRight": "10px"}),
                                    "Analytics",
                                ],
                                href="/analytics",
                                style={
                                    "display": "block",
                                    "padding": "15px 20px",
                                    "color": "#7dd87d",
                                    "textDecoration": "none",
                                    "borderLeft": "3px solid transparent",
                                    "transition": "all 0.3s",
                                },
                                className="nav-link",
                            ),
                            dcc.Link(
                                [
                                    html.Span("⚙️", style={"marginRight": "10px"}),
                                    "Settings",
                                ],
                                href="/settings",
                                style={
                                    "display": "block",
                                    "padding": "15px 20px",
                                    "color": "#7dd87d",
                                    "textDecoration": "none",
                                    "borderLeft": "3px solid transparent",
                                    "transition": "all 0.3s",
                                },
                                className="nav-link",
                            ),
                        ],
                        style={"marginTop": "20px"},
                    ),
                ],
                style={
                    "height": "100vh",
                    "position": "fixed",
                    "left": 0,
                    "top": 0,
                    "width": "250px",
                    "backgroundColor": "#0f0f0f",
                    "borderRight": "1px solid rgba(125, 216, 125, 0.2)",
                    "overflowY": "auto",
                },
            ),
        ],
    )

def create_page_container(children, use_sidebar=True):
    """Create page container with sidebar or navbar."""
    if use_sidebar:
        return html.Div(
            [
                create_sidebar(),
                html.Div(
                    children,
                    style={
                        "marginLeft": "250px",
                        "padding": "20px",
                        "minHeight": "100vh",
                        "backgroundColor": "#0a0a0a",
                    },
                ),
            ],
            style={"display": "flex"},
        )
    else:
        return html.Div(
            [
                create_navbar(),
                html.Div(
                    children,
                    style={
                        "padding": "20px",
                        "minHeight": "calc(100vh - 60px)",
                        "backgroundColor": "#0a0a0a",
                    },
                ),
            ],
        )


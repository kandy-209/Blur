"""
Settings Page - Configuration and preferences
"""

from dash import html, dcc

def create_settings_page() -> html.Div:
    """Create the settings page."""
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Settings", style={"color": "#7dd87d", "margin": "0", "fontFamily": "'Share Tech Mono', monospace"}),
                    html.P("Configure your trading dashboard", style={"color": "rgba(125, 216, 125, 0.7)", "marginTop": "5px"}),
                ],
                style={"marginBottom": "30px", "paddingBottom": "15px", "borderBottom": "1px solid rgba(125, 216, 125, 0.3)"},
            ),
            
            html.Div(
                [
                    html.Div("DATA EXPORT & TOOLS", className="metrics-section-title"),
                    html.Div(
                        [
                            html.Button(
                                "Export CSV",
                                id="export-csv-btn",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "rgba(0, 191, 255, 0.2)",
                                    "border": "1px solid rgba(0, 191, 255, 0.5)",
                                    "color": "#00bfff",
                                    "padding": "10px 20px",
                                    "margin": "5px",
                                    "borderRadius": "6px",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Button(
                                "Export JSON",
                                id="export-json-btn",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "rgba(125, 216, 125, 0.2)",
                                    "border": "1px solid rgba(125, 216, 125, 0.5)",
                                    "color": "#7dd87d",
                                    "padding": "10px 20px",
                                    "margin": "5px",
                                    "borderRadius": "6px",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Div(id="export-status", style={"marginTop": "10px", "color": "#7dd87d", "fontSize": "12px"}),
                        ],
                        style={"textAlign": "center", "padding": "15px"},
                    ),
                ],
                className="metrics-section",
                style={
                    "background": "linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%)",
                    "border": "1px solid rgba(0, 191, 255, 0.4)",
                    "borderRadius": "6px",
                    "padding": "20px",
                    "marginBottom": "20px",
                },
            ),
            
            html.Div(id="alerts-container", style={"marginBottom": "20px"}),
            html.Div(id="portfolio-container", style={"marginBottom": "20px"}),
        ],
        style={"padding": "20px"},
    )


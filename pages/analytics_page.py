"""
Analytics Page - ML predictions, macro research, and advanced analytics
"""

from dash import html, dcc
from typing import Dict

def create_analytics_page(symbols: Dict = None) -> html.Div:
    """Create the analytics page."""
    if symbols is None:
        symbols = {"ES=F": "E-mini S&P 500", "NQ=F": "E-mini Nasdaq", "YM=F": "E-mini Dow", "RTY=F": "E-mini Russell"}
    
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Analytics & Research", style={"color": "#7dd87d", "margin": "0", "fontFamily": "'Share Tech Mono', monospace"}),
                    html.P("Advanced analytics, ML predictions, and macro research", style={"color": "rgba(125, 216, 125, 0.7)", "marginTop": "5px"}),
                ],
                style={"marginBottom": "30px", "paddingBottom": "15px", "borderBottom": "1px solid rgba(125, 216, 125, 0.3)"},
            ),
            
            # Controls needed for callbacks
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Symbol:", style={"color": "rgba(125, 216, 125, 0.8)", "marginRight": "10px", "width": "80px", "display": "inline-block"}),
                            dcc.Dropdown(
                                id="symbol-dropdown",
                                options=[{"label": f"{v} ({k})", "value": k} for k, v in symbols.items()],
                                value="ES=F",
                                clearable=False,
                                style={"width": "200px", "backgroundColor": "#1a1a1a", "display": "inline-block"},
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "20px"},
                    ),
                    html.Div(
                        [
                            html.Label("Interval:", style={"color": "rgba(125, 216, 125, 0.8)", "marginRight": "10px", "width": "80px", "display": "inline-block"}),
                            dcc.Dropdown(
                                id="interval-dropdown",
                                options=[
                                    {"label": "1 MIN", "value": "1m"},
                                    {"label": "5 MIN", "value": "5m"},
                                    {"label": "15 MIN", "value": "15m"},
                                    {"label": "1 HOUR", "value": "1h"},
                                    {"label": "1 DAY", "value": "1d"},
                                ],
                                value="1m",
                                clearable=False,
                                style={"width": "150px", "backgroundColor": "#1a1a1a", "display": "inline-block"},
                            ),
                        ],
                        style={"display": "inline-block", "marginRight": "20px"},
                    ),
                    html.Div(
                        [
                            html.Label("Chart:", style={"color": "rgba(125, 216, 125, 0.8)", "marginRight": "10px", "width": "80px", "display": "inline-block"}),
                            dcc.Dropdown(
                                id="chart-type-dropdown",
                                options=[
                                    {"label": "Candlestick", "value": "candlestick"},
                                    {"label": "Line", "value": "line"},
                                    {"label": "OHLC", "value": "ohlc"},
                                ],
                                value="candlestick",
                                clearable=False,
                                style={"width": "150px", "backgroundColor": "#1a1a1a", "display": "inline-block"},
                            ),
                        ],
                        style={"display": "inline-block"},
                    ),
                ],
                style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "rgba(15, 15, 15, 0.5)", "borderRadius": "6px"},
            ),
            
            html.Div(id="ml-predictions-container", style={"marginBottom": "20px"}),
            html.Div(id="macro-research-container", style={"marginBottom": "20px"}),
            html.Div(id="news-container", style={"marginBottom": "20px"}),
            html.Div(id="economic-indicators-container", style={"marginBottom": "20px"}),
        ],
        style={"padding": "20px"},
    )


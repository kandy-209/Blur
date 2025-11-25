"""
Dashboard Page - Main trading charts and indicators
"""

from dash import html, dcc
from typing import Dict

def create_dashboard_page(symbols: Dict) -> html.Div:
    """Create the main dashboard page with charts and indicators."""
    return html.Div(
        [
            # Page Header
            html.Div(
                [
                    html.H2("Trading Dashboard", style={"color": "#7dd87d", "margin": "0", "fontFamily": "'Share Tech Mono', monospace"}),
                    html.Div(
                        [
                            html.Span("SYSTEM ONLINE", id="live-indicator"),
                            html.Span(" ●", id="live-dot", style={"color": "#00ff41", "marginLeft": "10px"}),
                        ],
                        style={"color": "rgba(0, 191, 255, 0.8)", "fontSize": "12px", "marginTop": "5px"},
                    ),
                ],
                style={"marginBottom": "30px", "paddingBottom": "15px", "borderBottom": "1px solid rgba(125, 216, 125, 0.3)"},
            ),
            
            # Controls
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
            
            # Market Status Banner
            html.Div(id="market-status-banner", style={"marginBottom": "20px"}),
            
            # Trading Signal
            html.Div(id="trading-signal-container", style={"marginBottom": "20px"}),
            
            # Metrics
            html.Div(id="metrics-container", style={"marginBottom": "20px"}),
            
            # Main Chart
            html.Div(
                dcc.Graph(id="price-graph", style={"height": "500px"}),
                style={"marginBottom": "20px"},
            ),
            
            # Indicator Charts
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(id="rsi-graph", style={"height": "300px"}),
                            html.Div(id="rsi-education"),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="macd-graph", style={"height": "300px"}),
                            html.Div(id="macd-education"),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="stoch-graph", style={"height": "300px"}),
                            html.Div(id="stoch-education"),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(
                        dcc.Graph(id="volume-graph", style={"height": "250px"}),
                        style={"marginBottom": "20px"},
                    ),
                ],
            ),
            
            # Additional containers for ML, News, Economic, Macro (populated by callbacks)
            html.Div(id="ml-predictions-container", style={"marginBottom": "20px"}),
            html.Div(id="news-container", style={"marginBottom": "20px"}),
            html.Div(id="economic-indicators-container", style={"marginBottom": "20px"}),
            html.Div(id="macro-research-container", style={"marginBottom": "20px"}),
            
            # Note: Interval and stores are in main layout
        ],
        style={"padding": "20px"},
    )


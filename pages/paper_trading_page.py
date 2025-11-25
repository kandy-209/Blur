"""
Paper Trading Page - Practice trading interface
"""

from dash import html, dcc
from typing import Dict

def create_paper_trading_page(symbols: Dict = None) -> html.Div:
    """Create the paper trading page."""
    if symbols is None:
        symbols = {"ES=F": "E-mini S&P 500", "NQ=F": "E-mini Nasdaq", "YM=F": "E-mini Dow", "RTY=F": "E-mini Russell"}
    
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Paper Trading", style={"color": "#7dd87d", "margin": "0", "fontFamily": "'Share Tech Mono', monospace"}),
                    html.P("Practice trading with virtual money", style={"color": "rgba(125, 216, 125, 0.7)", "marginTop": "5px"}),
                ],
                style={"marginBottom": "30px", "paddingBottom": "15px", "borderBottom": "1px solid rgba(125, 216, 125, 0.3)"},
            ),
            
            # Symbol selector for paper trading
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
                style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "rgba(15, 15, 15, 0.5)", "borderRadius": "6px"},
            ),
            
            # Order form components (required by callback)
            html.Div(
                [
                    dcc.RadioItems(
                        id="order-side",
                        options=[
                            {"label": "BUY", "value": "BUY"},
                            {"label": "SELL", "value": "SELL"},
                        ],
                        value="BUY",
                        inline=True,
                        style={"display": "none"},  # Hidden, will be shown in the container
                    ),
                    dcc.Input(
                        id="order-quantity",
                        type="number",
                        value=1,
                        min=0.01,
                        step=0.01,
                        style={"display": "none"},  # Hidden, will be shown in the container
                    ),
                    dcc.Input(
                        id="order-price",
                        type="number",
                        value=None,
                        min=0,
                        step=0.01,
                        style={"display": "none"},  # Hidden, will be shown in the container
                    ),
                    html.Button(
                        "Place Order",
                        id="place-order-btn",
                        n_clicks=0,
                        style={"display": "none"},  # Hidden, will be shown in the container
                    ),
                ],
                style={"display": "none"},
            ),
            
            html.Div(id="paper-trading-container"),
        ],
        style={"padding": "20px"},
    )


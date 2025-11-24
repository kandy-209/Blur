"""
Paper Trading Page - Practice trading interface
"""

from dash import html, dcc

def create_paper_trading_page() -> html.Div:
    """Create the paper trading page."""
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Paper Trading", style={"color": "#7dd87d", "margin": "0", "fontFamily": "'Share Tech Mono', monospace"}),
                    html.P("Practice trading with virtual money", style={"color": "rgba(125, 216, 125, 0.7)", "marginTop": "5px"}),
                ],
                style={"marginBottom": "30px", "paddingBottom": "15px", "borderBottom": "1px solid rgba(125, 216, 125, 0.3)"},
            ),
            
            html.Div(id="paper-trading-container"),
            
            # Use the main interval from layout
            html.Div(id="paper-trading-symbol", style={"display": "none"}),
        ],
        style={"padding": "20px"},
    )


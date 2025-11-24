"""
Analytics Page - ML predictions, macro research, and advanced analytics
"""

from dash import html, dcc

def create_analytics_page() -> html.Div:
    """Create the analytics page."""
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Analytics & Research", style={"color": "#7dd87d", "margin": "0", "fontFamily": "'Share Tech Mono', monospace"}),
                    html.P("Advanced analytics, ML predictions, and macro research", style={"color": "rgba(125, 216, 125, 0.7)", "marginTop": "5px"}),
                ],
                style={"marginBottom": "30px", "paddingBottom": "15px", "borderBottom": "1px solid rgba(125, 216, 125, 0.3)"},
            ),
            
            html.Div(id="ml-predictions-container", style={"marginBottom": "20px"}),
            html.Div(id="macro-research-container", style={"marginBottom": "20px"}),
            html.Div(id="news-container", style={"marginBottom": "20px"}),
            html.Div(id="economic-indicators-container", style={"marginBottom": "20px"}),
            
            # Note: Uses main interval from layout
        ],
        style={"padding": "20px"},
    )


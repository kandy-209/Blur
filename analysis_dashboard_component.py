#!/usr/bin/env python3
"""
Dashboard Component for Displaying Daily Analysis Logs
Integrates historical analysis viewing into the main dashboard
"""

from dash import html, dcc
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict
from daily_analysis_logger import get_daily_analyses, get_all_analysis_dates

def create_analysis_history_component():
    """Create component for viewing analysis history."""
    return html.Div([
        html.Div([
            html.H3(
                "📊 Daily Analysis History",
                style={
                    "color": "#7dd87d",
                    "fontFamily": "'Share Tech Mono', monospace",
                    "fontSize": "20px",
                    "marginBottom": "20px",
                    "textAlign": "center",
                }
            ),
            html.Div([
                html.Label(
                    "Select Date:",
                    style={
                        "color": "rgba(125, 216, 125, 0.7)",
                        "fontWeight": "600",
                        "marginRight": "10px",
                        "fontSize": "13px",
                    }
                ),
                dcc.Dropdown(
                    id="analysis-date-dropdown",
                    options=[
                        {"label": date, "value": date}
                        for date in get_all_analysis_dates()[:30]  # Last 30 days
                    ],
                    value=get_all_analysis_dates()[0] if get_all_analysis_dates() else None,
                    style={
                        "width": "200px",
                        "backgroundColor": "#1a1a1a",
                        "color": "#7dd87d",
                    }
                ),
            ], style={"marginBottom": "20px", "textAlign": "center"}),
        ]),
        html.Div(id="analysis-content-container", style={"marginTop": "20px"}),
    ], style={
        "padding": "20px",
        "backgroundColor": "#0a0a0a",
        "border": "1px solid rgba(125, 216, 125, 0.4)",
        "borderRadius": "4px",
        "marginBottom": "20px",
    })

def format_analysis_for_display(analyses: List[Dict], selected_time_index: int = None) -> html.Div:
    """Format analysis data for dashboard display."""
    if not analyses:
        return html.Div([
            html.P(
                "No analysis available for this date.",
                style={"color": "rgba(125, 216, 125, 0.6)", "textAlign": "center"}
            )
        ])
    
    # Use most recent analysis if no index specified
    if selected_time_index is None:
        selected_time_index = len(analyses) - 1
    
    analysis = analyses[selected_time_index]
    
    # Create tabs for different analysis entries
    tabs = []
    for idx, anal in enumerate(analyses):
        time_str = anal.get("time_est", "Unknown")
        tabs.append({
            "label": f"{time_str}",
            "value": str(idx),
        })
    
    content = html.Div([
        # Analysis metadata
        html.Div([
            html.Div([
                html.Span("Date: ", style={"color": "rgba(125, 216, 125, 0.7)"}),
                html.Span(analysis.get("date", "Unknown"), style={"color": "#7dd87d"}),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Span("Time: ", style={"color": "rgba(125, 216, 125, 0.7)"}),
                html.Span(analysis.get("time_est", "Unknown"), style={"color": "#7dd87d"}),
            ], style={"marginRight": "20px"}),
            html.Div([
                html.Span("Type: ", style={"color": "rgba(125, 216, 125, 0.7)"}),
                html.Span(analysis.get("analysis_type", "UNKNOWN"), style={"color": "#7dd87d"}),
            ]),
        ], style={
            "display": "flex",
            "marginBottom": "20px",
            "padding": "10px",
            "backgroundColor": "#1a1a1a",
            "borderRadius": "4px",
        }),
        
        # Written analysis
        html.Div([
            html.H4(
                "Written Analysis",
                style={
                    "color": "#7dd87d",
                    "fontFamily": "'Share Tech Mono', monospace",
                    "marginBottom": "15px",
                }
            ),
            html.Div(
                format_markdown_text(analysis.get("written_analysis", "No analysis available.")),
                style={
                    "color": "rgba(125, 216, 125, 0.8)",
                    "lineHeight": "1.6",
                    "padding": "15px",
                    "backgroundColor": "#1a1a1a",
                    "borderRadius": "4px",
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "'Share Tech Mono', monospace",
                    "fontSize": "14px",
                }
            ),
        ], style={"marginBottom": "30px"}),
        
        # Market data summary
        html.Div([
            html.H4(
                "Market Data Summary",
                style={
                    "color": "#7dd87d",
                    "fontFamily": "'Share Tech Mono', monospace",
                    "marginBottom": "15px",
                }
            ),
            create_market_data_table(analysis.get("market_data", {})),
        ], style={"marginBottom": "30px"}),
    ])
    
    # Add conditional sections outside the main list
    if analysis.get("price_action_analysis"):
        content.children.append(
            html.Div([
                html.H4(
                    "Price Action Analysis",
                    style={
                        "color": "#7dd87d",
                        "fontFamily": "'Share Tech Mono', monospace",
                        "marginBottom": "15px",
                    }
                ),
                create_price_action_summary(analysis.get("price_action_analysis", {})),
            ], style={"marginBottom": "30px"})
        )
    
    # ORB Analysis Summary
    if analysis.get("orb_analysis") and "error" not in analysis.get("orb_analysis", {}):
        content.children.append(
            html.Div([
                html.H4(
                    "Opening Range Breakout (ORB) Analysis",
                    style={
                        "color": "#7dd87d",
                        "fontFamily": "'Share Tech Mono', monospace",
                        "marginBottom": "15px",
                    }
                ),
                create_orb_summary(analysis.get("orb_analysis", {})),
            ])
        )
    
    return content

def format_markdown_text(text: str) -> str:
    """Simple markdown formatting (basic implementation)."""
    # Convert markdown headers to HTML
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        if line.startswith('# '):
            formatted_lines.append(f'<h2 style="color: #7dd87d; margin-top: 15px; margin-bottom: 10px;">{line[2:]}</h2>')
        elif line.startswith('## '):
            formatted_lines.append(f'<h3 style="color: #7dd87d; margin-top: 12px; margin-bottom: 8px;">{line[3:]}</h3>')
        elif line.startswith('### '):
            formatted_lines.append(f'<h4 style="color: #7dd87d; margin-top: 10px; margin-bottom: 6px;">{line[4:]}</h4>')
        elif line.startswith('**') and line.endswith('**'):
            formatted_lines.append(f'<strong style="color: #7dd87d;">{line[2:-2]}</strong>')
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def create_market_data_table(market_data: Dict) -> html.Div:
    """Create table showing market data summary."""
    symbols_data = market_data.get("symbols", {})
    
    if not symbols_data:
        return html.P("No market data available.", style={"color": "rgba(125, 216, 125, 0.6)"})
    
    rows = []
    for symbol, data in symbols_data.items():
        if "error" not in data:
            change_pct = data.get("price_change_pct", 0)
            change_color = "#7dd87d" if change_pct > 0 else "#ff6b6b" if change_pct < 0 else "rgba(125, 216, 125, 0.6)"
            
            rows.append(html.Tr([
                html.Td(symbol, style={"padding": "8px", "color": "#7dd87d"}),
                html.Td(f"${data.get('current_price', 0):.2f}", style={"padding": "8px", "color": "rgba(125, 216, 125, 0.8)"}),
                html.Td(
                    f"${data.get('price_change', 0):+.2f} ({data.get('price_change_pct', 0):+.2f}%)",
                    style={"padding": "8px", "color": change_color}
                ),
                html.Td(f"${data.get('low', 0):.2f} - ${data.get('high', 0):.2f}", style={"padding": "8px", "color": "rgba(125, 216, 125, 0.8)"}),
                html.Td(f"{data.get('volume', 0):,}", style={"padding": "8px", "color": "rgba(125, 216, 125, 0.8)"}),
            ]))
    
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Symbol", style={"padding": "10px", "color": "#7dd87d", "textAlign": "left"}),
                html.Th("Price", style={"padding": "10px", "color": "#7dd87d", "textAlign": "left"}),
                html.Th("Change", style={"padding": "10px", "color": "#7dd87d", "textAlign": "left"}),
                html.Th("Range", style={"padding": "10px", "color": "#7dd87d", "textAlign": "left"}),
                html.Th("Volume", style={"padding": "10px", "color": "#7dd87d", "textAlign": "left"}),
            ])
        ]),
        html.Tbody(rows)
    ], style={
        "width": "100%",
        "borderCollapse": "collapse",
        "backgroundColor": "#1a1a1a",
        "borderRadius": "4px",
    })

def create_price_action_summary(price_action: Dict) -> html.Div:
    """Create summary of price action analysis."""
    trend_analysis = price_action.get("trend_analysis", {})
    support_resistance = price_action.get("support_resistance", {})
    
    return html.Div([
        html.Div([
            html.Span("Trend: ", style={"color": "rgba(125, 216, 125, 0.7)"}),
            html.Span(trend_analysis.get("trend", "UNKNOWN"), style={"color": "#7dd87d", "fontWeight": "bold"}),
        ], style={"marginBottom": "10px"}),
        html.Div([
            html.Span("Support Levels: ", style={"color": "rgba(125, 216, 125, 0.7)"}),
            html.Span(
                ", ".join([f"${s['price']:.2f}" for s in support_resistance.get("support_levels", [])[:3]]),
                style={"color": "#7dd87d"}
            ),
        ], style={"marginBottom": "10px"}),
        html.Div([
            html.Span("Resistance Levels: ", style={"color": "rgba(125, 216, 125, 0.7)"}),
            html.Span(
                ", ".join([f"${r['price']:.2f}" for r in support_resistance.get("resistance_levels", [])[:3]]),
                style={"color": "#7dd87d"}
            ),
        ]),
    ], style={
        "padding": "15px",
        "backgroundColor": "#1a1a1a",
        "borderRadius": "4px",
    })

def create_orb_summary(orb_data: Dict) -> html.Div:
    """Create summary of ORB analysis."""
    status = orb_data.get("breakout_status", "UNKNOWN")
    status_color = "#7dd87d" if status == "BROKEN_ABOVE" else "#ff6b6b" if status == "BROKEN_BELOW" else "rgba(125, 216, 125, 0.6)"
    
    return html.Div([
        html.Div([
            html.Span("ORB Range: ", style={"color": "rgba(125, 216, 125, 0.7)"}),
            html.Span(
                f"${orb_data.get('orb_low', 0):.2f} - ${orb_data.get('orb_high', 0):.2f}",
                style={"color": "#7dd87d"}
            ),
        ], style={"marginBottom": "10px"}),
        html.Div([
            html.Span("Status: ", style={"color": "rgba(125, 216, 125, 0.7)"}),
            html.Span(status, style={"color": status_color, "fontWeight": "bold"}),
        ], style={"marginBottom": "10px"}),
        html.Div([
            html.Span("Current Price: ", style={"color": "rgba(125, 216, 125, 0.7)"}),
            html.Span(f"${orb_data.get('current_price', 0):.2f}", style={"color": "#7dd87d"}),
        ]),
    ], style={
        "padding": "15px",
        "backgroundColor": "#1a1a1a",
        "borderRadius": "4px",
    })

def get_analysis_callback():
    """Get callback function for updating analysis display."""
    def update_analysis_display(selected_date):
        if not selected_date:
            return html.Div("No date selected.")
        
        analyses = get_daily_analyses(selected_date)
        return format_analysis_for_display(analyses)
    
    return update_analysis_display


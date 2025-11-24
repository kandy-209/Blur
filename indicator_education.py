#!/usr/bin/env python3
"""
Technical Indicator Education Module
Provides educational content and explanations for beginners
"""

from typing import Dict, List
from dash import html, dcc

# ========== INDICATOR DEFINITIONS ==========

INDICATOR_INFO = {
    "RSI": {
        "name": "Relative Strength Index (RSI)",
        "description": "Measures the speed and magnitude of price changes to identify overbought or oversold conditions.",
        "range": "0-100",
        "interpretation": {
            "overbought": "RSI > 70: Asset may be overbought, potential sell signal",
            "oversold": "RSI < 30: Asset may be oversold, potential buy signal",
            "neutral": "RSI 30-70: Normal trading range",
            "divergence": "Price makes new highs/lows but RSI doesn't - potential reversal signal"
        },
        "formula": "RSI = 100 - (100 / (1 + RS)) where RS = Average Gain / Average Loss",
        "best_use": "Best for identifying entry and exit points, especially in ranging markets",
        "timeframe": "Typically uses 14 periods, but can be adjusted (9 for faster, 25 for slower)"
    },
    "MACD": {
        "name": "Moving Average Convergence Divergence (MACD)",
        "description": "Shows the relationship between two moving averages of price. Helps identify trend changes and momentum.",
        "range": "No fixed range (oscillates around zero)",
        "interpretation": {
            "bullish_cross": "MACD line crosses above signal line: Bullish signal, potential uptrend",
            "bearish_cross": "MACD line crosses below signal line: Bearish signal, potential downtrend",
            "above_zero": "MACD above zero: Bullish momentum",
            "below_zero": "MACD below zero: Bearish momentum",
            "divergence": "Price and MACD move in opposite directions: Potential trend reversal"
        },
        "formula": "MACD = 12-period EMA - 26-period EMA, Signal = 9-period EMA of MACD",
        "best_use": "Best for identifying trend changes and momentum shifts",
        "timeframe": "Default: 12, 26, 9 periods"
    },
    "Stochastic": {
        "name": "Stochastic Oscillator",
        "description": "Compares closing price to price range over a period to identify momentum and potential reversal points.",
        "range": "0-100",
        "interpretation": {
            "overbought": "%K > 80: Overbought condition, potential sell signal",
            "oversold": "%K < 20: Oversold condition, potential buy signal",
            "bullish_cross": "%K crosses above %D from oversold: Bullish signal",
            "bearish_cross": "%K crosses below %D from overbought: Bearish signal"
        },
        "formula": "%K = ((Current Close - Lowest Low) / (Highest High - Lowest Low)) * 100",
        "best_use": "Best for identifying overbought/oversold conditions in ranging markets",
        "timeframe": "Typically uses 14 periods for %K, 3 periods for %D"
    },
    "Williams %R": {
        "name": "Williams %R",
        "description": "Momentum indicator that measures overbought and oversold levels, similar to Stochastic but inverted.",
        "range": "-100 to 0",
        "interpretation": {
            "overbought": "%R > -20: Overbought, potential sell signal",
            "oversold": "%R < -80: Oversold, potential buy signal",
            "neutral": "%R between -20 and -80: Normal range"
        },
        "formula": "%R = ((Highest High - Close) / (Highest High - Lowest Low)) * -100",
        "best_use": "Best for confirming other indicators and identifying reversal points",
        "timeframe": "Typically uses 14 periods"
    },
    "CCI": {
        "name": "Commodity Channel Index (CCI)",
        "description": "Identifies cyclical trends by measuring price deviation from statistical mean.",
        "range": "Typically -200 to +200, but can extend beyond",
        "interpretation": {
            "overbought": "CCI > +100: Overbought, potential sell signal",
            "oversold": "CCI < -100: Oversold, potential buy signal",
            "strong_trend": "CCI > +100 or < -100: Strong trend, momentum continues",
            "weak_signal": "CCI between -100 and +100: Weak or no clear signal"
        },
        "formula": "CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)",
        "best_use": "Best for identifying cyclical trends and extreme price movements",
        "timeframe": "Typically uses 20 periods"
    },
    "MFI": {
        "name": "Money Flow Index (MFI)",
        "description": "Volume-weighted RSI that combines price and volume to identify overbought/oversold conditions.",
        "range": "0-100",
        "interpretation": {
            "overbought": "MFI > 80: Overbought, potential sell signal",
            "oversold": "MFI < 20: Oversold, potential buy signal",
            "divergence": "Price and MFI diverge: Potential reversal signal"
        },
        "formula": "MFI = 100 - (100 / (1 + Money Flow Ratio))",
        "best_use": "Best for confirming price movements with volume analysis",
        "timeframe": "Typically uses 14 periods"
    },
    "Bollinger Bands": {
        "name": "Bollinger Bands",
        "description": "Volatility bands placed above and below a moving average. Shows price volatility and potential support/resistance.",
        "range": "Dynamic based on standard deviation",
        "interpretation": {
            "squeeze": "Bands narrow: Low volatility, potential breakout coming",
            "expansion": "Bands widen: High volatility, strong trend",
            "upper_touch": "Price touches upper band: Overbought, potential reversal",
            "lower_touch": "Price touches lower band: Oversold, potential reversal",
            "middle_line": "Price at middle (SMA): Mean reversion point"
        },
        "formula": "Upper Band = SMA + (2 * Standard Deviation), Lower Band = SMA - (2 * Standard Deviation)",
        "best_use": "Best for identifying volatility and mean reversion opportunities",
        "timeframe": "Typically uses 20-period SMA with 2 standard deviations"
    },
    "ATR": {
        "name": "Average True Range (ATR)",
        "description": "Measures market volatility by calculating the average of true ranges over a period.",
        "range": "No fixed range (absolute value)",
        "interpretation": {
            "high_atr": "High ATR: High volatility, wider stop losses needed",
            "low_atr": "Low ATR: Low volatility, tighter stop losses possible",
            "increasing": "ATR increasing: Volatility rising, trend strengthening",
            "decreasing": "ATR decreasing: Volatility falling, consolidation possible"
        },
        "formula": "ATR = Average of True Range over N periods, where True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)",
        "best_use": "Best for setting stop losses and position sizing based on volatility",
        "timeframe": "Typically uses 14 periods"
    },
    "VWAP": {
        "name": "Volume Weighted Average Price (VWAP)",
        "description": "Average price weighted by volume. Institutional traders use it as a benchmark.",
        "range": "No fixed range (price level)",
        "interpretation": {
            "above_vwap": "Price above VWAP: Bullish, buyers in control",
            "below_vwap": "Price below VWAP: Bearish, sellers in control",
            "support": "VWAP acts as support in uptrends",
            "resistance": "VWAP acts as resistance in downtrends"
        },
        "formula": "VWAP = Sum(Price * Volume) / Sum(Volume)",
        "best_use": "Best for intraday trading and identifying institutional activity",
        "timeframe": "Typically calculated for the trading day"
    },
    "Moving Averages": {
        "name": "Moving Averages (MA)",
        "description": "Smooth out price data to identify trends by calculating average price over a period.",
        "range": "No fixed range (price level)",
        "interpretation": {
            "golden_cross": "Fast MA crosses above slow MA: Bullish signal, uptrend",
            "death_cross": "Fast MA crosses below slow MA: Bearish signal, downtrend",
            "support": "Price above MA: Uptrend, MA acts as support",
            "resistance": "Price below MA: Downtrend, MA acts as resistance"
        },
        "formula": "SMA = Sum of prices / Number of periods, EMA = More weight to recent prices",
        "best_use": "Best for identifying trend direction and support/resistance levels",
        "timeframe": "Common: 20 (short-term), 50 (medium-term), 200 (long-term)"
    }
}

# ========== HELPER FUNCTIONS ==========

def get_indicator_info(indicator_name: str) -> Dict:
    """Get information about a specific indicator."""
    # Normalize indicator name
    indicator_name = indicator_name.upper()
    
    # Map variations to standard names
    name_mapping = {
        "RSI": "RSI",
        "MACD": "MACD",
        "STOCH": "Stochastic",
        "STOCHASTIC": "Stochastic",
        "WILLIAMS": "Williams %R",
        "WILLIAMS %R": "Williams %R",
        "CCI": "CCI",
        "MFI": "MFI",
        "BOLLINGER": "Bollinger Bands",
        "BB": "Bollinger Bands",
        "ATR": "ATR",
        "VWAP": "VWAP",
        "MA": "Moving Averages",
        "MOVING AVERAGE": "Moving Averages",
        "SMA": "Moving Averages",
        "EMA": "Moving Averages"
    }
    
    standard_name = name_mapping.get(indicator_name, indicator_name)
    return INDICATOR_INFO.get(standard_name, {})

def create_indicator_tooltip(indicator_name: str, current_value: float = None) -> html.Div:
    """Create a tooltip with basic info about an indicator."""
    info = get_indicator_info(indicator_name)
    
    if not info:
        return html.Div()
    
    tooltip_content = [
        html.Div([
            html.Strong(f"{info['name']}", style={"color": "#7dd87d", "fontSize": "14px"}),
            html.Br(),
            html.Span(info['description'], style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "12px"}),
            html.Br(),
            html.Span(f"Range: {info['range']}", style={"color": "rgba(125, 216, 125, 0.7)", "fontSize": "11px", "fontStyle": "italic"}),
        ])
    ]
    
    if current_value is not None:
        # Add interpretation based on current value
        interpretation = get_value_interpretation(indicator_name, current_value, info)
        if interpretation:
            tooltip_content.append(
                html.Div([
                    html.Br(),
                    html.Strong("Current Status:", style={"color": "#7dd87d", "fontSize": "12px"}),
                    html.Br(),
                    html.Span(interpretation, style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "11px"})
                ])
            )
    
    return html.Div(
        tooltip_content,
        style={
            "backgroundColor": "#1a1a1a",
            "border": "1px solid rgba(125, 216, 125, 0.4)",
            "borderRadius": "4px",
            "padding": "10px",
            "marginTop": "5px",
            "fontSize": "12px"
        }
    )

def get_value_interpretation(indicator_name: str, value: float, info: Dict) -> str:
    """Get interpretation of current indicator value."""
    indicator_name = indicator_name.upper()
    
    if indicator_name == "RSI":
        if value > 70:
            return info['interpretation'].get('overbought', '')
        elif value < 30:
            return info['interpretation'].get('oversold', '')
        else:
            return info['interpretation'].get('neutral', '')
    
    elif indicator_name == "STOCHASTIC" or indicator_name == "STOCH":
        if value > 80:
            return info['interpretation'].get('overbought', '')
        elif value < 20:
            return info['interpretation'].get('oversold', '')
    
    elif indicator_name == "WILLIAMS %R" or indicator_name == "WILLIAMS":
        if value > -20:
            return info['interpretation'].get('overbought', '')
        elif value < -80:
            return info['interpretation'].get('oversold', '')
    
    elif indicator_name == "CCI":
        if value > 100:
            return info['interpretation'].get('overbought', '')
        elif value < -100:
            return info['interpretation'].get('oversold', '')
    
    elif indicator_name == "MFI":
        if value > 80:
            return info['interpretation'].get('overbought', '')
        elif value < 20:
            return info['interpretation'].get('oversold', '')
    
    return ""

def create_indicator_education_page(indicator_name: str) -> html.Div:
    """Create a full educational page for an indicator."""
    info = get_indicator_info(indicator_name)
    
    if not info:
        return html.Div("Indicator information not found.")
    
    return html.Div([
        html.Div([
            html.H2(
                info['name'],
                style={
                    "color": "#7dd87d",
                    "fontFamily": "'Share Tech Mono', monospace",
                    "marginBottom": "20px",
                    "textAlign": "center"
                }
            ),
            
            html.Div([
                html.H3("Description", style={"color": "#7dd87d", "fontSize": "18px", "marginTop": "20px"}),
                html.P(
                    info['description'],
                    style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "14px", "lineHeight": "1.6"}
                ),
            ]),
            
            html.Div([
                html.H3("Range", style={"color": "#7dd87d", "fontSize": "18px", "marginTop": "20px"}),
                html.P(
                    f"Typical Range: {info['range']}",
                    style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "14px"}
                ),
            ]),
            
            html.Div([
                html.H3("How to Interpret", style={"color": "#7dd87d", "fontSize": "18px", "marginTop": "20px"}),
                html.Ul([
                    html.Li([
                        html.Strong(key.replace('_', ' ').title() + ":", style={"color": "#7dd87d"}),
                        html.Span(f" {value}", style={"color": "rgba(125, 216, 125, 0.8)"})
                    ])
                    for key, value in info['interpretation'].items()
                ], style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "14px", "lineHeight": "1.8"})
            ]),
            
            html.Div([
                html.H3("Formula", style={"color": "#7dd87d", "fontSize": "18px", "marginTop": "20px"}),
                html.P(
                    info['formula'],
                    style={
                        "color": "rgba(125, 216, 125, 0.8)",
                        "fontSize": "13px",
                        "fontFamily": "monospace",
                        "backgroundColor": "#1a1a1a",
                        "padding": "10px",
                        "borderRadius": "4px",
                        "border": "1px solid rgba(125, 216, 125, 0.3)"
                    }
                ),
            ]),
            
            html.Div([
                html.H3("Best Use Cases", style={"color": "#7dd87d", "fontSize": "18px", "marginTop": "20px"}),
                html.P(
                    info['best_use'],
                    style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "14px", "lineHeight": "1.6"}
                ),
            ]),
            
            html.Div([
                html.H3("Recommended Timeframe", style={"color": "#7dd87d", "fontSize": "18px", "marginTop": "20px"}),
                html.P(
                    info['timeframe'],
                    style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "14px"}
                ),
            ]),
            
            html.Div([
                html.Button(
                    "← Back to Dashboard",
                    id="back-to-dashboard",
                    n_clicks=0,
                    style={
                        "backgroundColor": "rgba(125, 216, 125, 0.2)",
                        "border": "1px solid rgba(125, 216, 125, 0.5)",
                        "color": "#7dd87d",
                        "padding": "10px 20px",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "marginTop": "30px"
                    }
                )
            ], style={"textAlign": "center"})
        ], style={
            "maxWidth": "800px",
            "margin": "0 auto",
            "padding": "30px",
            "backgroundColor": "#0a0a0a",
            "border": "1px solid rgba(125, 216, 125, 0.4)",
            "borderRadius": "8px"
        })
    ], style={"padding": "20px"})

def create_indicator_link(indicator_name: str, current_value: float = None) -> html.Div:
    """Create a clickable link to indicator education."""
    info = get_indicator_info(indicator_name)
    
    if not info:
        return html.Div()
    
    return html.Div([
        html.A(
            f"📚 Learn about {info['name']}",
            href=f"/indicator/{indicator_name.lower().replace(' ', '-')}",
            style={
                "color": "#7dd87d",
                "textDecoration": "underline",
                "fontSize": "12px",
                "cursor": "pointer",
                "marginTop": "5px",
                "display": "block"
            }
        ),
        create_indicator_tooltip(indicator_name, current_value)
    ])

# ========== QUICK REFERENCE GUIDE ==========

def create_quick_reference_guide() -> html.Div:
    """Create a quick reference guide for all indicators."""
    return html.Div([
        html.H2(
            "Technical Indicators Quick Reference",
            style={
                "color": "#7dd87d",
                "fontFamily": "'Share Tech Mono', monospace",
                "marginBottom": "30px",
                "textAlign": "center"
            }
        ),
        
        html.Div([
            html.Div([
                html.H3(
                    info['name'],
                    style={"color": "#7dd87d", "fontSize": "16px", "marginBottom": "10px"}
                ),
                html.P(
                    info['description'],
                    style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "13px", "marginBottom": "5px"}
                ),
                html.P(
                    f"Range: {info['range']}",
                    style={"color": "rgba(125, 216, 125, 0.6)", "fontSize": "11px", "fontStyle": "italic"}
                ),
                html.A(
                    "Learn More →",
                    href=f"/indicator/{name.lower().replace(' ', '-')}",
                    style={
                        "color": "#7dd87d",
                        "textDecoration": "underline",
                        "fontSize": "12px",
                        "marginTop": "5px",
                        "display": "block"
                    }
                )
            ], style={
                "backgroundColor": "#1a1a1a",
                "border": "1px solid rgba(125, 216, 125, 0.3)",
                "borderRadius": "4px",
                "padding": "15px",
                "marginBottom": "15px"
            })
            for name, info in INDICATOR_INFO.items()
        ])
    ], style={
        "maxWidth": "900px",
        "margin": "0 auto",
        "padding": "30px"
    })


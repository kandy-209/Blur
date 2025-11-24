#!/usr/bin/env python3
"""
Price Action Analysis & ORB (Opening Range Breakout) Analysis Module
Comprehensive price action and opening range breakout analysis for futures trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pytz

# ========== PRICE ACTION ANALYSIS ==========

def identify_trend(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Identify current trend using price action principles.
    Returns trend direction, strength, and structure.
    """
    if df.empty or len(df) < lookback:
        return {"trend": "UNKNOWN", "strength": 0, "structure": []}
    
    recent = df.tail(lookback).copy()
    
    # Calculate swing highs and lows
    highs = recent["High"].values
    lows = recent["Low"].values
    
    # Identify higher highs and higher lows (uptrend)
    # Identify lower highs and lower lows (downtrend)
    
    # Get recent peaks and troughs
    peaks = []
    troughs = []
    
    for i in range(2, len(recent) - 2):
        # Peak: high is higher than 2 before and 2 after
        if highs[i] > highs[i-2] and highs[i] > highs[i+2]:
            peaks.append((i, highs[i]))
        # Trough: low is lower than 2 before and 2 after
        if lows[i] < lows[i-2] and lows[i] < lows[i+2]:
            troughs.append((i, lows[i]))
    
    # Determine trend
    trend = "SIDEWAYS"
    strength = 0
    structure = []
    
    if len(peaks) >= 2 and len(troughs) >= 2:
        # Check for uptrend (higher highs and higher lows)
        recent_peaks = sorted(peaks[-2:], key=lambda x: x[0])
        recent_troughs = sorted(troughs[-2:], key=lambda x: x[0])
        
        if (recent_peaks[-1][1] > recent_peaks[-2][1] and 
            recent_troughs[-1][1] > recent_troughs[-2][1]):
            trend = "UPTREND"
            strength = min(100, abs((recent_peaks[-1][1] - recent_peaks[-2][1]) / recent_peaks[-2][1]) * 1000)
            structure = ["Higher Highs", "Higher Lows"]
        
        # Check for downtrend (lower highs and lower lows)
        elif (recent_peaks[-1][1] < recent_peaks[-2][1] and 
              recent_troughs[-1][1] < recent_troughs[-2][1]):
            trend = "DOWNTREND"
            strength = min(100, abs((recent_peaks[-2][1] - recent_peaks[-1][1]) / recent_peaks[-2][1]) * 1000)
            structure = ["Lower Highs", "Lower Lows"]
    
    # Current price position
    current_price = float(recent["Close"].iloc[-1])
    price_range = float(recent["High"].max() - recent["Low"].min())
    price_position = ((current_price - recent["Low"].min()) / price_range * 100) if price_range > 0 else 50
    
    return {
        "trend": trend,
        "strength": round(strength, 2),
        "structure": structure,
        "peaks": len(peaks),
        "troughs": len(troughs),
        "price_position_pct": round(price_position, 2),
        "current_price": current_price,
    }

def identify_support_resistance(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Identify key support and resistance levels using price action.
    Returns multiple support/resistance zones with strength.
    """
    if df.empty or len(df) < 20:
        return {"support_levels": [], "resistance_levels": []}
    
    recent = df.tail(lookback).copy()
    
    # Find local minima (support) and maxima (resistance)
    window = max(5, len(recent) // 10)
    
    support_levels = []
    resistance_levels = []
    
    # Find support levels (local lows that were tested multiple times)
    for i in range(window, len(recent) - window):
        low_val = recent["Low"].iloc[i]
        # Check if this is a local minimum
        is_local_min = all(low_val <= recent["Low"].iloc[i-j] for j in range(1, window+1)) and \
                      all(low_val <= recent["Low"].iloc[i+j] for j in range(1, window+1))
        
        if is_local_min:
            # Count how many times this level was tested
            tolerance = low_val * 0.001  # 0.1% tolerance
            touches = sum(abs(recent["Low"] - low_val) < tolerance)
            
            if touches >= 2:  # At least 2 touches
                support_levels.append({
                    "price": float(low_val),
                    "strength": min(100, touches * 20),  # Strength based on touches
                    "touches": int(touches),
                })
    
    # Find resistance levels (local highs that were tested multiple times)
    for i in range(window, len(recent) - window):
        high_val = recent["High"].iloc[i]
        # Check if this is a local maximum
        is_local_max = all(high_val >= recent["High"].iloc[i-j] for j in range(1, window+1)) and \
                      all(high_val >= recent["High"].iloc[i+j] for j in range(1, window+1))
        
        if is_local_max:
            # Count how many times this level was tested
            tolerance = high_val * 0.001  # 0.1% tolerance
            touches = sum(abs(recent["High"] - high_val) < tolerance)
            
            if touches >= 2:  # At least 2 touches
                resistance_levels.append({
                    "price": float(high_val),
                    "strength": min(100, touches * 20),
                    "touches": int(touches),
                })
    
    # Sort and get most significant levels
    support_levels = sorted(support_levels, key=lambda x: x["price"], reverse=True)[:5]
    resistance_levels = sorted(resistance_levels, key=lambda x: x["price"])[:5]
    
    return {
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
    }

def analyze_candlestick_patterns(df: pd.DataFrame) -> Dict:
    """
    Comprehensive candlestick pattern analysis.
    Returns detected patterns with significance.
    """
    if df.empty or len(df) < 5:
        return {"patterns": [], "signals": []}
    
    patterns = []
    signals = []
    recent = df.tail(10).copy()
    
    # Analyze last few candles
    for i in range(max(0, len(recent) - 5), len(recent)):
        if i < 1:
            continue
            
        curr = recent.iloc[i]
        prev = recent.iloc[i-1]
        
        open_price = float(curr["Open"])
        close_price = float(curr["Close"])
        high_price = float(curr["High"])
        low_price = float(curr["Low"])
        
        prev_open = float(prev["Open"])
        prev_close = float(prev["Close"])
        prev_high = float(prev["High"])
        prev_low = float(prev["Low"])
        
        body = abs(close_price - open_price)
        total_range = high_price - low_price
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        is_bullish = close_price > open_price
        prev_bullish = prev_close > prev_open
        
        # Doji
        if total_range > 0 and body / total_range < 0.1:
            patterns.append({
                "name": "DOJI",
                "type": "REVERSAL",
                "strength": "MEDIUM",
                "candle_index": i,
            })
        
        # Hammer / Hanging Man
        if lower_shadow > body * 2 and upper_shadow < body * 0.3:
            if is_bullish:
                patterns.append({
                    "name": "HAMMER",
                    "type": "BULLISH_REVERSAL",
                    "strength": "HIGH",
                    "candle_index": i,
                })
                signals.append("BULLISH - Potential reversal from support")
            else:
                patterns.append({
                    "name": "HANGING_MAN",
                    "type": "BEARISH_REVERSAL",
                    "strength": "HIGH",
                    "candle_index": i,
                })
                signals.append("BEARISH - Potential reversal from resistance")
        
        # Engulfing patterns
        if i >= 1:
            prev_body = abs(prev_close - prev_open)
            curr_body = abs(close_price - open_price)
            
            # Bullish Engulfing
            if (not prev_bullish and is_bullish and
                open_price < prev_close and close_price > prev_open and
                curr_body > prev_body * 1.5):
                patterns.append({
                    "name": "BULLISH_ENGULFING",
                    "type": "BULLISH_REVERSAL",
                    "strength": "HIGH",
                    "candle_index": i,
                })
                signals.append("BULLISH - Strong reversal signal")
            
            # Bearish Engulfing
            if (prev_bullish and not is_bullish and
                open_price > prev_close and close_price < prev_open and
                curr_body > prev_body * 1.5):
                patterns.append({
                    "name": "BEARISH_ENGULFING",
                    "type": "BEARISH_REVERSAL",
                    "strength": "HIGH",
                    "candle_index": i,
                })
                signals.append("BEARISH - Strong reversal signal")
        
        # Shooting Star
        if upper_shadow > body * 2 and lower_shadow < body * 0.3 and not is_bullish:
            patterns.append({
                "name": "SHOOTING_STAR",
                "type": "BEARISH_REVERSAL",
                "strength": "MEDIUM",
                "candle_index": i,
            })
            signals.append("BEARISH - Rejection at highs")
        
        # Marubozu (strong trend continuation)
        if body / total_range > 0.9:
            if is_bullish:
                patterns.append({
                    "name": "BULLISH_MARUBOZU",
                    "type": "BULLISH_CONTINUATION",
                    "strength": "HIGH",
                    "candle_index": i,
                })
            else:
                patterns.append({
                    "name": "BEARISH_MARUBOZU",
                    "type": "BEARISH_CONTINUATION",
                    "strength": "HIGH",
                    "candle_index": i,
                })
    
    return {
        "patterns": patterns[-5:],  # Last 5 patterns
        "signals": signals[-3:],  # Last 3 signals
    }

def identify_liquidity_zones(df: pd.DataFrame) -> Dict:
    """
    Identify liquidity zones (areas where stops are likely placed).
    Returns zones above/below current price.
    """
    if df.empty or len(df) < 20:
        return {"liquidity_above": [], "liquidity_below": []}
    
    recent = df.tail(50).copy()
    current_price = float(recent["Close"].iloc[-1])
    
    # Find areas with high volume and price rejection
    liquidity_above = []
    liquidity_below = []
    
    # Look for wicks (rejections) that indicate stop losses
    for i in range(len(recent)):
        candle = recent.iloc[i]
        high = float(candle["High"])
        low = float(candle["Low"])
        close = float(candle["Close"])
        volume = float(candle.get("Volume", 0))
        
        # Upper wick rejection (potential liquidity above)
        upper_wick = high - max(float(candle["Open"]), close)
        if upper_wick > abs(close - float(candle["Open"])) * 1.5:  # Long upper wick
            liquidity_above.append({
                "price": high,
                "strength": min(100, volume / recent["Volume"].mean() * 50),
            })
        
        # Lower wick rejection (potential liquidity below)
        lower_wick = min(float(candle["Open"]), close) - low
        if lower_wick > abs(close - float(candle["Open"])) * 1.5:  # Long lower wick
            liquidity_below.append({
                "price": low,
                "strength": min(100, volume / recent["Volume"].mean() * 50),
            })
    
    # Consolidate nearby levels
    def consolidate_levels(levels, current, above=True):
        if not levels:
            return []
        sorted_levels = sorted(levels, key=lambda x: x["price"], reverse=above)
        consolidated = []
        tolerance = current_price * 0.002  # 0.2% tolerance
        
        for level in sorted_levels:
            if above and level["price"] > current_price:
                if not consolidated or abs(level["price"] - consolidated[-1]["price"]) > tolerance:
                    consolidated.append(level)
            elif not above and level["price"] < current_price:
                if not consolidated or abs(level["price"] - consolidated[-1]["price"]) > tolerance:
                    consolidated.append(level)
        
        return consolidated[:3]  # Top 3 levels
    
    return {
        "liquidity_above": consolidate_levels(liquidity_above, current_price, above=True),
        "liquidity_below": consolidate_levels(liquidity_below, current_price, above=False),
    }

# ========== ORB (OPENING RANGE BREAKOUT) ANALYSIS ==========

def calculate_opening_range(df: pd.DataFrame, minutes: int = 30) -> Dict:
    """
    Calculate Opening Range Breakout levels.
    Opening range is typically first 30-60 minutes of trading.
    
    Args:
        df: DataFrame with datetime index and OHLCV data
        minutes: Number of minutes for opening range (default 30)
    
    Returns:
        Dict with ORB high, low, range size, and breakout status
    """
    if df.empty:
        return {"error": "No data available"}
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df = df.set_index("Datetime")
        else:
            return {"error": "No datetime index or column found"}
    
    # Convert to market timezone (EST/EDT for US futures)
    market_tz = pytz.timezone('America/New_York')
    
    # Get today's date
    today = datetime.now(market_tz).date()
    
    # Filter today's data
    today_data = df[df.index.date == today] if hasattr(df.index, 'date') else df
    
    if today_data.empty:
        # Try to get most recent trading day
        today_data = df.tail(100)  # Get last 100 data points
    
    if today_data.empty:
        return {"error": "No trading data available"}
    
    # Get first N minutes of data
    if len(today_data) > 0:
        start_time = today_data.index[0]
        end_time = start_time + timedelta(minutes=minutes)
        
        orb_data = today_data[(today_data.index >= start_time) & (today_data.index <= end_time)]
        
        if orb_data.empty:
            # Fallback: use first available data points
            orb_data = today_data.head(min(30, len(today_data)))
    
    if orb_data.empty:
        return {"error": "Insufficient data for opening range"}
    
    # Calculate ORB high and low
    orb_high = float(orb_data["High"].max())
    orb_low = float(orb_data["Low"].min())
    orb_range = orb_high - orb_low
    orb_mid = (orb_high + orb_low) / 2
    
    # Current price
    current_price = float(df["Close"].iloc[-1])
    
    # Determine breakout status
    breakout_status = "INSIDE_RANGE"
    breakout_direction = None
    breakout_distance = 0
    
    if current_price > orb_high:
        breakout_status = "BROKEN_ABOVE"
        breakout_direction = "BULLISH"
        breakout_distance = current_price - orb_high
    elif current_price < orb_low:
        breakout_status = "BROKEN_BELOW"
        breakout_direction = "BEARISH"
        breakout_distance = orb_low - current_price
    
    # Calculate target levels (based on range size)
    target_above = orb_high + orb_range  # 1x range target
    target_below = orb_low - orb_range   # 1x range target
    
    # Calculate position within range (if inside)
    position_in_range = 0
    if breakout_status == "INSIDE_RANGE":
        position_in_range = ((current_price - orb_low) / orb_range * 100) if orb_range > 0 else 50
    
    return {
        "orb_high": orb_high,
        "orb_low": orb_low,
        "orb_mid": orb_mid,
        "orb_range": orb_range,
        "range_size_pct": (orb_range / orb_mid * 100) if orb_mid > 0 else 0,
        "current_price": current_price,
        "breakout_status": breakout_status,
        "breakout_direction": breakout_direction,
        "breakout_distance": breakout_distance,
        "breakout_distance_pct": (breakout_distance / orb_mid * 100) if orb_mid > 0 and breakout_distance > 0 else 0,
        "target_above": target_above,
        "target_below": target_below,
        "position_in_range_pct": position_in_range,
        "start_time": orb_data.index[0].isoformat() if len(orb_data) > 0 else None,
        "end_time": orb_data.index[-1].isoformat() if len(orb_data) > 0 else None,
    }

def analyze_orb_performance(df: pd.DataFrame, days: int = 20) -> Dict:
    """
    Analyze historical ORB performance to determine success rate.
    """
    if df.empty or len(df) < 100:
        return {"success_rate": 0, "avg_move": 0, "breakout_stats": {}}
    
    # This would require daily data to analyze past ORB breakouts
    # For now, return placeholder structure
    return {
        "success_rate": 0,  # Would calculate from historical data
        "avg_move": 0,
        "breakout_stats": {
            "bullish_breakouts": 0,
            "bearish_breakouts": 0,
            "failed_breakouts": 0,
        }
    }

# ========== COMPREHENSIVE PRICE ACTION ANALYSIS ==========

def comprehensive_price_action_analysis(df: pd.DataFrame) -> Dict:
    """
    Run comprehensive price action analysis.
    Returns all price action metrics and insights.
    """
    if df.empty:
        return {"error": "No data available"}
    
    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trend_analysis": identify_trend(df),
        "support_resistance": identify_support_resistance(df),
        "candlestick_patterns": analyze_candlestick_patterns(df),
        "liquidity_zones": identify_liquidity_zones(df),
    }
    
    return analysis

def comprehensive_orb_analysis(df: pd.DataFrame, orb_minutes: int = 30) -> Dict:
    """
    Run comprehensive ORB analysis.
    Returns ORB levels, breakout status, and targets.
    """
    if df.empty:
        return {"error": "No data available"}
    
    orb_data = calculate_opening_range(df, minutes=orb_minutes)
    
    if "error" in orb_data:
        return orb_data
    
    # Add additional ORB insights
    orb_data["analysis"] = {
        "range_quality": "NARROW" if orb_data["range_size_pct"] < 0.5 else "WIDE" if orb_data["range_size_pct"] > 1.0 else "NORMAL",
        "breakout_probability": "HIGH" if orb_data["range_size_pct"] < 0.5 else "MEDIUM",
        "recommended_action": _get_orb_recommendation(orb_data),
    }
    
    return orb_data

def _get_orb_recommendation(orb_data: Dict) -> str:
    """Get trading recommendation based on ORB analysis."""
    status = orb_data.get("breakout_status", "INSIDE_RANGE")
    
    if status == "BROKEN_ABOVE":
        return "BULLISH - Price broke above opening range. Target: ${:.2f}".format(orb_data.get("target_above", 0))
    elif status == "BROKEN_BELOW":
        return "BEARISH - Price broke below opening range. Target: ${:.2f}".format(orb_data.get("target_below", 0))
    else:
        position = orb_data.get("position_in_range_pct", 50)
        if position > 70:
            return "NEAR RESISTANCE - Watch for breakout above ${:.2f}".format(orb_data.get("orb_high", 0))
        elif position < 30:
            return "NEAR SUPPORT - Watch for breakdown below ${:.2f}".format(orb_data.get("orb_low", 0))
        else:
            return "INSIDE RANGE - Wait for breakout. Range: ${:.2f} - ${:.2f}".format(
                orb_data.get("orb_low", 0), orb_data.get("orb_high", 0))

# ========== TEST FUNCTION ==========

if __name__ == "__main__":
    # Test the analysis functions
    print("=" * 60)
    print("PRICE ACTION & ORB ANALYSIS TEST")
    print("=" * 60)
    
    # You would load your actual data here
    # For testing, this is a placeholder
    print("\nTo test, load your DataFrame and call:")
    print("  - comprehensive_price_action_analysis(df)")
    print("  - comprehensive_orb_analysis(df, orb_minutes=30)")
    print("\nExample:")
    print("  import yfinance as yf")
    print("  ticker = yf.Ticker('ES=F')")
    print("  df = ticker.history(period='5d', interval='1m')")
    print("  price_action = comprehensive_price_action_analysis(df)")
    print("  orb = comprehensive_orb_analysis(df)")


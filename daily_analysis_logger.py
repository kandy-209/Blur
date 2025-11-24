#!/usr/bin/env python3
"""
Daily Analysis Logger with LLM-Powered Written Analysis
Generates and stores comprehensive market analysis during opening hours (9:30 AM - 11:30 AM EST)
"""

import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import pytz
import requests
from pathlib import Path

# LLM API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
USE_LLM = bool(OPENAI_API_KEY or ANTHROPIC_API_KEY)

# Storage directory
ANALYSIS_DIR = Path("data/analysis_logs")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Market hours (EST/EDT)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
ANALYSIS_END_HOUR = 11
ANALYSIS_END_MINUTE = 30

# ========== TIME UTILITIES ==========

def get_market_timezone():
    """Get US Eastern timezone (handles EST/EDT automatically)."""
    return pytz.timezone('America/New_York')

def is_during_analysis_hours() -> bool:
    """Check if current time is during analysis hours (9:30 AM - 11:30 AM EST)."""
    try:
        est = get_market_timezone()
        now_est = datetime.now(est)
        
        # Check if it's a weekday (Monday=0, Friday=4)
        if now_est.weekday() >= 5:  # Saturday or Sunday
            return False
        
        current_time = now_est.time()
        open_time = datetime.strptime(f"{MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE}", "%H:%M").time()
        end_time = datetime.strptime(f"{ANALYSIS_END_HOUR}:{ANALYSIS_END_MINUTE}", "%H:%M").time()
        
        return open_time <= current_time <= end_time
    except Exception as e:
        print(f"Error checking analysis hours: {e}")
        return False

def get_today_date_str() -> str:
    """Get today's date as string (YYYY-MM-DD)."""
    est = get_market_timezone()
    return datetime.now(est).strftime("%Y-%m-%d")

# ========== DATA COLLECTION ==========

def collect_market_data(symbols: List[str] = None) -> Dict:
    """
    Collect comprehensive market data for analysis.
    """
    if symbols is None:
        symbols = ["ES=F", "NQ=F", "GC=F"]
    
    market_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "date": get_today_date_str(),
        "symbols": {},
    }
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # Get 1-minute data for today
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                latest = data.iloc[-1]
                
                # Calculate basic metrics
                current_price = float(latest["Close"])
                high = float(data["High"].max())
                low = float(data["Low"].min())
                volume = int(data["Volume"].sum())
                
                # Calculate price change from open
                open_price = float(data["Open"].iloc[0])
                price_change = current_price - open_price
                price_change_pct = (price_change / open_price * 100) if open_price > 0 else 0
                
                # Get recent trend
                recent_5m = data.tail(5)
                recent_trend = "NEUTRAL"
                if len(recent_5m) > 1:
                    if recent_5m["Close"].iloc[-1] > recent_5m["Close"].iloc[0]:
                        recent_trend = "BULLISH"
                    elif recent_5m["Close"].iloc[-1] < recent_5m["Close"].iloc[0]:
                        recent_trend = "BEARISH"
                
                market_data["symbols"][symbol] = {
                    "current_price": current_price,
                    "open_price": open_price,
                    "high": high,
                    "low": low,
                    "price_change": price_change,
                    "price_change_pct": price_change_pct,
                    "volume": volume,
                    "recent_trend": recent_trend,
                    "data_points": len(data),
                }
        except Exception as e:
            print(f"Error collecting data for {symbol}: {e}")
            market_data["symbols"][symbol] = {"error": str(e)}
    
    return market_data

# ========== LLM ANALYSIS GENERATION ==========

def generate_analysis_with_openai(market_data: Dict, price_action_data: Dict = None, orb_data: Dict = None) -> str:
    """Generate written analysis using OpenAI API."""
    if not OPENAI_API_KEY:
        return "OpenAI API key not configured"
    
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Build context from market data
        context = f"""
Market Data Summary for {market_data.get('date', 'today')}:
"""
        for symbol, data in market_data.get("symbols", {}).items():
            if "error" not in data:
                context += f"""
{symbol}:
- Current Price: ${data.get('current_price', 0):.2f}
- Open Price: ${data.get('open_price', 0):.2f}
- Change: ${data.get('price_change', 0):.2f} ({data.get('price_change_pct', 0):+.2f}%)
- High: ${data.get('high', 0):.2f}
- Low: ${data.get('low', 0):.2f}
- Volume: {data.get('volume', 0):,}
- Recent Trend: {data.get('recent_trend', 'NEUTRAL')}
"""
        
        if price_action_data:
            context += f"""
Price Action Analysis:
- Trend: {price_action_data.get('trend_analysis', {}).get('trend', 'UNKNOWN')}
- Support/Resistance Levels Identified: {len(price_action_data.get('support_resistance', {}).get('support_levels', []))} support, {len(price_action_data.get('support_resistance', {}).get('resistance_levels', []))} resistance
- Candlestick Patterns: {len(price_action_data.get('candlestick_patterns', {}).get('patterns', []))} patterns detected
"""
        
        if orb_data:
            context += f"""
Opening Range Breakout (ORB) Analysis:
- ORB High: ${orb_data.get('orb_high', 0):.2f}
- ORB Low: ${orb_data.get('orb_low', 0):.2f}
- Current Status: {orb_data.get('breakout_status', 'UNKNOWN')}
- Breakout Direction: {orb_data.get('breakout_direction', 'NONE')}
"""
        
        prompt = f"""You are an expert futures market analyst. Based on the following market data from the opening hours (9:30 AM - 11:30 AM EST), write a comprehensive, professional market analysis.

{context}

Write a detailed analysis that includes:
1. Market Overview: Summary of overall market conditions
2. Key Observations: Notable price movements, volume patterns, and trends
3. Technical Analysis: Support/resistance levels, patterns, and indicators
4. Trading Opportunities: Potential setups and risk considerations
5. Outlook: Short-term expectations for the remainder of the session

Write in a clear, professional tone suitable for active traders. Be specific with price levels and percentages. Keep it concise but comprehensive (300-500 words)."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using cost-effective model
            messages=[
                {"role": "system", "content": "You are an expert futures market analyst with deep knowledge of technical analysis, price action, and market psychology."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error generating analysis with OpenAI: {str(e)}"

def generate_analysis_with_anthropic(market_data: Dict, price_action_data: Dict = None, orb_data: Dict = None) -> str:
    """Generate written analysis using Anthropic Claude API."""
    if not ANTHROPIC_API_KEY:
        return "Anthropic API key not configured"
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Build context (same as OpenAI)
        context = f"""
Market Data Summary for {market_data.get('date', 'today')}:
"""
        for symbol, data in market_data.get("symbols", {}).items():
            if "error" not in data:
                context += f"""
{symbol}:
- Current Price: ${data.get('current_price', 0):.2f}
- Open Price: ${data.get('open_price', 0):.2f}
- Change: ${data.get('price_change', 0):.2f} ({data.get('price_change_pct', 0):+.2f}%)
- High: ${data.get('high', 0):.2f}
- Low: ${data.get('low', 0):.2f}
- Volume: {data.get('volume', 0):,}
- Recent Trend: {data.get('recent_trend', 'NEUTRAL')}
"""
        
        if price_action_data:
            context += f"""
Price Action Analysis:
- Trend: {price_action_data.get('trend_analysis', {}).get('trend', 'UNKNOWN')}
- Support/Resistance Levels: {len(price_action_data.get('support_resistance', {}).get('support_levels', []))} support, {len(price_action_data.get('support_resistance', {}).get('resistance_levels', []))} resistance
"""
        
        if orb_data:
            context += f"""
Opening Range Breakout (ORB) Analysis:
- ORB High: ${orb_data.get('orb_high', 0):.2f}
- ORB Low: ${orb_data.get('orb_low', 0):.2f}
- Status: {orb_data.get('breakout_status', 'UNKNOWN')}
"""
        
        message = client.messages.create(
            model="claude-3-haiku-20240307",  # Cost-effective model
            max_tokens=1000,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are an expert futures market analyst. Based on the following market data from opening hours (9:30 AM - 11:30 AM EST), write a comprehensive, professional market analysis.

{context}

Write a detailed analysis that includes:
1. Market Overview: Summary of overall market conditions
2. Key Observations: Notable price movements, volume patterns, and trends
3. Technical Analysis: Support/resistance levels, patterns, and indicators
4. Trading Opportunities: Potential setups and risk considerations
5. Outlook: Short-term expectations for the remainder of the session

Write in a clear, professional tone suitable for active traders. Be specific with price levels and percentages. Keep it concise but comprehensive (300-500 words)."""
                }
            ]
        )
        
        return message.content[0].text.strip()
    
    except Exception as e:
        return f"Error generating analysis with Anthropic: {str(e)}"

def generate_analysis_fallback(market_data: Dict, price_action_data: Dict = None, orb_data: Dict = None) -> str:
    """Generate analysis without LLM (rule-based)."""
    analysis = f"# Market Analysis - {market_data.get('date', 'Today')}\n\n"
    analysis += "## Market Overview\n\n"
    
    # Overall market assessment
    total_change = 0
    bullish_count = 0
    bearish_count = 0
    
    for symbol, data in market_data.get("symbols", {}).items():
        if "error" not in data:
            change_pct = data.get("price_change_pct", 0)
            total_change += change_pct
            if change_pct > 0.1:
                bullish_count += 1
            elif change_pct < -0.1:
                bearish_count += 1
    
    if bullish_count > bearish_count:
        analysis += "The market is showing **bullish** characteristics during the opening hours. "
    elif bearish_count > bullish_count:
        analysis += "The market is showing **bearish** characteristics during the opening hours. "
    else:
        analysis += "The market is showing **mixed** signals during the opening hours. "
    
    # Calculate average change only if we have symbols
    symbols_dict = market_data.get('symbols', {})
    if symbols_dict:
        avg_change = total_change / len(symbols_dict)
        analysis += f"Overall average change: {avg_change:.2f}%.\n\n"
    else:
        analysis += "Overall average change: N/A (no data available).\n\n"
    
    # Individual symbol analysis
    analysis += "## Symbol Analysis\n\n"
    for symbol, data in market_data.get("symbols", {}).items():
        if "error" not in data:
            analysis += f"### {symbol}\n\n"
            analysis += f"- **Current Price**: ${data.get('current_price', 0):.2f}\n"
            analysis += f"- **Change from Open**: ${data.get('price_change', 0):.2f} ({data.get('price_change_pct', 0):+.2f}%)\n"
            analysis += f"- **Range**: ${data.get('low', 0):.2f} - ${data.get('high', 0):.2f}\n"
            analysis += f"- **Volume**: {data.get('volume', 0):,}\n"
            analysis += f"- **Trend**: {data.get('recent_trend', 'NEUTRAL')}\n\n"
    
    # Price action insights
    if price_action_data:
        analysis += "## Price Action Insights\n\n"
        trend = price_action_data.get('trend_analysis', {}).get('trend', 'UNKNOWN')
        analysis += f"- **Current Trend**: {trend}\n"
        
        support_levels = price_action_data.get('support_resistance', {}).get('support_levels', [])
        resistance_levels = price_action_data.get('support_resistance', {}).get('resistance_levels', [])
        
        if support_levels:
            analysis += f"- **Key Support Levels**: {', '.join([f'${s["price"]:.2f}' for s in support_levels[:3]])}\n"
        if resistance_levels:
            analysis += f"- **Key Resistance Levels**: {', '.join([f'${r["price"]:.2f}' for r in resistance_levels[:3]])}\n"
    
    # ORB insights
    if orb_data and "error" not in orb_data:
        analysis += "\n## Opening Range Breakout (ORB) Analysis\n\n"
        analysis += f"- **ORB Range**: ${orb_data.get('orb_low', 0):.2f} - ${orb_data.get('orb_high', 0):.2f}\n"
        analysis += f"- **Status**: {orb_data.get('breakout_status', 'UNKNOWN')}\n"
        if orb_data.get('breakout_direction'):
            analysis += f"- **Breakout Direction**: {orb_data.get('breakout_direction')}\n"
            analysis += f"- **Target**: ${orb_data.get('target_above' if orb_data.get('breakout_direction') == 'BULLISH' else 'target_below', 0):.2f}\n"
    
    analysis += "\n## Trading Outlook\n\n"
    analysis += "Monitor key support and resistance levels for potential breakout or reversal opportunities. "
    analysis += "Volume patterns and price action will be key indicators for the remainder of the session.\n"
    
    return analysis

# ========== ANALYSIS GENERATION ==========

def generate_comprehensive_analysis(symbols: List[str] = None, use_llm: bool = True) -> Dict:
    """
    Generate comprehensive written analysis.
    """
    # Import analysis modules
    try:
        from price_action_orb_analysis import (
            comprehensive_price_action_analysis,
            comprehensive_orb_analysis
        )
    except ImportError:
        print("Warning: price_action_orb_analysis module not found")
        comprehensive_price_action_analysis = None
        comprehensive_orb_analysis = None
    
    # Collect market data
    print("Collecting market data...")
    market_data = collect_market_data(symbols)
    
    # Get price action and ORB data if available
    price_action_data = None
    orb_data = None
    
    if symbols and len(symbols) > 0:
        try:
            # Get data for first symbol for analysis
            main_symbol = symbols[0]
            ticker = yf.Ticker(main_symbol)
            df = ticker.history(period="1d", interval="1m")
            
            if not df.empty and comprehensive_price_action_analysis:
                print("Analyzing price action...")
                price_action_data = comprehensive_price_action_analysis(df)
            
            if not df.empty and comprehensive_orb_analysis:
                print("Analyzing opening range breakout...")
                orb_data = comprehensive_orb_analysis(df, orb_minutes=30)
        except Exception as e:
            print(f"Error in technical analysis: {e}")
    
    # Generate written analysis
    print("Generating written analysis...")
    if use_llm and USE_LLM:
        if OPENAI_API_KEY:
            written_analysis = generate_analysis_with_openai(market_data, price_action_data, orb_data)
        elif ANTHROPIC_API_KEY:
            written_analysis = generate_analysis_with_anthropic(market_data, price_action_data, orb_data)
        else:
            written_analysis = generate_analysis_fallback(market_data, price_action_data, orb_data)
    else:
        written_analysis = generate_analysis_fallback(market_data, price_action_data, orb_data)
    
    # Compile complete analysis
    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "date": get_today_date_str(),
        "time_est": datetime.now(get_market_timezone()).strftime("%H:%M:%S"),
        "market_data": market_data,
        "price_action_analysis": price_action_data,
        "orb_analysis": orb_data,
        "written_analysis": written_analysis,
        "analysis_type": "LLM" if (use_llm and USE_LLM) else "RULE_BASED",
    }
    
    return analysis

# ========== STORAGE ==========

def save_daily_analysis(analysis: Dict) -> str:
    """Save analysis to daily log file."""
    date_str = analysis.get("date", get_today_date_str())
    filename = ANALYSIS_DIR / f"analysis_{date_str}.json"
    
    # Load existing analysis if file exists
    existing_analyses = []
    if filename.exists():
        try:
            with open(filename, 'r') as f:
                existing_analyses = json.load(f)
        except:
            existing_analyses = []
    
    # Append new analysis
    existing_analyses.append(analysis)
    
    # Save
    with open(filename, 'w') as f:
        json.dump(existing_analyses, f, indent=2, default=str)
    
    return str(filename)

def get_daily_analyses(date_str: str = None) -> List[Dict]:
    """Get all analyses for a specific date."""
    if date_str is None:
        date_str = get_today_date_str()
    
    filename = ANALYSIS_DIR / f"analysis_{date_str}.json"
    
    if not filename.exists():
        return []
    
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return []

def get_all_analysis_dates() -> List[str]:
    """Get list of all dates with saved analyses."""
    dates = []
    for file in ANALYSIS_DIR.glob("analysis_*.json"):
        date_str = file.stem.replace("analysis_", "")
        dates.append(date_str)
    return sorted(dates, reverse=True)

# ========== AUTOMATED ANALYSIS ==========

def run_automated_analysis(symbols: List[str] = None, force: bool = False) -> Optional[Dict]:
    """
    Run automated analysis if during analysis hours or if forced.
    Returns analysis dict if generated, None otherwise.
    """
    if not force and not is_during_analysis_hours():
        print(f"Not during analysis hours (9:30 AM - 11:30 AM EST). Current time: {datetime.now(get_market_timezone())}")
        return None
    
    print("Running automated market analysis...")
    analysis = generate_comprehensive_analysis(symbols, use_llm=USE_LLM)
    
    # Save to daily log
    filename = save_daily_analysis(analysis)
    print(f"Analysis saved to: {filename}")
    
    return analysis

# ========== TEST FUNCTION ==========

if __name__ == "__main__":
    print("=" * 60)
    print("DAILY ANALYSIS LOGGER TEST")
    print("=" * 60)
    
    # Test analysis generation
    print("\nGenerating test analysis...")
    analysis = generate_comprehensive_analysis(
        symbols=["ES=F", "NQ=F", "GC=F"],
        use_llm=False  # Use fallback for testing
    )
    
    print("\nAnalysis generated!")
    print(f"Date: {analysis['date']}")
    print(f"Time: {analysis['time_est']}")
    print(f"\nWritten Analysis Preview:")
    print("-" * 60)
    print(analysis['written_analysis'][:500] + "...")
    print("-" * 60)
    
    # Save test analysis
    filename = save_daily_analysis(analysis)
    print(f"\nAnalysis saved to: {filename}")
    
    # Test retrieval
    retrieved = get_daily_analyses()
    print(f"\nRetrieved {len(retrieved)} analyses for today")


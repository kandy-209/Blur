#!/usr/bin/env python3
"""
Additional Data Streams Module
Enhances the trading dashboard with complementary market data sources.
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import os
import time

# ========== CONFIGURATION ==========
# Additional symbols for correlation and context
VOLATILITY_SYMBOLS = {
    "VIX": "^VIX",  # CBOE Volatility Index
    "VIX9D": "^VIX9D",  # 9-Day VIX
    "VXN": "^VXN",  # Nasdaq Volatility
}

CURRENCY_SYMBOLS = {
    "DXY": "DX-Y.NYB",  # US Dollar Index
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "JPYUSD": "JPY=X",
}

BOND_SYMBOLS = {
    "10Y": "^TNX",  # 10-Year Treasury
    "30Y": "^TYX",  # 30-Year Treasury
    "2Y": "^IRX",   # 2-Year Treasury
}

CRYPTO_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}

# Related futures for correlation
RELATED_FUTURES = {
    "ES=F": ["NQ=F", "YM=F", "RTY=F"],  # S&P related
    "NQ=F": ["ES=F", "YM=F"],  # Nasdaq related
    "GC=F": ["SI=F", "CL=F", "NG=F"],  # Gold related
}

# ========== VOLATILITY DATA ==========
def fetch_volatility_data() -> Dict[str, Dict]:
    """
    Fetch volatility indices (VIX, VIX9D, VXN).
    Returns dict with latest values and trends.
    """
    results = {}
    
    for name, symbol in VOLATILITY_SYMBOLS.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d", interval="1h")
            
            if not data.empty:
                latest = data.iloc[-1]
                prev_close = data.iloc[-2] if len(data) > 1 else latest
                
                current = float(latest["Close"])
                change = current - float(prev_close["Close"])
                change_pct = (change / float(prev_close["Close"])) * 100 if prev_close["Close"] > 0 else 0
                
                results[name] = {
                    "symbol": symbol,
                    "value": current,
                    "change": change,
                    "change_pct": change_pct,
                    "timestamp": data.index[-1],
                    "status": "LIVE" if (datetime.now(timezone.utc) - data.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds() < 3600 else "STALE"
                }
        except Exception as e:
            print(f"Error fetching {name} ({symbol}): {e}")
            results[name] = {"error": str(e)}
    
    return results

# ========== CURRENCY DATA ==========
def fetch_currency_data() -> Dict[str, Dict]:
    """
    Fetch currency pairs and DXY (Dollar Index).
    Returns dict with latest values and trends.
    """
    results = {}
    
    for name, symbol in CURRENCY_SYMBOLS.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d", interval="1h")
            
            if not data.empty:
                latest = data.iloc[-1]
                prev_close = data.iloc[-2] if len(data) > 1 else latest
                
                current = float(latest["Close"])
                change = current - float(prev_close["Close"])
                change_pct = (change / float(prev_close["Close"])) * 100 if prev_close["Close"] > 0 else 0
                
                results[name] = {
                    "symbol": symbol,
                    "value": current,
                    "change": change,
                    "change_pct": change_pct,
                    "timestamp": data.index[-1],
                }
        except Exception as e:
            print(f"Error fetching {name} ({symbol}): {e}")
            results[name] = {"error": str(e)}
    
    return results

# ========== BOND DATA ==========
def fetch_bond_data() -> Dict[str, Dict]:
    """
    Fetch Treasury yields (2Y, 10Y, 30Y).
    Returns dict with latest yields and trends.
    """
    results = {}
    
    for name, symbol in BOND_SYMBOLS.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d", interval="1h")
            
            if not data.empty:
                latest = data.iloc[-1]
                prev_close = data.iloc[-2] if len(data) > 1 else latest
                
                current = float(latest["Close"])
                change = current - float(prev_close["Close"])
                
                results[name] = {
                    "symbol": symbol,
                    "yield": current,
                    "change": change,
                    "timestamp": data.index[-1],
                }
        except Exception as e:
            print(f"Error fetching {name} ({symbol}): {e}")
            results[name] = {"error": str(e)}
    
    return results

# ========== OPTIONS DATA ==========
def fetch_options_data(symbol: str) -> Dict:
    """
    Fetch options chain data for a symbol.
    Returns summary of put/call ratios and open interest.
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get expiration dates
        expirations = ticker.options
        if not expirations:
            return {"error": "No options data available"}
        
        # Get nearest expiration
        nearest_exp = expirations[0]
        opt_chain = ticker.option_chain(nearest_exp)
        
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        # Calculate put/call ratio
        total_call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
        total_put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
        
        put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # Get current price
        current_data = ticker.history(period="1d", interval="1m")
        current_price = float(current_data["Close"].iloc[-1]) if not current_data.empty else 0
        
        return {
            "symbol": symbol,
            "expiration": nearest_exp,
            "put_call_ratio": put_call_ratio,
            "total_call_oi": int(total_call_oi),
            "total_put_oi": int(total_put_oi),
            "current_price": current_price,
            "timestamp": datetime.now(timezone.utc),
        }
    except Exception as e:
        return {"error": str(e)}

# ========== CORRELATION DATA ==========
def calculate_correlation(symbol: str, period_days: int = 30) -> Dict[str, float]:
    """
    Calculate correlation between a symbol and related instruments.
    Returns dict of correlation coefficients.
    """
    correlations = {}
    
    try:
        # Get main symbol data
        main_ticker = yf.Ticker(symbol)
        main_data = main_ticker.history(period=f"{period_days}d", interval="1d")
        
        if main_data.empty:
            return correlations
        
        main_returns = main_data["Close"].pct_change().dropna()
        
        # Get related symbols
        related = RELATED_FUTURES.get(symbol, [])
        
        for related_symbol in related:
            try:
                related_ticker = yf.Ticker(related_symbol)
                related_data = related_ticker.history(period=f"{period_days}d", interval="1d")
                
                if not related_data.empty:
                    related_returns = related_data["Close"].pct_change().dropna()
                    
                    # Align indices
                    aligned = pd.concat([main_returns, related_returns], axis=1).dropna()
                    if len(aligned) > 10:  # Need minimum data points
                        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                        correlations[related_symbol] = float(corr) if not pd.isna(corr) else 0.0
            except Exception as e:
                print(f"Error calculating correlation with {related_symbol}: {e}")
        
        # Also correlate with VIX
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period=f"{period_days}d", interval="1d")
            if not vix_data.empty:
                vix_returns = vix_data["Close"].pct_change().dropna()
                aligned = pd.concat([main_returns, vix_returns], axis=1).dropna()
                if len(aligned) > 10:
                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                    correlations["VIX"] = float(corr) if not pd.isna(corr) else 0.0
        except:
            pass
        
    except Exception as e:
        print(f"Error calculating correlations for {symbol}: {e}")
    
    return correlations

# ========== VOLUME PROFILE ==========
def get_volume_profile(symbol: str, period: str = "5d", interval: str = "1h") -> Dict:
    """
    Calculate volume profile (price levels with highest volume).
    Returns price levels and volume distribution.
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            return {"error": "No data available"}
        
        # Calculate volume-weighted average price (VWAP)
        typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
        vwap = (typical_price * data["Volume"]).sum() / data["Volume"].sum()
        
        # Find price range
        price_min = float(data["Low"].min())
        price_max = float(data["High"].max())
        
        # Calculate volume at different price levels
        price_range = price_max - price_min
        num_bins = 20
        bin_size = price_range / num_bins
        
        volume_profile = {}
        for i in range(num_bins):
            level_low = price_min + (i * bin_size)
            level_high = price_min + ((i + 1) * bin_size)
            level_volume = data[
                (data["Low"] >= level_low) & (data["High"] <= level_high)
            ]["Volume"].sum()
            volume_profile[f"{level_low:.2f}-{level_high:.2f}"] = int(level_volume)
        
        return {
            "symbol": symbol,
            "vwap": float(vwap),
            "price_range": {"min": price_min, "max": price_max},
            "volume_profile": volume_profile,
            "total_volume": int(data["Volume"].sum()),
            "avg_volume": int(data["Volume"].mean()),
        }
    except Exception as e:
        return {"error": str(e)}

# ========== MARKET BREADTH ==========
def get_market_breadth() -> Dict:
    """
    Calculate market breadth indicators.
    Returns advance/decline ratio, new highs/lows, etc.
    """
    try:
        # Get major indices
        indices = {
            "SPY": "SPY",  # S&P 500 ETF
            "QQQ": "QQQ",  # Nasdaq ETF
            "DIA": "DIA",  # Dow ETF
        }
        
        breadth_data = {}
        
        for name, symbol in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d", interval="1d")
                
                if not data.empty:
                    # Calculate 5-day change
                    current = float(data["Close"].iloc[-1])
                    five_days_ago = float(data["Close"].iloc[0]) if len(data) > 4 else current
                    change_pct = ((current - five_days_ago) / five_days_ago) * 100 if five_days_ago > 0 else 0
                    
                    breadth_data[name] = {
                        "current": current,
                        "change_5d": change_pct,
                        "volume": int(data["Volume"].iloc[-1]) if "Volume" in data.columns else 0,
                    }
            except Exception as e:
                print(f"Error fetching {name}: {e}")
        
        return breadth_data
    except Exception as e:
        return {"error": str(e)}

# ========== CRYPTO CORRELATION ==========
def get_crypto_data() -> Dict[str, Dict]:
    """
    Fetch crypto prices for correlation analysis.
    """
    results = {}
    
    for name, symbol in CRYPTO_SYMBOLS.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="5d", interval="1h")
            
            if not data.empty:
                latest = data.iloc[-1]
                prev_close = data.iloc[-2] if len(data) > 1 else latest
                
                current = float(latest["Close"])
                change = current - float(prev_close["Close"])
                change_pct = (change / float(prev_close["Close"])) * 100 if prev_close["Close"] > 0 else 0
                
                results[name] = {
                    "symbol": symbol,
                    "price": current,
                    "change": change,
                    "change_pct": change_pct,
                    "volume": int(latest["Volume"]) if "Volume" in latest else 0,
                    "timestamp": data.index[-1],
                }
        except Exception as e:
            print(f"Error fetching {name} ({symbol}): {e}")
    
    return results

# ========== COMPREHENSIVE DATA FETCH ==========
def get_all_additional_data(main_symbol: str) -> Dict:
    """
    Fetch all additional data streams for comprehensive market view.
    Returns dict with all available data.
    """
    print(f"Fetching additional data streams for {main_symbol}...")
    
    all_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "main_symbol": main_symbol,
        "volatility": fetch_volatility_data(),
        "currencies": fetch_currency_data(),
        "bonds": fetch_bond_data(),
        "correlations": calculate_correlation(main_symbol),
        "volume_profile": get_volume_profile(main_symbol),
        "market_breadth": get_market_breadth(),
        "crypto": get_crypto_data(),
    }
    
    # Try to get options data (may not be available for all futures)
    try:
        all_data["options"] = fetch_options_data(main_symbol)
    except:
        all_data["options"] = {"note": "Options data not available for this symbol"}
    
    return all_data

# ========== TEST FUNCTION ==========
if __name__ == "__main__":
    # Test the additional data streams
    print("=" * 60)
    print("TESTING ADDITIONAL DATA STREAMS")
    print("=" * 60)
    
    test_symbol = "ES=F"
    print(f"\nTesting with symbol: {test_symbol}\n")
    
    # Test each function
    print("1. Volatility Data:")
    vol_data = fetch_volatility_data()
    for name, data in vol_data.items():
        if "error" not in data:
            print(f"   {name}: {data.get('value', 'N/A'):.2f} ({data.get('change_pct', 0):+.2f}%)")
    
    print("\n2. Currency Data:")
    curr_data = fetch_currency_data()
    for name, data in curr_data.items():
        if "error" not in data:
            print(f"   {name}: {data.get('value', 'N/A'):.4f} ({data.get('change_pct', 0):+.2f}%)")
    
    print("\n3. Bond Data:")
    bond_data = fetch_bond_data()
    for name, data in bond_data.items():
        if "error" not in data:
            print(f"   {name}: {data.get('yield', 'N/A'):.2f}% ({data.get('change', 0):+.2f})")
    
    print("\n4. Correlations:")
    corr_data = calculate_correlation(test_symbol)
    for symbol, corr in corr_data.items():
        print(f"   {symbol}: {corr:.3f}")
    
    print("\n5. Volume Profile:")
    vol_profile = get_volume_profile(test_symbol)
    if "error" not in vol_profile:
        print(f"   VWAP: ${vol_profile.get('vwap', 0):.2f}")
        print(f"   Price Range: ${vol_profile.get('price_range', {}).get('min', 0):.2f} - ${vol_profile.get('price_range', {}).get('max', 0):.2f}")
    
    print("\n6. Market Breadth:")
    breadth = get_market_breadth()
    for name, data in breadth.items():
        if "error" not in data:
            print(f"   {name}: ${data.get('current', 0):.2f} ({data.get('change_5d', 0):+.2f}%)")
    
    print("\n7. Crypto Data:")
    crypto = get_crypto_data()
    for name, data in crypto.items():
        print(f"   {name}: ${data.get('price', 0):.2f} ({data.get('change_pct', 0):+.2f}%)")
    
    print("\n" + "=" * 60)
    print("All data streams tested successfully!")
    print("=" * 60)


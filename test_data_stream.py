#!/usr/bin/env python3
"""
Test script to verify the data stream is live and active.
Checks connectivity, data freshness, and symbol availability.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
import sys
import time

# Symbols to test
SYMBOLS = ["ES=F", "NQ=F", "GC=F"]  # S&P 500, Nasdaq, Gold futures

def check_symbol(symbol: str, timeout: int = 10) -> dict:
    """
    Check if a symbol's data stream is live and active.
    
    Returns:
        dict with status, latest_price, latest_time, data_age_minutes, error
    """
    result = {
        "symbol": symbol,
        "status": "UNKNOWN",
        "latest_price": None,
        "latest_time": None,
        "data_age_minutes": None,
        "error": None,
        "data_points": 0
    }
    
    try:
        print(f"  Fetching {symbol}...", end=" ", flush=True)
        
        # Fetch recent data (last 1 day, 1-minute intervals)
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m", timeout=timeout)
        
        if data.empty:
            result["status"] = "NO_DATA"
            result["error"] = "No data returned"
            print("[X] NO DATA")
            return result
        
        # Get the latest data point
        latest = data.iloc[-1]
        latest_price = float(latest["Close"])
        latest_time = data.index[-1]
        
        # Convert to UTC if needed
        if latest_time.tzinfo is None:
            latest_time = latest_time.tz_localize("UTC")
        else:
            latest_time = latest_time.astimezone(timezone.utc)
        
        # Calculate data age
        now_utc = datetime.now(timezone.utc)
        data_age = (now_utc - latest_time).total_seconds() / 60  # minutes
        
        result["latest_price"] = latest_price
        result["latest_time"] = latest_time
        result["data_age_minutes"] = data_age
        result["data_points"] = len(data)
        
        # Determine status based on data freshness
        if data_age < 5:
            result["status"] = "LIVE"
            print(f"[OK] LIVE (Price: ${latest_price:.2f}, Age: {data_age:.1f} min)")
        elif data_age < 15:
            result["status"] = "ACTIVE"
            print(f"[!] ACTIVE (Price: ${latest_price:.2f}, Age: {data_age:.1f} min)")
        elif data_age < 60:
            result["status"] = "STALE"
            print(f"[!] STALE (Price: ${latest_price:.2f}, Age: {data_age:.1f} min)")
        else:
            result["status"] = "INACTIVE"
            print(f"[X] INACTIVE (Price: ${latest_price:.2f}, Age: {data_age:.1f} min)")
        
        return result
        
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        print(f"[X] ERROR: {str(e)}")
        return result

def test_connection():
    """Test basic connectivity to yfinance."""
    print("Testing yfinance connectivity...")
    try:
        # Try fetching a simple ticker
        test_ticker = yf.Ticker("AAPL")
        test_data = test_ticker.history(period="1d", timeout=5)
        if not test_data.empty:
            print("[OK] yfinance connection: OK")
            return True
        else:
            print("[!] yfinance connection: No data returned")
            return False
    except Exception as e:
        print(f"[X] yfinance connection: FAILED - {str(e)}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("DATA STREAM STATUS CHECK")
    print("=" * 60)
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    # Test basic connectivity
    if not test_connection():
        print("\n[X] Cannot connect to yfinance. Check your internet connection.")
        sys.exit(1)
    
    print()
    print("Testing futures symbols:")
    print("-" * 60)
    
    results = []
    for symbol in SYMBOLS:
        result = check_symbol(symbol)
        results.append(result)
        time.sleep(0.5)  # Small delay between requests
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Count statuses
    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Overall status
    if all(r["status"] == "LIVE" for r in results):
        overall_status = "[OK] ALL STREAMS LIVE"
    elif all(r["status"] in ["LIVE", "ACTIVE"] for r in results):
        overall_status = "[!] SOME STREAMS ACTIVE"
    elif any(r["status"] == "LIVE" for r in results):
        overall_status = "[!] MIXED STATUS"
    else:
        overall_status = "[X] STREAMS INACTIVE"
    
    print(f"Overall Status: {overall_status}")
    print()
    
    # Detailed results
    print("Detailed Results:")
    for r in results:
        status_icon = {
            "LIVE": "[OK]",
            "ACTIVE": "[!]",
            "STALE": "[!]",
            "INACTIVE": "[X]",
            "NO_DATA": "[X]",
            "ERROR": "[X]",
            "UNKNOWN": "[?]"
        }.get(r["status"], "[?]")
        
        print(f"  {status_icon} {r['symbol']}: {r['status']}")
        if r["latest_price"]:
            print(f"     Price: ${r['latest_price']:.2f}")
        if r["latest_time"]:
            print(f"     Latest: {r['latest_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if r["data_age_minutes"] is not None:
            print(f"     Age: {r['data_age_minutes']:.1f} minutes")
        if r["data_points"] > 0:
            print(f"     Data Points: {r['data_points']}")
        if r["error"]:
            print(f"     Error: {r['error']}")
        print()
    
    # Recommendations
    print("Recommendations:")
    if any(r["status"] == "ERROR" for r in results):
        print("  - Check internet connection")
        print("  - Verify yfinance is up to date: pip install --upgrade yfinance")
    if any(r["status"] in ["STALE", "INACTIVE"] for r in results):
        print("  - Market may be closed (check trading hours)")
        print("  - Data may be delayed during off-hours")
    if all(r["status"] == "LIVE" for r in results):
        print("  - All data streams are live and active! [OK]")
    
    print()
    print("=" * 60)
    
    # Exit code based on status
    if all(r["status"] in ["LIVE", "ACTIVE"] for r in results):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Some issues detected

if __name__ == "__main__":
    main()


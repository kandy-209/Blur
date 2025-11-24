#!/usr/bin/env python3
"""
Advanced API Manager for Trading System
Integrates multiple premium and free APIs for comprehensive market data
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass
import json

# ========== API CONFIGURATION ==========

# API Keys (set via environment variables)
API_KEYS = {
    "ALPHA_VANTAGE": os.environ.get("ALPHA_VANTAGE_KEY", ""),
    "POLYGON": os.environ.get("POLYGON_API_KEY", ""),
    "IEX_CLOUD": os.environ.get("IEX_CLOUD_KEY", ""),
    "FINNHUB": os.environ.get("FINNHUB_API_KEY", ""),
    "TWELVE_DATA": os.environ.get("TWELVE_DATA_API_KEY", ""),
    "QUANDL": os.environ.get("QUANDL_API_KEY", ""),
    "NEWS_API": os.environ.get("NEWS_API_KEY", ""),
    "FMP": os.environ.get("FMP_API_KEY", ""),  # Financial Modeling Prep
}

# API Rate Limits (requests per minute)
RATE_LIMITS = {
    "ALPHA_VANTAGE": 5,  # Free tier: 5 calls/minute
    "POLYGON": 5,  # Depends on plan
    "IEX_CLOUD": 100,  # Depends on plan
    "FINNHUB": 60,  # Free tier: 60 calls/minute
    "TWELVE_DATA": 8,  # Free tier: 8 calls/minute
    "QUANDL": 50,  # Depends on plan
    "FMP": 250,  # Free tier: 250 calls/day
}

# ========== API PROVIDERS ==========

@dataclass
class APIProvider:
    """API provider configuration."""
    name: str
    base_url: str
    api_key: str
    rate_limit: int
    has_real_time: bool
    has_historical: bool
    has_options: bool
    has_fundamentals: bool

# ========== ALPHA VANTAGE API ==========

class AlphaVantageAPI:
    """Alpha Vantage API integration."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEYS["ALPHA_VANTAGE"]
        self.base_url = "https://www.alphavantage.co/query"
        self.last_call_time = 0
        self.min_interval = 12  # 5 calls per minute = 12 seconds between calls
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call_time = time.time()
    
    def get_intraday_data(self, symbol: str, interval: str = "1min", outputsize: str = "full") -> pd.DataFrame:
        """Get intraday data."""
        if not self.api_key:
            return pd.DataFrame()
        
        self._rate_limit()
        
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol.replace("=F", ""),  # Remove futures suffix
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": outputsize,
            "datatype": "json"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if "Error Message" in data or "Note" in data:
                return pd.DataFrame()
            
            time_series_key = f"Time Series ({interval})"
            if time_series_key not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            return df
        except Exception as e:
            print(f"Alpha Vantage API error: {e}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol: str, indicator: str, **kwargs) -> pd.DataFrame:
        """Get technical indicators."""
        if not self.api_key:
            return pd.DataFrame()
        
        self._rate_limit()
        
        params = {
            "function": indicator,
            "symbol": symbol.replace("=F", ""),
            "apikey": self.api_key,
            "datatype": "json",
            **kwargs
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if "Error Message" in data or "Note" in data:
                return pd.DataFrame()
            
            # Extract time series data
            for key in data.keys():
                if "Time Series" in key or "Technical Analysis" in key:
                    df = pd.DataFrame.from_dict(data[key], orient="index")
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    return df.astype(float)
            
            return pd.DataFrame()
        except Exception as e:
            print(f"Alpha Vantage indicator error: {e}")
            return pd.DataFrame()

# ========== POLYGON.IO API ==========

class PolygonAPI:
    """Polygon.io API integration - Premium real-time data."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEYS["POLYGON"]
        self.base_url = "https://api.polygon.io"
    
    def get_aggregates(self, symbol: str, multiplier: int = 1, timespan: str = "minute",
                      from_date: str = None, to_date: str = None) -> pd.DataFrame:
        """Get aggregated bars."""
        if not self.api_key:
            return pd.DataFrame()
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"apiKey": self.api_key, "adjusted": "true", "sort": "asc"}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") != "OK" or "results" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["results"])
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(columns={
                "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
            })
            df = df.set_index("timestamp")
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            
            return df
        except Exception as e:
            print(f"Polygon API error: {e}")
            return pd.DataFrame()
    
    def get_ticker_details(self, symbol: str) -> Dict:
        """Get ticker details and fundamentals."""
        if not self.api_key:
            return {}
        
        url = f"{self.base_url}/v3/reference/tickers/{symbol}"
        params = {"apiKey": self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            return data.get("results", {})
        except Exception as e:
            print(f"Polygon ticker details error: {e}")
            return {}

# ========== FINNHUB API ==========

class FinnhubAPI:
    """Finnhub API - Free tier with good limits."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEYS["FINNHUB"]
        self.base_url = "https://finnhub.io/api/v1"
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote."""
        if not self.api_key:
            return {}
        
        url = f"{self.base_url}/quote"
        params = {"symbol": symbol, "token": self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=5)
            return response.json()
        except Exception as e:
            print(f"Finnhub quote error: {e}")
            return {}
    
    def get_candles(self, symbol: str, resolution: str = "1", 
                   from_timestamp: int = None, to_timestamp: int = None) -> pd.DataFrame:
        """Get candle data."""
        if not self.api_key:
            return pd.DataFrame()
        
        if not from_timestamp:
            from_timestamp = int((datetime.now() - timedelta(days=1)).timestamp())
        if not to_timestamp:
            to_timestamp = int(datetime.now().timestamp())
        
        url = f"{self.base_url}/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": from_timestamp,
            "to": to_timestamp,
            "token": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("s") != "ok" or "c" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame({
                "Open": data["o"],
                "High": data["h"],
                "Low": data["l"],
                "Close": data["c"],
                "Volume": data["v"]
            })
            df.index = pd.to_datetime(data["t"], unit="s")
            
            return df
        except Exception as e:
            print(f"Finnhub candles error: {e}")
            return pd.DataFrame()
    
    def get_news(self, symbol: str, from_date: str = None, to_date: str = None) -> List[Dict]:
        """Get company news."""
        if not self.api_key:
            return []
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/company-news"
        params = {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.json()[:10]  # Return top 10
        except Exception as e:
            print(f"Finnhub news error: {e}")
            return []

# ========== TWELVE DATA API ==========

class TwelveDataAPI:
    """Twelve Data API - Good for real-time and historical."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEYS["TWELVE_DATA"]
        self.base_url = "https://api.twelvedata.com"
    
    def get_time_series(self, symbol: str, interval: str = "1min",
                       outputsize: int = 5000) -> pd.DataFrame:
        """Get time series data."""
        if not self.api_key:
            return pd.DataFrame()
        
        url = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": outputsize,
            "format": "json"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("status") != "ok" or "values" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["values"])
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume", "datetime": "timestamp"
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            df = df.sort_index()
            
            return df
        except Exception as e:
            print(f"Twelve Data API error: {e}")
            return pd.DataFrame()

# ========== FINANCIAL MODELING PREP API ==========

class FMPAPI:
    """Financial Modeling Prep API - Comprehensive financial data."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or API_KEYS["FMP"]
        self.base_url = "https://financialmodelingprep.com/api/v3"
    
    def get_historical_data(self, symbol: str, from_date: str = None, to_date: str = None) -> pd.DataFrame:
        """Get historical data."""
        if not self.api_key:
            return pd.DataFrame()
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/historical-chart/1min/{symbol}"
        params = {
            "from": from_date,
            "to": to_date,
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if not data or isinstance(data, dict) and data.get("Error"):
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            
            return df
        except Exception as e:
            print(f"FMP API error: {e}")
            return pd.DataFrame()
    
    def get_key_metrics(self, symbol: str) -> Dict:
        """Get key financial metrics."""
        if not self.api_key:
            return {}
        
        url = f"{self.base_url}/key-metrics/{symbol}"
        params = {"apikey": self.api_key, "limit": 1}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            return data[0] if data else {}
        except Exception as e:
            print(f"FMP key metrics error: {e}")
            return {}

# ========== UNIFIED API MANAGER ==========

class APIManager:
    """Unified API manager with fallback mechanisms."""
    
    def __init__(self):
        self.alpha_vantage = AlphaVantageAPI()
        self.polygon = PolygonAPI()
        self.finnhub = FinnhubAPI()
        self.twelve_data = TwelveDataAPI()
        self.fmp = FMPAPI()
        
        # Priority order (best to worst)
        self.data_apis = [
            ("polygon", self.polygon),
            ("finnhub", self.finnhub),
            ("twelve_data", self.twelve_data),
            ("alpha_vantage", self.alpha_vantage),
            ("fmp", self.fmp),
        ]
    
    def get_market_data(self, symbol: str, interval: str = "1min", 
                       period: str = "1d", fallback_to_yfinance: bool = True) -> pd.DataFrame:
        """Get market data with automatic fallback."""
        # Try APIs in priority order
        for api_name, api in self.data_apis:
            try:
                if api_name == "polygon" and hasattr(api, "get_aggregates"):
                    df = api.get_aggregates(symbol, multiplier=1, timespan="minute")
                    if not df.empty:
                        print(f"✓ Data from Polygon.io")
                        return df
                
                elif api_name == "finnhub" and hasattr(api, "get_candles"):
                    resolution = "1" if interval == "1min" else "5" if interval == "5min" else "60"
                    df = api.get_candles(symbol, resolution=resolution)
                    if not df.empty:
                        print(f"✓ Data from Finnhub")
                        return df
                
                elif api_name == "twelve_data" and hasattr(api, "get_time_series"):
                    df = api.get_time_series(symbol, interval=interval)
                    if not df.empty:
                        print(f"✓ Data from Twelve Data")
                        return df
                
                elif api_name == "alpha_vantage" and hasattr(api, "get_intraday_data"):
                    df = api.get_intraday_data(symbol, interval=interval)
                    if not df.empty:
                        print(f"✓ Data from Alpha Vantage")
                        return df
                
                elif api_name == "fmp" and hasattr(api, "get_historical_data"):
                    df = api.get_historical_data(symbol)
                    if not df.empty:
                        print(f"✓ Data from Financial Modeling Prep")
                        return df
            except Exception as e:
                print(f"  {api_name} failed: {e}")
                continue
        
        # Fallback to yfinance
        if fallback_to_yfinance:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                if not df.empty:
                    print(f"✓ Data from Yahoo Finance (fallback)")
                    return df
            except Exception as e:
                print(f"  Yahoo Finance fallback failed: {e}")
        
        return pd.DataFrame()
    
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote."""
        # Try Finnhub first (good free tier)
        if self.finnhub.api_key:
            quote = self.finnhub.get_quote(symbol)
            if quote and "c" in quote:
                return quote
        
        # Fallback to other APIs or return empty
        return {}
    
    def get_news(self, symbol: str) -> List[Dict]:
        """Get news for symbol."""
        news = []
        
        # Try Finnhub
        if self.finnhub.api_key:
            news.extend(self.finnhub.get_news(symbol))
        
        return news[:10]  # Return top 10
    
    def get_technical_indicator(self, symbol: str, indicator: str, **kwargs) -> pd.DataFrame:
        """Get technical indicator."""
        # Alpha Vantage has good technical indicators
        if self.alpha_vantage.api_key:
            df = self.alpha_vantage.get_technical_indicators(symbol, indicator, **kwargs)
            if not df.empty:
                return df
        
        return pd.DataFrame()
    
    def get_fundamentals(self, symbol: str) -> Dict:
        """Get fundamental data."""
        fundamentals = {}
        
        # Try Polygon
        if self.polygon.api_key:
            details = self.polygon.get_ticker_details(symbol)
            if details:
                fundamentals.update(details)
        
        # Try FMP
        if self.fmp.api_key:
            metrics = self.fmp.get_key_metrics(symbol)
            if metrics:
                fundamentals.update(metrics)
        
        return fundamentals

# ========== API SETUP HELPER ==========

def setup_api_keys():
    """Interactive API key setup."""
    print("=" * 70)
    print("API KEY SETUP")
    print("=" * 70)
    print("\nRecommended APIs (in order of quality):")
    print("\n1. Polygon.io (Best for real-time)")
    print("   - Sign up: https://polygon.io/")
    print("   - Free tier: Limited")
    print("   - Paid: $29-199/month")
    
    print("\n2. Finnhub (Best free tier)")
    print("   - Sign up: https://finnhub.io/")
    print("   - Free tier: 60 calls/minute")
    print("   - Good for: Quotes, candles, news")
    
    print("\n3. Twelve Data")
    print("   - Sign up: https://twelvedata.com/")
    print("   - Free tier: 8 calls/minute")
    print("   - Good for: Historical data")
    
    print("\n4. Alpha Vantage")
    print("   - Sign up: https://www.alphavantage.co/")
    print("   - Free tier: 5 calls/minute")
    print("   - Good for: Technical indicators")
    
    print("\n5. Financial Modeling Prep")
    print("   - Sign up: https://site.financialmodelingprep.com/")
    print("   - Free tier: 250 calls/day")
    print("   - Good for: Fundamentals")
    
    print("\n" + "=" * 70)
    print("Set API keys as environment variables:")
    print("=" * 70)
    print("export POLYGON_API_KEY='your-key'")
    print("export FINNHUB_API_KEY='your-key'")
    print("export TWELVE_DATA_API_KEY='your-key'")
    print("export ALPHA_VANTAGE_KEY='your-key'")
    print("export FMP_API_KEY='your-key'")
    print("\nOr create a .env file with these variables.")

# ========== TEST FUNCTION ==========

if __name__ == "__main__":
    print("=" * 70)
    print("API MANAGER TEST")
    print("=" * 70)
    
    manager = APIManager()
    
    # Test data retrieval
    print("\nTesting market data retrieval...")
    symbol = "ES=F"
    
    df = manager.get_market_data(symbol, interval="1min", period="1d")
    
    if not df.empty:
        print(f"\n✓ Successfully retrieved {len(df)} data points")
        print(f"  Latest: {df.index[-1]}")
        print(f"  Close: ${df['Close'].iloc[-1]:.2f}")
    else:
        print("\n✗ No data retrieved (may need API keys)")
        print("\nRun setup_api_keys() for instructions")
    
    # Test real-time quote
    print("\nTesting real-time quote...")
    quote = manager.get_real_time_quote(symbol)
    if quote:
        print(f"✓ Quote retrieved: {quote}")
    else:
        print("✗ No quote available (may need API keys)")
    
    # Show setup instructions
    print("\n" + "=" * 70)
    setup_api_keys()


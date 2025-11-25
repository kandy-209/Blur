# ---------- IMPORTS ----------
import pandas as pd
import numpy as np
import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import requests
from typing import Dict, List, Optional, Tuple
import pytz

# Optional Dash auth for securing the dashboard
try:
    import dash_auth
    DASH_AUTH_AVAILABLE = True
except ImportError:
    DASH_AUTH_AVAILABLE = False
    print("Warning: dash-auth not available. Basic auth protection disabled.")

# Macro research engine
def _parse_macro_weights(raw: str) -> Dict[str, float]:
    weights = {}
    if not raw:
        return weights
    for chunk in raw.split(","):
        if "=" in chunk:
            key, val = chunk.split("=", 1)
            try:
                weights[key.strip().lower()] = float(val.strip())
            except ValueError:
                continue
    return weights

try:
    from macro_research_algo import MacroResearchEngine
    MACRO_ENGINE_AVAILABLE = True
    macro_engine = MacroResearchEngine(
        lookback=int(os.environ.get("MACRO_LOOKBACK", 90)),
        weights=_parse_macro_weights(os.environ.get("MACRO_WEIGHTS", "")),
        sentiment_weight=float(os.environ.get("MACRO_SENTIMENT_WEIGHT", 0.05)),
    )
except ImportError:
    MACRO_ENGINE_AVAILABLE = False
    macro_engine = None
    print("Warning: macro_research_algo module not available")

# Import indicator education module
try:
    from indicator_education import (
        create_indicator_tooltip,
        create_indicator_link,
        get_indicator_info,
        get_value_interpretation
    )
    INDICATOR_EDUCATION_AVAILABLE = True
except ImportError:
    INDICATOR_EDUCATION_AVAILABLE = False
    print("Warning: indicator_education module not available")

# Import dashboard enhancements
try:
    from dashboard_enhancements import (
        DataExporter,
        get_alert_system,
        get_portfolio_tracker,
        CorrelationAnalyzer
    )
    ENHANCEMENTS_AVAILABLE = True
    data_exporter = DataExporter()
    alert_system = get_alert_system()
    portfolio_tracker = get_portfolio_tracker()
    correlation_analyzer = CorrelationAnalyzer()
except ImportError:
    ENHANCEMENTS_AVAILABLE = False
    print("Warning: dashboard_enhancements module not available")

# Import paper trading system
try:
    from paper_trading import get_paper_trading_system
    PAPER_TRADING_AVAILABLE = True
    paper_trading = get_paper_trading_system(initial_capital=100000.0)
except ImportError:
    PAPER_TRADING_AVAILABLE = False
    paper_trading = None
    print("Warning: paper_trading module not available")

# Advanced technical analysis
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("Warning: pandas-ta not available. Some advanced indicators will be disabled.")

# Machine Learning
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ML predictions will be disabled.")

# Sentiment Analysis
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Warning: Sentiment analysis libraries not available.")

# Economic Data
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("Warning: FRED API not available. Economic indicators will be disabled.")

# Modal and ASGI imports (only needed for deployment)
try:
    import modal
    from asgiref.wsgi import WsgiToAsgi
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    # Create a dummy modal for local testing
    class DummyModal:
        class App:
            def __init__(self, name):
                self.name = name
            def asgi_app(self, image=None):
                def decorator(func):
                    return func
                return decorator
        class Image:
            @staticmethod
            def debian_slim():
                return DummyImage()
    class DummyImage:
        def pip_install(self, packages):
            return self
    modal = DummyModal()

# ---------- MODAL SETUP ----------
if MODAL_AVAILABLE:
    app = modal.App("live_futures_dashboard")
    image = (
        modal.Image.debian_slim()
        .pip_install([
            "dash",
            "pandas",
            "numpy",
            "plotly",
            "yfinance",
            "asgiref",
            "pandas-ta",
            "scikit-learn",
            "requests",
            "textblob",
            "vaderSentiment",
            "beautifulsoup4",
            "fredapi",
        ])
    )
else:
    app = None
    image = None

# ---------- API CONFIGURATION ----------
# Get API keys from environment variables (set these in your deployment)
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# Initialize FRED if available
fred = None
if FRED_AVAILABLE and FRED_API_KEY:
    try:
        fred = Fred(api_key=FRED_API_KEY)
    except:
        fred = None

# ---------- DATA FETCHING ----------
symbols = {"ES=F": "S&P 500", "NQ=F": "Nasdaq", "GC=F": "Gold", "YM=F": "Dow", "CL=F": "Crude Oil"}

# Period and interval mapping
PERIOD_INTERVALS = {
    "1m": ("1d", "1m"),
    "5m": ("5d", "5m"),
    "15m": ("5d", "15m"),
    "1h": ("1mo", "1h"),
    "1d": ("1y", "1d"),
}

def is_market_closed():
    """Check if market is closed (Friday 2pm PST to Sunday 2pm PST)."""
    try:
        # Get current time in PST
        pst = pytz.timezone('America/Los_Angeles')
        now_pst = datetime.now(pst)
        weekday = now_pst.weekday()  # 0=Monday, 4=Friday, 6=Sunday
        hour = now_pst.hour
        
        # Market is closed:
        # - Friday after 2pm PST (14:00)
        # - All day Saturday
        # - Sunday before 2pm PST (14:00)
        if weekday == 4 and hour >= 14:  # Friday 2pm+
            return True
        elif weekday == 5:  # Saturday
            return True
        elif weekday == 6 and hour < 14:  # Sunday before 2pm
            return True
        return False
    except Exception:
        # If timezone fails, assume market might be closed
        return False

def get_last_week_dates():
    """Get start and end dates for the previous trading week."""
    try:
        pst = pytz.timezone('America/Los_Angeles')
        now_pst = datetime.now(pst)
        
        # Calculate last Monday (start of last week)
        days_since_monday = now_pst.weekday()
        last_monday = now_pst - timedelta(days=days_since_monday + 7)
        last_monday = last_monday.replace(hour=9, minute=30, second=0, microsecond=0)  # Market open time
        
        # Calculate last Friday (end of last week)
        last_friday = last_monday + timedelta(days=4)
        last_friday = last_friday.replace(hour=16, minute=0, second=0, microsecond=0)  # Market close time
        
        return last_monday, last_friday
    except Exception:
        # Fallback: use 7 days ago to 2 days ago
        end_date = datetime.now() - timedelta(days=2)
        start_date = end_date - timedelta(days=5)
        return start_date, end_date

def fetch_data(sym, period="1d", interval="1m"):
    """Fetch data from Yahoo Finance with specified period and interval.
    If market is closed, fetches data from the previous trading week."""
    try:
        # First, try to fetch current data
        data = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=True)
        
        # If data is empty or market is closed, try fetching last week's data
        market_closed = is_market_closed()
        if data.empty or (market_closed and (data.empty or len(data) < 10)):
            if market_closed:
                print(f"Market is closed for {sym}. Fetching last week's data...")
            else:
                print(f"Insufficient data for {sym}. Fetching last week's data...")
            start_date, end_date = get_last_week_dates()
            
            # For different intervals, adjust the fetch strategy
            if interval in ["1m", "5m", "15m"]:
                # For intraday, fetch a wider range to ensure we get enough data
                # Fetch from last Monday to last Friday, but extend a bit for safety
                extended_start = start_date - timedelta(days=2)
                extended_end = end_date + timedelta(days=1)
                data = yf.download(
                    sym, 
                    start=extended_start.strftime('%Y-%m-%d'),
                    end=extended_end.strftime('%Y-%m-%d'),
                    interval=interval,
                    progress=False,
                    auto_adjust=True
                )
                # Filter to last week's trading hours if we got data
                if not data.empty:
                    # Keep data from last Monday 9:30 AM to last Friday 4:00 PM PST
                    data = data[(data.index >= start_date) & (data.index <= end_date)]
            else:
                # For daily/hourly, use a longer period to ensure we have data
                # Fetch last month, then filter to last week
                data = yf.download(
                    sym,
                    period="1mo",  # Get last month to ensure we have data
                    interval=interval,
                    progress=False,
                    auto_adjust=True
                )
                # Filter to last week's range if we got data
                if not data.empty:
                    data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            # If still empty after fallback, try a more aggressive approach
            if data.empty:
                print(f"Still no data for {sym}. Trying extended period...")
                # Try fetching with a longer period
                extended_periods = {
                    "1m": "5d",
                    "5m": "5d",
                    "15m": "5d",
                    "1h": "1mo",
                    "1d": "3mo"
                }
                fallback_period = extended_periods.get(interval, "1mo")
                data = yf.download(sym, period=fallback_period, interval=interval, progress=False, auto_adjust=True)
                # If we got data, take the most recent portion
                if not data.empty and len(data) > 100:
                    data = data.tail(100)  # Take last 100 data points
        
        if data.empty:
            return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        
        # Handle MultiIndex columns (yfinance sometimes returns these)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        
        data = data.reset_index()
        
        # Find datetime column
        datetime_col = None
        for col in ["Datetime", "Date", "index"]:
            if col in data.columns:
                datetime_col = col
                break
        
        if datetime_col:
            data["Datetime"] = pd.to_datetime(data[datetime_col], utc=True)
        else:
            # If no datetime column found, create one from index
            data["Datetime"] = pd.to_datetime(data.index, utc=True)
        
        # Ensure we have all required columns (case-insensitive)
        required_cols_lower = {col.lower(): col for col in ["Open", "High", "Low", "Close", "Volume"]}
        data_cols_lower = {col.lower(): col for col in data.columns}
        
        result_data = {"Datetime": data["Datetime"]}
        
        for req_lower, req_orig in required_cols_lower.items():
            if req_lower in data_cols_lower:
                result_data[req_orig] = data[data_cols_lower[req_lower]]
            else:
                # Default values if column missing
                if req_orig == "Volume":
                    result_data[req_orig] = 0
                elif req_orig == "Close":
                    # Try to get Close from other columns
                    if "close" in data_cols_lower:
                        result_data[req_orig] = data[data_cols_lower["close"]]
                    else:
                        result_data[req_orig] = 0
                else:
                    # For Open, High, Low, use Close if available
                    if "close" in data_cols_lower:
                        result_data[req_orig] = data[data_cols_lower["close"]]
                    else:
                        result_data[req_orig] = 0
        
        return pd.DataFrame(result_data)
    except Exception as e:
        print(f"Error fetching {sym}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])

def calculate_indicators(df):
    """Calculate technical indicators."""
    if df.empty or len(df) < 14:
        return df
    
    # Moving Averages
    df["MA_20"] = df["Close"].rolling(window=min(20, len(df))).mean()
    df["MA_50"] = df["Close"].rolling(window=min(50, len(df))).mean()
    df["MA_200"] = df["Close"].rolling(window=min(200, len(df))).mean()
    
    # RSI Calculation
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=min(14, len(df))).mean()
    avg_loss = loss.rolling(window=min(14, len(df))).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=min(20, len(df))).mean()
    bb_std = df["Close"].rolling(window=min(20, len(df))).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df["Close"].ewm(span=min(12, len(df)), adjust=False).mean()
    ema_26 = df["Close"].ewm(span=min(26, len(df)), adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=min(9, len(df)), adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    
    # Stochastic Oscillator
    low_14 = df["Low"].rolling(window=min(14, len(df))).min()
    high_14 = df["High"].rolling(window=min(14, len(df))).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=min(3, len(df))).mean()
    
    # Average Directional Index (ADX) - simplified calculation
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    atr = true_range.rolling(window=min(14, len(df))).mean()
    plus_di = 100 * (plus_dm.rolling(window=min(14, len(df))).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=min(14, len(df))).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df["ADX"] = dx.rolling(window=min(14, len(df))).mean()
    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di
    
    # Volume indicators
    df["Volume_MA"] = df["Volume"].rolling(window=min(20, len(df))).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"] if df["Volume_MA"].iloc[-1] != 0 else 0
    
    # ATR (Average True Range) - already calculated above, store it
    df["ATR"] = atr
    
    # Advanced indicators using pandas-ta if available
    if PANDAS_TA_AVAILABLE and len(df) >= 20:
        try:
            # Ichimoku Cloud
            ichimoku = ta.ichimoku(df["High"], df["Low"], df["Close"])
            if ichimoku is not None and isinstance(ichimoku, tuple):
                df["Ichimoku_A"] = ichimoku[0].iloc[:, 0] if len(ichimoku[0].columns) > 0 else None
                df["Ichimoku_B"] = ichimoku[0].iloc[:, 1] if len(ichimoku[0].columns) > 1 else None
            
            # Parabolic SAR
            psar = ta.psar(df["High"], df["Low"], df["Close"])
            if psar is not None:
                df["PSAR"] = psar
            
            # Supertrend
            supertrend = ta.supertrend(df["High"], df["Low"], df["Close"])
            if supertrend is not None:
                df["SuperTrend"] = supertrend.iloc[:, 0] if len(supertrend.columns) > 0 else None
            
            # Aroon
            aroon = ta.aroon(df["High"], df["Low"])
            if aroon is not None:
                df["Aroon_Up"] = aroon.iloc[:, 0] if len(aroon.columns) > 0 else None
                df["Aroon_Down"] = aroon.iloc[:, 1] if len(aroon.columns) > 1 else None
        except Exception as e:
            print(f"Warning: pandas-ta indicators failed: {e}")
    
    # Momentum Indicators
    df["Momentum"] = df["Close"].diff(periods=min(10, len(df)))
    df["ROC"] = ((df["Close"] - df["Close"].shift(min(10, len(df)))) / df["Close"].shift(min(10, len(df)))) * 100
    
    # Williams %R
    wr_period = min(14, len(df))
    highest_high = df["High"].rolling(window=wr_period).max()
    lowest_low = df["Low"].rolling(window=wr_period).min()
    df["Williams_R"] = -100 * ((highest_high - df["Close"]) / (highest_high - lowest_low))
    
    # Commodity Channel Index (CCI)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cci_period = min(20, len(df))
    sma_tp = typical_price.rolling(window=cci_period).mean()
    mad = typical_price.rolling(window=cci_period).apply(lambda x: abs(x - x.mean()).mean())
    df["CCI"] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Money Flow Index (MFI)
    typical_price_mfi = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price_mfi * df["Volume"]
    positive_flow = money_flow.where(typical_price_mfi > typical_price_mfi.shift(1), 0).rolling(window=min(14, len(df))).sum()
    negative_flow = money_flow.where(typical_price_mfi < typical_price_mfi.shift(1), 0).rolling(window=min(14, len(df))).sum()
    mfi_ratio = positive_flow / negative_flow
    df["MFI"] = 100 - (100 / (1 + mfi_ratio))
    
    # Support and Resistance levels (simplified - using local minima/maxima)
    window = min(10, len(df) // 4)
    if len(df) >= window * 2:
        df["Support"] = df["Low"].rolling(window=window, center=True).min()
        df["Resistance"] = df["High"].rolling(window=window, center=True).max()
    else:
        df["Support"] = df["Low"]
        df["Resistance"] = df["High"]
    
    return df

def detect_patterns(df):
    """Detect candlestick patterns and chart patterns."""
    if df.empty or len(df) < 5:
        return []
    
    patterns = []
    latest_idx = len(df) - 1  # Use actual last index
    
    if latest_idx >= 0 and latest_idx < len(df):
        # Get recent candles
        recent = df.iloc[max(0, latest_idx-4):latest_idx+1] if len(df) >= 5 else df
        
        if len(recent) >= 3:
            # Doji pattern
            body = abs(recent["Close"].iloc[-1] - recent["Open"].iloc[-1])
            range_candle = recent["High"].iloc[-1] - recent["Low"].iloc[-1]
            if range_candle > 0 and body / range_candle < 0.1:
                patterns.append("DOJI - Reversal signal")
            
            # Engulfing patterns
            if len(recent) >= 2:
                prev_body = abs(recent["Close"].iloc[-2] - recent["Open"].iloc[-2])
                curr_body = abs(recent["Close"].iloc[-1] - recent["Open"].iloc[-1])
                
                # Bullish engulfing
                if (recent["Open"].iloc[-2] > recent["Close"].iloc[-2] and  # prev red
                    recent["Close"].iloc[-1] > recent["Open"].iloc[-1] and  # curr green
                    recent["Open"].iloc[-1] < recent["Close"].iloc[-2] and
                    recent["Close"].iloc[-1] > recent["Open"].iloc[-2]):
                    patterns.append("BULLISH ENGULFING - Buy signal")
                
                # Bearish engulfing
                if (recent["Open"].iloc[-2] < recent["Close"].iloc[-2] and  # prev green
                    recent["Close"].iloc[-1] < recent["Open"].iloc[-1] and  # curr red
                    recent["Open"].iloc[-1] > recent["Close"].iloc[-2] and
                    recent["Close"].iloc[-1] < recent["Open"].iloc[-2]):
                    patterns.append("BEARISH ENGULFING - Sell signal")
            
            # Hammer pattern
            if len(recent) >= 1:
                body_size = abs(recent["Close"].iloc[-1] - recent["Open"].iloc[-1])
                lower_shadow = min(recent["Open"].iloc[-1], recent["Close"].iloc[-1]) - recent["Low"].iloc[-1]
                upper_shadow = recent["High"].iloc[-1] - max(recent["Open"].iloc[-1], recent["Close"].iloc[-1])
                total_range = recent["High"].iloc[-1] - recent["Low"].iloc[-1]
                
                if total_range > 0 and lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
                    patterns.append("HAMMER - Potential reversal")
    
    return patterns

def calculate_fibonacci_levels(df):
    """Calculate Fibonacci retracement levels."""
    if df.empty or len(df) < 20:
        return None, None
    
    # Use recent high and low for fib levels
    recent_period = min(50, len(df))
    recent_data = df.tail(recent_period)
    
    high_price = float(recent_data["High"].max())
    low_price = float(recent_data["Low"].min())
    
    diff = high_price - low_price
    
    fib_levels = {
        "0.0": high_price,
        "0.236": high_price - (diff * 0.236),
        "0.382": high_price - (diff * 0.382),
        "0.500": high_price - (diff * 0.500),
        "0.618": high_price - (diff * 0.618),
        "0.786": high_price - (diff * 0.786),
        "1.0": low_price,
    }
    
    return high_price, fib_levels

def calculate_risk_metrics(df):
    """Calculate risk and performance metrics."""
    if df.empty or len(df) < 10:
        return {}
    
    returns = df["Close"].pct_change().dropna()
    
    if len(returns) == 0:
        return {}
    
    # Volatility
    volatility = returns.std() * (252 ** 0.5) * 100  # Annualized
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Sharpe Ratio (simplified - assuming 0 risk-free rate)
    sharpe_ratio = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
    
    # Average True Range
    atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns and not pd.isna(df["ATR"].iloc[-1]) else 0
    
    # Trend Strength
    trend_strength = float(df["ADX"].iloc[-1]) if "ADX" in df.columns and not pd.isna(df["ADX"].iloc[-1]) else 0
    
    # Market Regime (simplified)
    current_price = float(df["Close"].iloc[-1])
    ma_20 = float(df["MA_20"].iloc[-1]) if "MA_20" in df.columns and not pd.isna(df["MA_20"].iloc[-1]) else current_price
    ma_50 = float(df["MA_50"].iloc[-1]) if "MA_50" in df.columns and not pd.isna(df["MA_50"].iloc[-1]) else current_price
    
    if current_price > ma_20 > ma_50:
        regime = "BULLISH"
    elif current_price < ma_20 < ma_50:
        regime = "BEARISH"
    else:
        regime = "NEUTRAL"
    
    return {
        "volatility_annual": volatility,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "atr": atr,
        "trend_strength": trend_strength,
        "market_regime": regime,
    }

def fetch_market_news(symbol: str, limit: int = 5) -> List[Dict]:
    """Fetch market news using NewsAPI."""
    if not NEWS_API_KEY:
        return []
    
    try:
        # Map futures symbols to search terms
        symbol_map = {
            "ES=F": "S&P 500",
            "NQ=F": "NASDAQ",
            "GC=F": "gold",
            "YM=F": "Dow Jones",
            "CL=F": "crude oil"
        }
        query = symbol_map.get(symbol, symbol.replace("=F", ""))
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": limit
        }
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            return [
                {
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "publishedAt": article.get("publishedAt", ""),
                }
                for article in articles[:limit]
            ]
    except Exception as e:
        print(f"Error fetching news: {e}")
    
    return []

def analyze_sentiment(text: str) -> Dict:
    """Analyze sentiment of text using VADER and TextBlob."""
    if not SENTIMENT_AVAILABLE:
        return {"compound": 0, "polarity": 0, "subjectivity": 0}
    
    try:
        # VADER sentiment
        analyzer = SentimentIntensityAnalyzer()
        vader_scores = analyzer.polarity_scores(text)
        
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        return {
            "compound": vader_scores["compound"],
            "polarity": polarity,
            "subjectivity": subjectivity,
            "positive": vader_scores["pos"],
            "negative": vader_scores["neg"],
            "neutral": vader_scores["neu"],
        }
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {"compound": 0, "polarity": 0, "subjectivity": 0}

def predict_price_ml(df: pd.DataFrame, periods: int = 5) -> Optional[Dict]:
    """Predict future prices using machine learning."""
    if not SKLEARN_AVAILABLE or df.empty or len(df) < 50:
        return None
    
    try:
        # Prepare features
        features = []
        target = []
        
        # Create features from past data
        lookback = 10
        for i in range(lookback, len(df)):
            row_features = []
            # Price features
            row_features.extend([
                df["Close"].iloc[i-j] for j in range(1, lookback+1)
            ])
            # Technical indicators
            if "RSI" in df.columns:
                row_features.append(df["RSI"].iloc[i-1] if i > 0 else 50)
            if "MACD" in df.columns:
                row_features.append(df["MACD"].iloc[i-1] if i > 0 else 0)
            if "Volume" in df.columns:
                row_features.append(df["Volume"].iloc[i-1] if i > 0 else 0)
            
            features.append(row_features)
            target.append(df["Close"].iloc[i])
        
        if len(features) < 20:
            return None
        
        X = np.array(features)
        y = np.array(target)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train_scaled, y_train)
        
        # Predict next periods
        last_features = features[-1]
        predictions = []
        current_features = last_features.copy()
        
        for _ in range(periods):
            # Scale and predict
            X_pred = scaler.transform([current_features])
            pred = model.predict(X_pred)[0]
            predictions.append(float(pred))
            
            # Update features for next prediction (shift and add new prediction)
            current_features = current_features[1:] + [pred]
        
        # Calculate confidence (R² score)
        score = model.score(X_test_scaled, y_test)
        confidence = max(0, min(100, score * 100))
        
        return {
            "predictions": predictions,
            "confidence": confidence,
            "next_price": predictions[0] if predictions else None,
        }
    except Exception as e:
        print(f"Error in ML prediction: {e}")
        return None

def fetch_economic_indicators() -> Dict:
    """Fetch key economic indicators from FRED."""
    if not FRED_AVAILABLE or not fred:
        return {}
    
    try:
        indicators = {}
        
        # VIX (Volatility Index)
        try:
            vix = fred.get_series("VIXCLS", observation_start=datetime.now() - timedelta(days=30))
            if not vix.empty:
                indicators["VIX"] = float(vix.iloc[-1])
        except:
            pass
        
        # 10-Year Treasury Rate
        try:
            dgs10 = fred.get_series("DGS10", observation_start=datetime.now() - timedelta(days=30))
            if not dgs10.empty:
                indicators["10Y_Treasury"] = float(dgs10.iloc[-1])
        except:
            pass
        
        # Unemployment Rate
        try:
            unrate = fred.get_series("UNRATE", observation_start=datetime.now() - timedelta(days=90))
            if not unrate.empty:
                indicators["Unemployment"] = float(unrate.iloc[-1])
        except:
            pass
        
        # GDP Growth
        try:
            gdp = fred.get_series("GDPC1", observation_start=datetime.now() - timedelta(days=365))
            if not gdp.empty and len(gdp) >= 2:
                gdp_growth = ((gdp.iloc[-1] - gdp.iloc[-2]) / gdp.iloc[-2]) * 100
                indicators["GDP_Growth"] = float(gdp_growth)
        except:
            pass
        
        return indicators
    except Exception as e:
        print(f"Error fetching economic indicators: {e}")
        return {}

def calculate_correlation(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """Calculate correlation between two price series."""
    if df1.empty or df2.empty:
        return 0.0
    
    try:
        # Align the dataframes by datetime
        merged = pd.merge(
            df1[["Datetime", "Close"]].rename(columns={"Close": "Close1"}),
            df2[["Datetime", "Close"]].rename(columns={"Close": "Close2"}),
            on="Datetime",
            how="inner"
        )
        
        if len(merged) < 10:
            return 0.0
        
        correlation = merged["Close1"].corr(merged["Close2"])
        return float(correlation) if not pd.isna(correlation) else 0.0
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return 0.0

def calculate_trading_signal(df):
    """Calculate Buy/Hold/Sell signal based on multiple indicators."""
    if df.empty or len(df) < 20:
        return "HOLD", 0, "Insufficient data", []
    
    signals = []
    score = 0
    signal_strength = []
    
    # Get latest values
    latest_idx = -1
    current_price = float(df["Close"].iloc[latest_idx])
    
    # RSI signals
    if "RSI" in df.columns and not pd.isna(df["RSI"].iloc[latest_idx]):
        rsi = float(df["RSI"].iloc[latest_idx])
        if rsi < 30:
            score += 2
            signals.append("RSI oversold")
            signal_strength.append(("RSI", "STRONG BUY", "#7dd87d"))
        elif rsi > 70:
            score -= 2
            signals.append("RSI overbought")
            signal_strength.append(("RSI", "STRONG SELL", "#ff6b6b"))
        elif rsi < 40:
            score += 1
            signal_strength.append(("RSI", "WEAK BUY", "#7dd87d"))
        elif rsi > 60:
            score -= 1
            signal_strength.append(("RSI", "WEAK SELL", "#ff6b6b"))
        else:
            signal_strength.append(("RSI", "NEUTRAL", "#00bfff"))
    
    # Moving Average signals
    if "MA_20" in df.columns and "MA_50" in df.columns:
        ma20 = float(df["MA_20"].iloc[latest_idx]) if not pd.isna(df["MA_20"].iloc[latest_idx]) else None
        ma50 = float(df["MA_50"].iloc[latest_idx]) if not pd.isna(df["MA_50"].iloc[latest_idx]) else None
        
        if ma20 and ma50:
            if ma20 > ma50:
                score += 1.5
                signals.append("MA20 > MA50 (bullish)")
            else:
                score -= 1.5
                signals.append("MA20 < MA50 (bearish)")
            
            if current_price > ma20:
                score += 1
                signals.append("Price above MA20")
            else:
                score -= 1
    
    # MACD signals
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        macd = float(df["MACD"].iloc[latest_idx]) if not pd.isna(df["MACD"].iloc[latest_idx]) else None
        macd_signal = float(df["MACD_Signal"].iloc[latest_idx]) if not pd.isna(df["MACD_Signal"].iloc[latest_idx]) else None
        
        if macd and macd_signal:
            if macd > macd_signal:
                score += 1.5
                signals.append("MACD bullish crossover")
            else:
                score -= 1.5
                signals.append("MACD bearish crossover")
    
    # Stochastic signals
    if "Stoch_K" in df.columns and "Stoch_D" in df.columns:
        stoch_k = float(df["Stoch_K"].iloc[latest_idx]) if not pd.isna(df["Stoch_K"].iloc[latest_idx]) else None
        stoch_d = float(df["Stoch_D"].iloc[latest_idx]) if not pd.isna(df["Stoch_D"].iloc[latest_idx]) else None
        
        if stoch_k and stoch_d:
            if stoch_k < 20:
                score += 1
                signals.append("Stoch oversold")
            elif stoch_k > 80:
                score -= 1
                signals.append("Stoch overbought")
            
            if stoch_k > stoch_d:
                score += 0.5
            else:
                score -= 0.5
    
    # ADX signals (trend strength)
    if "ADX" in df.columns and not pd.isna(df["ADX"].iloc[latest_idx]):
        adx = float(df["ADX"].iloc[latest_idx])
        if adx > 25:
            signals.append(f"Strong trend (ADX: {adx:.1f})")
        elif adx < 20:
            signals.append(f"Weak trend (ADX: {adx:.1f})")
    
    # Bollinger Bands
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        bb_upper = float(df["BB_Upper"].iloc[latest_idx]) if not pd.isna(df["BB_Upper"].iloc[latest_idx]) else None
        bb_lower = float(df["BB_Lower"].iloc[latest_idx]) if not pd.isna(df["BB_Lower"].iloc[latest_idx]) else None
        
        if bb_upper and bb_lower:
            if current_price < bb_lower:
                score += 1
                signals.append("Price near lower BB")
            elif current_price > bb_upper:
                score -= 1
                signals.append("Price near upper BB")
    
    # Volume confirmation
    if "Volume_Ratio" in df.columns and not pd.isna(df["Volume_Ratio"].iloc[latest_idx]):
        vol_ratio = float(df["Volume_Ratio"].iloc[latest_idx])
        if vol_ratio > 1.5:
            signals.append(f"High volume ({vol_ratio:.2f}x avg)")
            if score > 0:
                score += 0.5
            elif score < 0:
                score -= 0.5
    
    # Additional indicators
    # Williams %R
    if "Williams_R" in df.columns and not pd.isna(df["Williams_R"].iloc[latest_idx]):
        wr = float(df["Williams_R"].iloc[latest_idx])
        if wr < -80:
            score += 1
            signals.append("Williams %R oversold")
        elif wr > -20:
            score -= 1
            signals.append("Williams %R overbought")
    
    # MFI
    if "MFI" in df.columns and not pd.isna(df["MFI"].iloc[latest_idx]):
        mfi = float(df["MFI"].iloc[latest_idx])
        if mfi < 20:
            score += 1
            signals.append("MFI oversold")
        elif mfi > 80:
            score -= 1
            signals.append("MFI overbought")
    
    # Pattern detection
    patterns = detect_patterns(df)
    for pattern in patterns:
        if "BULLISH" in pattern or "HAMMER" in pattern:
            score += 1.5
        elif "BEARISH" in pattern:
            score -= 1.5
        signals.append(pattern)
    
    # Determine signal
    if score >= 4:
        signal = "BUY"
        confidence = min(100, 50 + (score * 8))
    elif score <= -4:
        signal = "SELL"
        confidence = min(100, 50 + (abs(score) * 8))
    elif score >= 2:
        signal = "BUY"
        confidence = min(85, 50 + (score * 7))
    elif score <= -2:
        signal = "SELL"
        confidence = min(85, 50 + (abs(score) * 7))
    else:
        signal = "HOLD"
        confidence = 50 + (abs(score) * 5)
    
    reason = " | ".join(signals[:5]) if signals else "Mixed signals"
    
    return signal, confidence, reason, signal_strength

def calculate_metrics(df, symbol):
    """Calculate key metrics for display."""
    if df.empty:
        return {
            "current_price": 0,
            "change": 0,
            "change_pct": 0,
            "high": 0,
            "low": 0,
            "volume": 0,
            "rsi": 0,
            "volatility": 0,
            "macd": 0,
            "macd_signal": 0,
            "stoch_k": 0,
            "stoch_d": 0,
            "adx": 0,
            "volume_ratio": 0,
            "ma_20": 0,
            "ma_50": 0,
        }
    
    current_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[0]) if len(df) > 1 else current_price
    change = current_price - prev_price
    change_pct = (change / prev_price * 100) if prev_price != 0 else 0
    
    high = float(df["High"].max())
    low = float(df["Low"].min())
    volume = int(df["Volume"].sum())
    
    rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns and not pd.isna(df["RSI"].iloc[-1]) else 0
    macd = float(df["MACD"].iloc[-1]) if "MACD" in df.columns and not pd.isna(df["MACD"].iloc[-1]) else 0
    macd_signal = float(df["MACD_Signal"].iloc[-1]) if "MACD_Signal" in df.columns and not pd.isna(df["MACD_Signal"].iloc[-1]) else 0
    stoch_k = float(df["Stoch_K"].iloc[-1]) if "Stoch_K" in df.columns and not pd.isna(df["Stoch_K"].iloc[-1]) else 0
    stoch_d = float(df["Stoch_D"].iloc[-1]) if "Stoch_D" in df.columns and not pd.isna(df["Stoch_D"].iloc[-1]) else 0
    adx = float(df["ADX"].iloc[-1]) if "ADX" in df.columns and not pd.isna(df["ADX"].iloc[-1]) else 0
    volume_ratio = float(df["Volume_Ratio"].iloc[-1]) if "Volume_Ratio" in df.columns and not pd.isna(df["Volume_Ratio"].iloc[-1]) else 0
    ma_20 = float(df["MA_20"].iloc[-1]) if "MA_20" in df.columns and not pd.isna(df["MA_20"].iloc[-1]) else 0
    ma_50 = float(df["MA_50"].iloc[-1]) if "MA_50" in df.columns and not pd.isna(df["MA_50"].iloc[-1]) else 0
    
    # Volatility (standard deviation of returns)
    returns = df["Close"].pct_change().dropna()
    volatility = float(returns.std() * 100) if len(returns) > 0 else 0
    
    return {
        "current_price": current_price,
        "change": change,
        "change_pct": change_pct,
        "high": high,
        "low": low,
        "volume": volume,
        "rsi": rsi,
        "volatility": volatility,
        "macd": macd,
        "macd_signal": macd_signal,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "adx": adx,
        "volume_ratio": volume_ratio,
        "ma_20": ma_20,
        "ma_50": ma_50,
    }

# ---------- DASH APP ----------
def create_dash_app():
    dash_app = dash.Dash(__name__, requests_pathname_prefix="/", suppress_callback_exceptions=True)

    # Optional HTTP Basic Auth protection controlled via env vars
    basic_auth_username = os.environ.get("DASH_ADMIN_USERNAME")
    basic_auth_password = os.environ.get("DASH_ADMIN_PASSWORD")
    if basic_auth_username and basic_auth_password:
        if DASH_AUTH_AVAILABLE:
            dash_auth.BasicAuth(dash_app, {basic_auth_username: basic_auth_password})
            print("Basic authentication enabled for dashboard access.")
        else:
            print(
                "Warning: DASH_ADMIN credentials provided but dash-auth is missing. "
                "Install dash-auth or remove the credentials to disable this warning."
            )
    else:
        print("Dashboard running without HTTP Basic Auth. Set DASH_ADMIN_USERNAME/PASSWORD to enable it.")

    # Harden HTTP responses with a light security header middleware
    server = dash_app.server
    if not hasattr(server, "_security_headers_added"):
        @server.after_request
        def add_security_headers(response):
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
            response.headers.setdefault("Referrer-Policy", "no-referrer-when-downgrade")
            response.headers.setdefault(
                "Permissions-Policy",
                "geolocation=(), microphone=(), camera=(), fullscreen=(self)"
            )
            csp = (
                "default-src 'self' data: blob: https://fonts.googleapis.com https://fonts.gstatic.com https://cdn.plot.ly; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com data:; "
                "img-src 'self' data: blob: https://cdn.plot.ly; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.plot.ly;"
            )
            response.headers.setdefault("Content-Security-Policy", csp)
            if os.environ.get("ENABLE_HSTS", "1") == "1":
                response.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
            return response
        server._security_headers_added = True
    
    # Custom CSS styling - Dark theme with Fallout/TopStepX vibes
    dash_app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Futures Trading Terminal</title>
            {%favicon%}
            {%css%}
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&display=swap');
                
                * {
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Rajdhani', 'Courier New', monospace;
                    background: #0a0a0a;
                    background-image: 
                        radial-gradient(circle at 20% 50%, rgba(125, 216, 125, 0.05) 0%, transparent 50%),
                        radial-gradient(circle at 80% 80%, rgba(255, 20, 147, 0.04) 0%, transparent 50%),
                        radial-gradient(circle at 50% 20%, rgba(0, 191, 255, 0.04) 0%, transparent 50%);
                    margin: 0;
                    padding: 0;
                    color: #7dd87d;
                    min-height: 100vh;
                }
                
                /* Navigation Link Styles */
                a[id^="nav-"] {
                    transition: all 0.3s ease;
                }
                
                a[id^="nav-"]:hover {
                    backgroundColor: rgba(125, 216, 125, 0.1);
                    borderLeft: 3px solid rgba(125, 216, 125, 0.6) !important;
                    paddingLeft: 17px;
                }
                
                /* Active navigation state */
                a[id^="nav-"].active {
                    backgroundColor: rgba(125, 216, 125, 0.15);
                    borderLeft: 3px solid #7dd87d !important;
                }
                
                .metric-card {
                    background: linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%);
                    border: 1px solid rgba(125, 216, 125, 0.3);
                    border-top: 2px solid rgba(125, 216, 125, 0.5);
                    border-radius: 6px;
                    padding: 20px;
                    margin: 10px;
                    box-shadow: 
                        0 0 20px rgba(125, 216, 125, 0.12),
                        0 0 40px rgba(255, 20, 147, 0.05),
                        inset 0 0 10px rgba(125, 216, 125, 0.03);
                    text-align: center;
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                    backdrop-filter: blur(3px);
                }
                
                .metric-card:hover {
                    border-color: rgba(125, 216, 125, 0.5);
                    box-shadow: 0 4px 12px rgba(125, 216, 125, 0.2);
                    transform: translateY(-2px);
                }
                
                .metric-value {
                    font-size: 26px;
                    font-weight: 600;
                    margin: 8px 0;
                    font-family: 'Share Tech Mono', monospace;
                    letter-spacing: 0.5px;
                }
                
                .metric-label {
                    font-size: 11px;
                    color: rgba(125, 216, 125, 0.7);
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    font-weight: 500;
                    opacity: 0.8;
                    filter: blur(0.3px);
                }
                
                .positive { 
                    color: #7dd87d; 
                    text-shadow: 0 0 10px #7dd87d, 0 0 20px rgba(125, 216, 125, 0.4), 0 0 30px rgba(0, 191, 255, 0.2);
                }
                
                .negative { 
                    color: #ff6b6b; 
                    text-shadow: 0 0 10px #ff6b6b, 0 0 20px rgba(255, 107, 107, 0.4), 0 0 30px rgba(255, 20, 147, 0.2);
                }
                
                .neutral { 
                    color: #00bfff; 
                    text-shadow: 0 0 10px #00bfff, 0 0 20px rgba(0, 191, 255, 0.4), 0 0 30px rgba(138, 43, 226, 0.2);
                }
                
                h1, h2, h3 {
                    font-family: 'Share Tech Mono', monospace;
                    text-shadow: 
                        0 0 15px currentColor, 
                        0 0 25px rgba(125, 216, 125, 0.5),
                        0 0 40px rgba(0, 191, 255, 0.3),
                        0 0 60px rgba(255, 20, 147, 0.2);
                    filter: blur(0.4px);
                }
                
                /* Dropdown styling */
                .Select-control {
                    background-color: rgba(26, 26, 26, 0.95) !important;
                    border: 1px solid rgba(125, 216, 125, 0.4) !important;
                    border-top: 2px solid rgba(0, 191, 255, 0.5) !important;
                    border-radius: 6px !important;
                    color: #7dd87d !important;
                    backdrop-filter: blur(3px);
                    box-shadow: 0 0 15px rgba(125, 216, 125, 0.1), 0 0 30px rgba(0, 191, 255, 0.05) !important;
                }
                
                .Select-menu-outer {
                    background-color: rgba(26, 26, 26, 0.98) !important;
                    border: 1px solid rgba(125, 216, 125, 0.4) !important;
                    border-top: 2px solid rgba(255, 20, 147, 0.5) !important;
                    backdrop-filter: blur(3px);
                    box-shadow: 0 0 20px rgba(255, 20, 147, 0.2) !important;
                }
                
                .Select-option {
                    background-color: transparent !important;
                    color: #7dd87d !important;
                    transition: all 0.2s ease;
                }
                
                .Select-option.is-focused {
                    background: linear-gradient(90deg, rgba(125, 216, 125, 0.15), rgba(0, 191, 255, 0.15)) !important;
                    color: #00bfff !important;
                    text-shadow: 0 0 10px #00bfff;
                }
                
                .Select-value-label {
                    color: #7dd87d !important;
                    filter: blur(0.2px);
                }
                
                .Select-placeholder {
                    color: rgba(125, 216, 125, 0.6) !important;
                    opacity: 0.6;
                }
                
                label {
                    color: rgba(125, 216, 125, 0.8) !important;
                    text-shadow: 0 0 8px rgba(125, 216, 125, 0.5), 0 0 15px rgba(0, 191, 255, 0.3);
                    font-weight: 600;
                    letter-spacing: 1.5px;
                    filter: blur(0.3px);
                }
                
                /* Scrollbar styling */
                ::-webkit-scrollbar {
                    width: 10px;
                    height: 10px;
                }
                
                ::-webkit-scrollbar-track {
                    background: #0a0a0a;
                }
                
                ::-webkit-scrollbar-thumb {
                    background: linear-gradient(180deg, rgba(125, 216, 125, 0.6), rgba(0, 191, 255, 0.6));
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(125, 216, 125, 0.3);
                }
                
                ::-webkit-scrollbar-thumb:hover {
                    background: linear-gradient(180deg, rgba(125, 216, 125, 0.8), rgba(255, 20, 147, 0.8));
                    box-shadow: 0 0 15px rgba(125, 216, 125, 0.5), 0 0 25px rgba(255, 20, 147, 0.3);
                }
                
                /* Live indicator pulse animation */
                @keyframes pulse {
                    0%, 100% {
                        opacity: 1;
                        text-shadow: 0 0 10px #00ff41, 0 0 20px #00ff41, 0 0 30px #00ff41;
                    }
                    50% {
                        opacity: 0.4;
                        text-shadow: 0 0 5px #00ff41, 0 0 10px #00ff41;
                    }
                }
                
                #live-dot {
                    animation: pulse 1.5s ease-in-out infinite;
                    display: inline-block;
                }
                
                /* Smooth chart transitions */
                .js-plotly-plot {
                    transition: opacity 0.4s ease-in-out;
                }
                
                /* Fade in animation for new data */
                @keyframes fadeIn {
                    from {
                        opacity: 0;
                        transform: translateY(-5px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                
                .metric-card {
                    animation: fadeIn 0.6s ease-out;
                }
                
                /* Subtle glow pulse on data updates */
                @keyframes dataUpdate {
                    0% {
                        box-shadow: 0 0 20px rgba(125, 216, 125, 0.12), 0 0 40px rgba(255, 20, 147, 0.05);
                    }
                    50% {
                        box-shadow: 0 0 30px rgba(125, 216, 125, 0.2), 0 0 60px rgba(0, 191, 255, 0.1);
                    }
                    100% {
                        box-shadow: 0 0 20px rgba(125, 216, 125, 0.12), 0 0 40px rgba(255, 20, 147, 0.05);
                    }
                }
                
                .metric-card:has(.metric-value) {
                    animation: fadeIn 0.6s ease-out, dataUpdate 2s ease-in-out infinite;
                }
                
                /* Trading Signal Display */
                .trading-signal-container {
                    background: linear-gradient(135deg, rgba(26, 26, 26, 0.98) 0%, rgba(15, 15, 15, 0.98) 100%);
                    border: 2px solid;
                    border-radius: 8px;
                    padding: 25px;
                    margin: 20px auto;
                    max-width: 600px;
                    text-align: center;
                    box-shadow: 
                        0 0 40px rgba(125, 216, 125, 0.3),
                        0 0 80px rgba(0, 191, 255, 0.2),
                        inset 0 0 20px rgba(125, 216, 125, 0.1);
                    backdrop-filter: blur(5px);
                    position: relative;
                    overflow: hidden;
                }
                
                .signal-buy {
                    border-color: #7dd87d;
                    box-shadow: 
                        0 0 40px rgba(125, 216, 125, 0.4),
                        0 0 80px rgba(125, 216, 125, 0.3),
                        inset 0 0 30px rgba(125, 216, 125, 0.15);
                }
                
                .signal-sell {
                    border-color: #ff6b6b;
                    box-shadow: 
                        0 0 40px rgba(255, 107, 107, 0.4),
                        0 0 80px rgba(255, 107, 107, 0.3),
                        inset 0 0 30px rgba(255, 107, 107, 0.15);
                }
                
                .signal-hold {
                    border-color: #00bfff;
                    box-shadow: 
                        0 0 40px rgba(0, 191, 255, 0.4),
                        0 0 80px rgba(0, 191, 255, 0.3),
                        inset 0 0 30px rgba(0, 191, 255, 0.15);
                }
                
                .signal-label {
                    font-size: 14px;
                    color: rgba(125, 216, 125, 0.7);
                    text-transform: uppercase;
                    letter-spacing: 3px;
                    margin-bottom: 10px;
                    font-weight: 600;
                }
                
                .signal-value {
                    font-size: 48px;
                    font-weight: bold;
                    font-family: 'Share Tech Mono', monospace;
                    margin: 15px 0;
                    text-shadow: 
                        0 0 20px currentColor,
                        0 0 40px currentColor,
                        0 0 60px currentColor;
                    letter-spacing: 5px;
                }
                
                .signal-confidence {
                    font-size: 16px;
                    margin-top: 10px;
                    opacity: 0.8;
                }
                
                .signal-reason {
                    font-size: 12px;
                    color: rgba(125, 216, 125, 0.6);
                    margin-top: 15px;
                    line-height: 1.6;
                    max-width: 500px;
                    margin-left: auto;
                    margin-right: auto;
                }
                
                /* Metrics Grid Layout */
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                    padding: 0 10px;
                }
                
                .metrics-section {
                    margin: 25px 0;
                }
                
                .metrics-section-title {
                    font-size: 18px;
                    color: rgba(125, 216, 125, 0.8);
                    text-transform: uppercase;
                    letter-spacing: 3px;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid rgba(125, 216, 125, 0.3);
                    font-family: 'Share Tech Mono', monospace;
                    font-weight: 600;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Import page components
    try:
        from pages.dashboard_page import create_dashboard_page
        from pages.paper_trading_page import create_paper_trading_page
        from pages.analytics_page import create_analytics_page
        from pages.settings_page import create_settings_page
        PAGES_AVAILABLE = True
    except ImportError:
        PAGES_AVAILABLE = False
        print("Warning: Page components not available, using single-page layout")
    
    # Multi-page layout with sidebar navigation
    dash_app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            # Sidebar Navigation
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("📈", style={"fontSize": "28px", "marginRight": "10px"}),
                                    html.Div(
                                        [
                                            html.Div("FUTURES", style={"fontSize": "14px", "color": "#7dd87d", "fontWeight": "600", "lineHeight": "1.2"}),
                                            html.Div("TRADING", style={"fontSize": "14px", "color": "#7dd87d", "fontWeight": "600", "lineHeight": "1.2"}),
                                        ],
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "padding": "20px",
                                    "borderBottom": "1px solid rgba(125, 216, 125, 0.2)",
                                },
                            ),
                            dcc.Link(
                                [
                                    html.Span("📊", style={"marginRight": "10px", "fontSize": "18px"}),
                                    html.Span("Dashboard", style={"fontSize": "14px"}),
                                ],
                                href="/",
                                id="nav-dashboard",
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "padding": "15px 20px",
                                    "color": "#7dd87d",
                                    "textDecoration": "none",
                                    "borderLeft": "3px solid transparent",
                                    "transition": "all 0.3s",
                                },
                            ),
                            dcc.Link(
                                [
                                    html.Span("💰", style={"marginRight": "10px", "fontSize": "18px"}),
                                    html.Span("Paper Trading", style={"fontSize": "14px"}),
                                ],
                                href="/paper-trading",
                                id="nav-paper-trading",
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "padding": "15px 20px",
                                    "color": "#7dd87d",
                                    "textDecoration": "none",
                                    "borderLeft": "3px solid transparent",
                                    "transition": "all 0.3s",
                                },
                            ),
                            dcc.Link(
                                [
                                    html.Span("📈", style={"marginRight": "10px", "fontSize": "18px"}),
                                    html.Span("Analytics", style={"fontSize": "14px"}),
                                ],
                                href="/analytics",
                                id="nav-analytics",
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "padding": "15px 20px",
                                    "color": "#7dd87d",
                                    "textDecoration": "none",
                                    "borderLeft": "3px solid transparent",
                                    "transition": "all 0.3s",
                                },
                            ),
                            dcc.Link(
                                [
                                    html.Span("⚙️", style={"marginRight": "10px", "fontSize": "18px"}),
                                    html.Span("Settings", style={"fontSize": "14px"}),
                                ],
                                href="/settings",
                                id="nav-settings",
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "padding": "15px 20px",
                                    "color": "#7dd87d",
                                    "textDecoration": "none",
                                    "borderLeft": "3px solid transparent",
                                    "transition": "all 0.3s",
                                },
                            ),
                        ],
                        style={"marginTop": "10px"},
                    ),
                ],
                style={
                    "position": "fixed",
                    "left": 0,
                    "top": 0,
                    "width": "250px",
                    "height": "100vh",
                    "backgroundColor": "#0f0f0f",
                    "borderRight": "1px solid rgba(125, 216, 125, 0.2)",
                    "overflowY": "auto",
                    "zIndex": "1000",
                },
            ),
            # Main Content Area
            html.Div(
                id="page-content",
                style={
                    "marginLeft": "250px",
                    "padding": "20px",
                    "minHeight": "100vh",
                    "backgroundColor": "#0a0a0a",
                },
            ),
            # Global components (modals, stores, intervals)
            dcc.Store(id="indicator-modal-store", data={"open": False, "indicator": None}),
            html.Div(
                id="indicator-education-modal",
                style={"display": "none"}
            ),
            dcc.Store(id="current-data-store", data={}),
            dcc.Store(id="current-metrics-store", data={}),
            dcc.Interval(id="interval", interval=30 * 1000, n_intervals=0),
        ],
        style={"display": "flex"},
    )
    
    # Page routing callback
    @dash_app.callback(
        Output("page-content", "children"),
        [Input("url", "pathname")],
    )
    def display_page(pathname):
        if pathname == "/paper-trading":
            if PAGES_AVAILABLE:
                return create_paper_trading_page(symbols)
            else:
                return html.Div("Paper Trading page not available")
        elif pathname == "/analytics":
            if PAGES_AVAILABLE:
                return create_analytics_page(symbols)
            else:
                return html.Div("Analytics page not available")
        elif pathname == "/settings":
            if PAGES_AVAILABLE:
                return create_settings_page()
            else:
                return html.Div("Settings page not available")
        else:  # Default to dashboard
            if PAGES_AVAILABLE:
                return create_dashboard_page(symbols)
            else:
                return html.Div("Dashboard page not available. Please ensure pages/ directory exists.")
    
    # Callback for indicator education modal
    @dash_app.callback(
        [Output("indicator-education-modal", "children"),
         Output("indicator-education-modal", "style")],
        [Input("rsi-learn-link", "n_clicks"),
         Input("macd-learn-link", "n_clicks"),
         Input("stoch-learn-link", "n_clicks"),
         Input("close-modal", "n_clicks")],
        [State("indicator-education-modal", "style")],
        prevent_initial_call=True
    )
    def toggle_indicator_education(rsi_clicks, macd_clicks, stoch_clicks, close_clicks, current_style):
        """Show/hide indicator education modal."""
        from dash import callback_context
        
        ctx = callback_context
        if not ctx.triggered:
            return html.Div(), {"display": "none"}
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Close modal
        if button_id == "close-modal" and close_clicks:
            return html.Div(), {"display": "none"}
        
        # Open modal for specific indicator
        indicator_map = {
            "rsi-learn-link": "RSI",
            "macd-learn-link": "MACD",
            "stoch-learn-link": "Stochastic"
        }
        
        indicator_name = indicator_map.get(button_id)
        if not indicator_name or not INDICATOR_EDUCATION_AVAILABLE:
            return html.Div(), {"display": "none"}
        
        info = get_indicator_info(indicator_name)
        if not info:
            return html.Div(), {"display": "none"}
        
        modal_content = html.Div([
            html.Div([
                html.Div([
                    html.H2(
                        info['name'],
                        style={
                            "color": "#7dd87d",
                            "fontFamily": "'Share Tech Mono', monospace",
                            "marginBottom": "20px"
                        }
                    ),
                    html.Button(
                        "X",
                        id="close-modal",
                        n_clicks=0,
                        style={
                            "position": "absolute",
                            "top": "10px",
                            "right": "10px",
                            "backgroundColor": "transparent",
                            "border": "1px solid rgba(125, 216, 125, 0.5)",
                            "color": "#7dd87d",
                            "fontSize": "20px",
                            "cursor": "pointer",
                            "width": "30px",
                            "height": "30px",
                            "borderRadius": "4px"
                        }
                    ),
                    html.Div([
                        html.H3("Description", style={"color": "#7dd87d", "fontSize": "16px", "marginTop": "15px"}),
                        html.P(
                            info['description'],
                            style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "13px", "lineHeight": "1.6"}
                        ),
                        html.H3("Range", style={"color": "#7dd87d", "fontSize": "16px", "marginTop": "15px"}),
                        html.P(
                            f"Typical Range: {info['range']}",
                            style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "13px"}
                        ),
                        html.H3("How to Interpret", style={"color": "#7dd87d", "fontSize": "16px", "marginTop": "15px"}),
                        html.Ul([
                            html.Li([
                                html.Strong(key.replace('_', ' ').title() + ":", style={"color": "#7dd87d"}),
                                html.Span(f" {value}", style={"color": "rgba(125, 216, 125, 0.8)"})
                            ])
                            for key, value in info['interpretation'].items()
                        ], style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "12px", "lineHeight": "1.8"}),
                        html.H3("Best Use Cases", style={"color": "#7dd87d", "fontSize": "16px", "marginTop": "15px"}),
                        html.P(
                            info['best_use'],
                            style={"color": "rgba(125, 216, 125, 0.8)", "fontSize": "13px", "lineHeight": "1.6"}
                        ),
                    ])
                ], style={
                    "position": "relative",
                    "backgroundColor": "#0a0a0a",
                    "border": "2px solid rgba(125, 216, 125, 0.5)",
                    "borderRadius": "8px",
                    "padding": "30px",
                    "maxWidth": "700px",
                    "maxHeight": "80vh",
                    "overflowY": "auto"
                })
            ], style={
                "position": "fixed",
                "top": "50%",
                "left": "50%",
                "transform": "translate(-50%, -50%)",
                "zIndex": "1000",
                "width": "90%",
                "maxWidth": "800px"
            }),
            html.Div(
                id="modal-backdrop",
                n_clicks=0,
                style={
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "width": "100%",
                    "height": "100%",
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "zIndex": "999"
                }
            )
        ])
        
        return modal_content, {"display": "block"}

    @dash_app.callback(
        [
            Output("price-graph", "figure"),
            Output("rsi-graph", "figure"),
            Output("macd-graph", "figure"),
            Output("stoch-graph", "figure"),
            Output("volume-graph", "figure"),
            Output("market-status-banner", "children"),
            Output("trading-signal-container", "children"),
            Output("metrics-container", "children"),
            Output("ml-predictions-container", "children"),
            Output("news-container", "children"),
            Output("economic-indicators-container", "children"),
            Output("macro-research-container", "children"),
            Output("rsi-education", "children"),
            Output("macd-education", "children"),
            Output("stoch-education", "children"),
        ] + ([
            Output("current-data-store", "data"),
            Output("current-metrics-store", "data"),
        ] if ENHANCEMENTS_AVAILABLE else []),
        [
            Input("symbol-dropdown", "value"),
            Input("interval-dropdown", "value"),
            Input("chart-type-dropdown", "value"),
            Input("interval", "n_intervals"),
        ],
        prevent_initial_call=False,  # Allow initial call to load data immediately
    )
    def update_all(symbol, interval_key, chart_type, n_intervals):
        try:
            period, interval = PERIOD_INTERVALS.get(interval_key, ("1d", "1m"))
            print(f"[UPDATE] Fetching data for {symbol} - Period: {period}, Interval: {interval}, Update #{n_intervals}")
            
            df = fetch_data(symbol, period, interval)
            
            # Check if market is closed and show banner
            market_closed = is_market_closed()
            market_status_banner = html.Div()  # Default: no banner
            if market_closed and not df.empty:
                # Show banner indicating we're displaying last week's data
                start_date, end_date = get_last_week_dates()
                market_status_banner = html.Div(
                    [
                        html.Div(
                            f"⚠️ Market is closed. Displaying data from last trading week ({start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')})",
                            style={
                                "backgroundColor": "rgba(255, 165, 0, 0.2)",
                                "border": "1px solid rgba(255, 165, 0, 0.5)",
                                "borderRadius": "8px",
                                "padding": "12px 20px",
                                "textAlign": "center",
                                "color": "#ffa500",
                                "fontSize": "14px",
                                "fontWeight": "500",
                                "backdropFilter": "blur(10px)",
                                "boxShadow": "0 0 15px rgba(255, 165, 0, 0.3)",
                            }
                        )
                    ]
                )
            
            if df.empty:
                print(f"[WARNING] No data returned for {symbol}")
                empty_fig = {
                    "data": [], 
                    "layout": {
                        "title": f"No data for {symbol}",
                        "plot_bgcolor": "#0a0a0a",
                        "paper_bgcolor": "#0a0a0a",
                    }
                }
                error_msg = html.Div(
                    f"No data available for {symbol}. Market may be closed or symbol unavailable.",
                    style={"color": "#ff6b6b", "textAlign": "center", "padding": "20px"}
                )
                # Return values matching the callback outputs
                empty_base_returns = (
                    empty_fig,  # price-graph
                    empty_fig,  # rsi-graph
                    empty_fig,  # macd-graph
                    empty_fig,  # stoch-graph
                    empty_fig,  # volume-graph
                    html.Div(),  # market-status-banner
                    error_msg,  # trading-signal-container
                    error_msg,  # metrics-container
                    error_msg,  # ml-predictions-container
                    error_msg,  # news-container
                    error_msg,  # economic-indicators-container
                    html.Div(),  # macro-research-container
                    html.Div(),  # rsi-education
                    html.Div(),  # macd-education
                    html.Div(),  # stoch-education
                )
                
                if ENHANCEMENTS_AVAILABLE:
                    return empty_base_returns + ({}, {})  # current-data-store, current-metrics-store
                else:
                    return empty_base_returns
            
            print(f"[SUCCESS] Fetched {len(df)} rows for {symbol}")
            
            df = calculate_indicators(df)
            metrics = calculate_metrics(df, symbol)
            trading_signal, confidence, reason, signal_strength = calculate_trading_signal(df)
            
            # Fetch additional data
            news_articles = fetch_market_news(symbol, limit=3)
            ml_prediction = predict_price_ml(df, periods=5)
            economic_data = fetch_economic_indicators()
            
            # Analyze news sentiment
            news_sentiment = []
            overall_sentiment = {"compound": 0, "polarity": 0}
            if news_articles:
                for article in news_articles:
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    sentiment = analyze_sentiment(text)
                    news_sentiment.append({
                        "article": article,
                        "sentiment": sentiment
                    })
                    overall_sentiment["compound"] += sentiment.get("compound", 0)
                    overall_sentiment["polarity"] += sentiment.get("polarity", 0)
                
                if len(news_sentiment) > 0:
                    overall_sentiment["compound"] /= len(news_sentiment)
                    overall_sentiment["polarity"] /= len(news_sentiment)
        
            # Main Price Chart
            fig_price = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[1],
        )
        
            if chart_type == "candlestick":
                fig_price.add_trace(
                    go.Candlestick(
                        x=df["Datetime"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="PRICE",
                        increasing_line_color="#7dd87d",
                        decreasing_line_color="#ff6b6b",
                        increasing_fillcolor="rgba(125, 216, 125, 0.8)",
                        decreasing_fillcolor="rgba(255, 107, 107, 0.8)",
                    ),
                    row=1, col=1,
                )
            elif chart_type == "ohlc":
                fig_price.add_trace(
                    go.Ohlc(
                        x=df["Datetime"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="PRICE",
                        increasing_line_color="#7dd87d",
                        decreasing_line_color="#ff6b6b",
                    ),
                    row=1, col=1,
                )
            else:  # line
                fig_price.add_trace(
                    go.Scatter(
                        x=df["Datetime"], 
                        y=df["Close"], 
                        mode="lines+markers", 
                        name="CLOSE", 
                        line=dict(color="#7dd87d", width=2.5, shape="spline"),
                        marker=dict(size=3, color="#7dd87d", opacity=0.6),
                        fill="tozeroy",
                        fillcolor="rgba(125, 216, 125, 0.1)",
                    ),
                    row=1, col=1,
                )
            
            # Add Moving Averages
            if "MA_20" in df.columns:
                fig_price.add_trace(
                    go.Scatter(x=df["Datetime"], y=df["MA_20"], mode="lines", name="MA 20", line=dict(color="#00bfff", width=2.5, dash="dot"), opacity=0.85),
                    row=1, col=1,
                )
            if "MA_50" in df.columns:
                fig_price.add_trace(
                    go.Scatter(x=df["Datetime"], y=df["MA_50"], mode="lines", name="MA 50", line=dict(color="#ff1493", width=2.5, dash="dash"), opacity=0.85),
                    row=1, col=1,
                )
            
            # Add Bollinger Bands
            if "BB_Upper" in df.columns:
                fig_price.add_trace(
                    go.Scatter(x=df["Datetime"], y=df["BB_Upper"], mode="lines", name="BB UPPER", line=dict(color="rgba(125, 216, 125, 0.5)", width=1.5, dash="dash"), showlegend=False, opacity=0.4),
                    row=1, col=1,
                )
                fig_price.add_trace(
                    go.Scatter(x=df["Datetime"], y=df["BB_Lower"], mode="lines", name="BB LOWER", line=dict(color="rgba(125, 216, 125, 0.5)", width=1.5, dash="dash"), fill="tonexty", fillcolor="rgba(125, 216, 125, 0.03)", showlegend=False, opacity=0.4),
                    row=1, col=1,
                )
            
            fig_price.update_layout(
                title=dict(
                    text=f"{symbols[symbol]} ({symbol}) — {interval_key.upper()}",
                    font=dict(color="#7dd87d", size=18, family="'Share Tech Mono', monospace"),
                ),
                xaxis=dict(
                    title=dict(text="TIME", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                    rangeslider=dict(visible=False),
                ),
                yaxis=dict(
                    title=dict(text="PRICE", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                ),
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                hovermode="x unified",
                height=500,
                font=dict(color="#7dd87d", family="'Share Tech Mono', monospace"),
                # Add smooth transitions
                transition=dict(duration=500, easing="cubic-in-out"),
                # Auto-scroll to latest data
                xaxis_rangeslider_visible=False,
            )
            
            # Add smooth animation by updating x-axis range to show latest data
            if len(df) > 0:
                # Show last 100 data points for better live feel
                max_points = min(100, len(df))
                fig_price.update_xaxes(range=[df["Datetime"].iloc[-max_points], df["Datetime"].iloc[-1]])
            
            # RSI Chart
            fig_rsi = go.Figure()
            if "RSI" in df.columns:
                fig_rsi.add_trace(
                    go.Scatter(
                        x=df["Datetime"], 
                        y=df["RSI"], 
                        mode="lines+markers", 
                        name="RSI", 
                        line=dict(color="#00bfff", width=3, shape="spline"),
                        marker=dict(size=4, color="#00bfff", opacity=0.7),
                        fill="tozeroy", 
                        fillcolor="rgba(0, 191, 255, 0.2)",
                    ),
                )
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="rgba(255, 20, 147, 0.8)", line_width=2.5, annotation_text="OVERBOUGHT (70)", annotation_font=dict(color="#ff1493", size=11, family="'Share Tech Mono', monospace"))
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="rgba(125, 216, 125, 0.8)", line_width=2.5, annotation_text="OVERSOLD (30)", annotation_font=dict(color="#7dd87d", size=11, family="'Share Tech Mono', monospace"))
                fig_rsi.add_hline(y=50, line_dash="dot", line_color="rgba(138, 43, 226, 0.5)", opacity=0.5, line_width=1.5)
            
            fig_rsi.update_layout(
                title=dict(
                    text="RSI (RELATIVE STRENGTH INDEX)",
                    font=dict(color="#7dd87d", size=16, family="'Share Tech Mono', monospace"),
                ),
                xaxis=dict(
                    title=dict(text="TIME", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                ),
                yaxis=dict(
                    title=dict(text="RSI", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                    range=[0, 100],
                ),
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                height=300,
                font=dict(color="#7dd87d", family="'Share Tech Mono', monospace"),
                # Add smooth transitions
                transition=dict(duration=500, easing="cubic-in-out"),
            )
            
            # Auto-scroll RSI chart to latest data
            if len(df) > 0:
                max_points = min(100, len(df))
                fig_rsi.update_xaxes(range=[df["Datetime"].iloc[-max_points], df["Datetime"].iloc[-1]])
            
            # MACD Chart
            fig_macd = go.Figure()
            if "MACD" in df.columns and "MACD_Signal" in df.columns:
                fig_macd.add_trace(
                    go.Scatter(
                        x=df["Datetime"],
                        y=df["MACD"],
                        mode="lines+markers",
                        name="MACD",
                        line=dict(color="#00bfff", width=2.5, shape="spline"),
                        marker=dict(size=3, color="#00bfff", opacity=0.7),
                    ),
                )
                fig_macd.add_trace(
                    go.Scatter(
                        x=df["Datetime"],
                        y=df["MACD_Signal"],
                        mode="lines+markers",
                        name="Signal",
                        line=dict(color="#ff1493", width=2.5, shape="spline"),
                        marker=dict(size=3, color="#ff1493", opacity=0.7),
                    ),
                )
                # Histogram
                colors_hist = ["#7dd87d" if val >= 0 else "#ff6b6b" for val in df["MACD_Histogram"]]
                fig_macd.add_trace(
                    go.Bar(
                        x=df["Datetime"],
                        y=df["MACD_Histogram"],
                        name="Histogram",
                        marker_color=colors_hist,
                        opacity=0.6,
                    ),
                )
                fig_macd.add_hline(y=0, line_dash="dash", line_color="rgba(125, 216, 125, 0.5)", line_width=1.5)
            
            fig_macd.update_layout(
                title=dict(
                    text="MACD (MOVING AVERAGE CONVERGENCE DIVERGENCE)",
                    font=dict(color="#7dd87d", size=16, family="'Share Tech Mono', monospace"),
                ),
                xaxis=dict(
                    title=dict(text="TIME", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                ),
                yaxis=dict(
                    title=dict(text="MACD", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                ),
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                height=300,
                font=dict(color="#7dd87d", family="'Share Tech Mono', monospace"),
                transition=dict(duration=500, easing="cubic-in-out"),
            )
            
            if len(df) > 0:
                max_points = min(100, len(df))
                fig_macd.update_xaxes(range=[df["Datetime"].iloc[-max_points], df["Datetime"].iloc[-1]])
            
            # Stochastic Chart
            fig_stoch = go.Figure()
            if "Stoch_K" in df.columns and "Stoch_D" in df.columns:
                fig_stoch.add_trace(
                    go.Scatter(
                        x=df["Datetime"],
                        y=df["Stoch_K"],
                        mode="lines+markers",
                        name="%K",
                        line=dict(color="#00bfff", width=2.5, shape="spline"),
                        marker=dict(size=4, color="#00bfff", opacity=0.7),
                    ),
                )
                fig_stoch.add_trace(
                    go.Scatter(
                        x=df["Datetime"],
                        y=df["Stoch_D"],
                        mode="lines+markers",
                        name="%D",
                        line=dict(color="#ff1493", width=2.5, shape="spline"),
                        marker=dict(size=4, color="#ff1493", opacity=0.7),
                    ),
                )
                fig_stoch.add_hline(y=80, line_dash="dash", line_color="rgba(255, 20, 147, 0.8)", line_width=2, annotation_text="OVERBOUGHT (80)", annotation_font=dict(color="#ff1493", size=11, family="'Share Tech Mono', monospace"))
                fig_stoch.add_hline(y=20, line_dash="dash", line_color="rgba(125, 216, 125, 0.8)", line_width=2, annotation_text="OVERSOLD (20)", annotation_font=dict(color="#7dd87d", size=11, family="'Share Tech Mono', monospace"))
                fig_stoch.add_hline(y=50, line_dash="dot", line_color="rgba(138, 43, 226, 0.5)", opacity=0.5, line_width=1.5)
            
            fig_stoch.update_layout(
                title=dict(
                    text="STOCHASTIC OSCILLATOR",
                    font=dict(color="#7dd87d", size=16, family="'Share Tech Mono', monospace"),
                ),
                xaxis=dict(
                    title=dict(text="TIME", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                ),
                yaxis=dict(
                    title=dict(text="STOCHASTIC", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                    range=[0, 100],
                ),
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                height=300,
                font=dict(color="#7dd87d", family="'Share Tech Mono', monospace"),
                transition=dict(duration=500, easing="cubic-in-out"),
            )
            
            if len(df) > 0:
                max_points = min(100, len(df))
                fig_stoch.update_xaxes(range=[df["Datetime"].iloc[-max_points], df["Datetime"].iloc[-1]])
            
            # Volume Chart
            fig_volume = go.Figure()
            if len(df) > 0:
                # Color code volume bars: red for down, green for up
                colors = []
                for i in range(len(df)):
                    if i < len(df) and pd.notna(df["Close"].iloc[i]) and pd.notna(df["Open"].iloc[i]):
                        colors.append("#ff6b6b" if df["Close"].iloc[i] < df["Open"].iloc[i] else "#7dd87d")
                    else:
                        colors.append("rgba(125, 216, 125, 0.6)")
                fig_volume.add_trace(
                    go.Bar(x=df["Datetime"], y=df["Volume"], name="VOLUME", marker_color=colors if colors else "#7dd87d", marker_line_width=0, opacity=0.8),
                )
            else:
                fig_volume.add_trace(
                    go.Bar(x=[], y=[], name="VOLUME"),
                )
            fig_volume.update_layout(
                title=dict(
                    text="VOLUME",
                    font=dict(color="#7dd87d", size=16, family="'Share Tech Mono', monospace"),
                ),
                xaxis=dict(
                    title=dict(text="TIME", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                ),
                yaxis=dict(
                    title=dict(text="VOLUME", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                ),
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                height=250,
                font=dict(color="#7dd87d", family="'Share Tech Mono', monospace"),
                # Add smooth transitions
                transition=dict(duration=500, easing="cubic-in-out"),
            )
            
            # Auto-scroll volume chart to latest data
            if len(df) > 0:
                max_points = min(100, len(df))
                fig_volume.update_xaxes(range=[df["Datetime"].iloc[-max_points], df["Datetime"].iloc[-1]])
            
            # Trading Signal Display
            signal_color = "#7dd87d" if trading_signal == "BUY" else "#ff6b6b" if trading_signal == "SELL" else "#00bfff"
            signal_class = "signal-buy" if trading_signal == "BUY" else "signal-sell" if trading_signal == "SELL" else "signal-hold"
            
            trading_signal_display = html.Div(
                [
                    html.Div("TRADING SIGNAL", className="signal-label"),
                    html.Div(
                        trading_signal,
                        className="signal-value",
                        style={"color": signal_color},
                    ),
                    html.Div(
                        f"Confidence: {confidence:.0f}%",
                        className="signal-confidence",
                        style={"color": "rgba(125, 216, 125, 0.8)"},
                    ),
                    html.Div(
                        reason,
                        className="signal-reason",
                    ),
                ],
                className=f"trading-signal-container {signal_class}",
            )
            
            # Reorganized Metrics Cards - Cleaner Layout
            metrics_cards = html.Div(
                [
                    # Price & Change Section
                    html.Div(
                        [
                            html.Div("PRICE & CHANGE", className="metrics-section-title"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(f"{metrics['current_price']:.2f}", className="metric-value", style={"color": "#667eea"}),
                                            html.Div("Current Price", className="metric-label"),
                                        ],
                                        className="metric-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{metrics['change']:+.2f} ({metrics['change_pct']:+.2f}%)",
                                                className="metric-value",
                                                style={"color": "#00c853" if metrics["change"] >= 0 else "#ff1744"},
                                            ),
                                            html.Div("Change", className="metric-label"),
                                        ],
                                        className="metric-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(f"{metrics['high']:.2f}", className="metric-value", style={"color": "#00c853"}),
                                            html.Div("High", className="metric-label"),
                                        ],
                                        className="metric-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(f"{metrics['low']:.2f}", className="metric-value", style={"color": "#ff1744"}),
                                            html.Div("Low", className="metric-label"),
                                        ],
                                        className="metric-card",
                                    ),
                                ],
                                className="metrics-grid",
                            ),
                        ],
                        className="metrics-section",
                    ),
                    
                    # Technical Indicators Section
                    html.Div(
                        [
                            html.Div("TECHNICAL INDICATORS", className="metrics-section-title"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{metrics['rsi']:.1f}",
                                                className="metric-value",
                                                style={"color": "#ff1744" if metrics["rsi"] > 70 else "#00c853" if metrics["rsi"] < 30 else "#667eea"},
                                            ),
                                            html.Div("RSI", className="metric-label"),
                                            html.Div(
                                                "Range: 0-100 | " + ("Overbought" if metrics["rsi"] > 70 else "Oversold" if metrics["rsi"] < 30 else "Neutral"),
                                                style={"color": "rgba(125, 216, 125, 0.6)", "fontSize": "10px", "marginTop": "5px", "fontStyle": "italic"}
                                            ) if INDICATOR_EDUCATION_AVAILABLE else html.Div(),
                                            html.A(
                                                "📚 Learn More",
                                                href="#",
                                                id="rsi-learn-link",
                                                style={
                                                    "color": "#7dd87d",
                                                    "textDecoration": "underline",
                                                    "fontSize": "10px",
                                                    "marginTop": "3px",
                                                    "display": "block",
                                                    "cursor": "pointer"
                                                }
                                            ) if INDICATOR_EDUCATION_AVAILABLE else html.Div(),
                                        ],
                                        className="metric-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{metrics['macd']:.3f}",
                                                className="metric-value",
                                                style={"color": "#00bfff"},
                                            ),
                                            html.Div("MACD", className="metric-label"),
                                            html.Div(
                                                "Oscillates around zero | Shows momentum",
                                                style={"color": "rgba(125, 216, 125, 0.6)", "fontSize": "10px", "marginTop": "5px", "fontStyle": "italic"}
                                            ) if INDICATOR_EDUCATION_AVAILABLE else html.Div(),
                                            html.A(
                                                "📚 Learn More",
                                                href="#",
                                                id="macd-learn-link",
                                                style={
                                                    "color": "#7dd87d",
                                                    "textDecoration": "underline",
                                                    "fontSize": "10px",
                                                    "marginTop": "3px",
                                                    "display": "block",
                                                    "cursor": "pointer"
                                                }
                                            ) if INDICATOR_EDUCATION_AVAILABLE else html.Div(),
                                        ],
                                        className="metric-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{metrics['stoch_k']:.1f}",
                                                className="metric-value",
                                                style={"color": "#ff1493"},
                                            ),
                                            html.Div("Stoch %K", className="metric-label"),
                                            html.Div(
                                                "Range: 0-100 | " + ("Overbought" if metrics.get('stoch_k', 50) > 80 else "Oversold" if metrics.get('stoch_k', 50) < 20 else "Neutral"),
                                                style={"color": "rgba(125, 216, 125, 0.6)", "fontSize": "10px", "marginTop": "5px", "fontStyle": "italic"}
                                            ) if INDICATOR_EDUCATION_AVAILABLE else html.Div(),
                                            html.A(
                                                "📚 Learn More",
                                                href="#",
                                                id="stoch-learn-link",
                                                style={
                                                    "color": "#7dd87d",
                                                    "textDecoration": "underline",
                                                    "fontSize": "10px",
                                                    "marginTop": "3px",
                                                    "display": "block",
                                                    "cursor": "pointer"
                                                }
                                            ) if INDICATOR_EDUCATION_AVAILABLE else html.Div(),
                                        ],
                                        className="metric-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{metrics['adx']:.1f}",
                                                className="metric-value",
                                                style={"color": "#8a2be2" if metrics["adx"] > 25 else "#666"},
                                            ),
                                            html.Div("ADX", className="metric-label"),
                                        ],
                                        className="metric-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{metrics['ma_20']:.2f}",
                                                className="metric-value",
                                                style={"color": "#00bfff"},
                                            ),
                                            html.Div("MA 20", className="metric-label"),
                                        ],
                                        className="metric-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{metrics['ma_50']:.2f}",
                                                className="metric-value",
                                                style={"color": "#ff1493"},
                                            ),
                                            html.Div("MA 50", className="metric-label"),
                                        ],
                                        className="metric-card",
                                    ),
                                ],
                                className="metrics-grid",
                            ),
                        ],
                        className="metrics-section",
                    ),
                    
                    # Market Data Section
                    html.Div(
                        [
                            html.Div("MARKET DATA", className="metrics-section-title"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(f"{metrics['volatility']:.2f}%", className="metric-value", style={"color": "#666"}),
                                            html.Div("Volatility", className="metric-label"),
                                        ],
                                        className="metric-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{metrics['volume_ratio']:.2f}x",
                                                className="metric-value",
                                                style={"color": "#00c853" if metrics["volume_ratio"] > 1.5 else "#666"},
                                            ),
                                            html.Div("Volume Ratio", className="metric-label"),
                                        ],
                                        className="metric-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(f"{metrics['volume']:,}", className="metric-value", style={"color": "#666"}),
                                            html.Div("Total Volume", className="metric-label"),
                                        ],
                                        className="metric-card",
                                    ),
                                ],
                                className="metrics-grid",
                            ),
                        ],
                        className="metrics-section",
                    ),
                ],
            )
            
            # ML Predictions Component
            ml_component = html.Div()
            if ml_prediction:
                predictions = ml_prediction.get("predictions", [])
                confidence_ml = ml_prediction.get("confidence", 0)
                next_price = ml_prediction.get("next_price", 0)
                current_price = metrics["current_price"]
                price_change = next_price - current_price if next_price else 0
                price_change_pct = (price_change / current_price * 100) if current_price > 0 else 0
                
                ml_component = html.Div(
                    [
                        html.Div("ML PRICE PREDICTION", className="metrics-section-title"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(f"{next_price:.2f}", className="metric-value", style={"color": "#00bfff"}),
                                        html.Div("Predicted Next Price", className="metric-label"),
                                    ],
                                    className="metric-card",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
                                            className="metric-value",
                                            style={"color": "#00c853" if price_change >= 0 else "#ff1744"},
                                        ),
                                        html.Div("Predicted Change", className="metric-label"),
                                    ],
                                    className="metric-card",
                                ),
                                html.Div(
                                    [
                                        html.Div(f"{confidence_ml:.1f}%", className="metric-value", style={"color": "#667eea"}),
                                        html.Div("Model Confidence", className="metric-label"),
                                    ],
                                    className="metric-card",
                                ),
                            ],
                            className="metrics-grid",
                            style={"maxWidth": "600px", "margin": "0 auto"},
                        ),
                    ],
                    className="metrics-section",
                    style={
                        "background": "linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%)",
                        "border": "1px solid rgba(0, 191, 255, 0.4)",
                        "borderRadius": "6px",
                        "padding": "20px",
                        "margin": "20px 0",
                    },
                )
            
            # News Component
            news_component = html.Div()
            if news_articles:
                news_items = []
                for item in news_sentiment:
                    article = item["article"]
                    sentiment = item["sentiment"]
                    compound = sentiment.get("compound", 0)
                    sentiment_color = "#7dd87d" if compound > 0.1 else "#ff6b6b" if compound < -0.1 else "#00bfff"
                    sentiment_label = "POSITIVE" if compound > 0.1 else "NEGATIVE" if compound < -0.1 else "NEUTRAL"
                    
                    news_items.append(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(article.get("title", "No title"), style={"fontSize": "14px", "fontWeight": "600", "color": "#7dd87d", "marginBottom": "5px"}),
                                        html.Div(article.get("description", "")[:150] + "...", style={"fontSize": "12px", "color": "rgba(125, 216, 125, 0.7)", "marginBottom": "10px"}),
                                        html.Div(
                                            [
                                                html.Span(f"Sentiment: {sentiment_label}", style={"color": sentiment_color, "fontSize": "11px", "marginRight": "15px"}),
                                                html.Span(f"Source: {article.get('source', {}).get('name', 'Pro Feed')}", style={"color": "#00bfff", "fontSize": "11px"}),
                                            ],
                                            style={"display": "flex", "justifyContent": "space-between"},
                                        ),
                                    ],
                                    style={"padding": "15px", "borderBottom": "1px solid rgba(125, 216, 125, 0.2)"},
                                ),
                            ],
                            style={"marginBottom": "10px"},
                        )
                    )
                
                overall_sentiment_color = "#7dd87d" if overall_sentiment["compound"] > 0.1 else "#ff6b6b" if overall_sentiment["compound"] < -0.1 else "#00bfff"
                
                news_component = html.Div(
                    [
                        html.Div(
                            [
                                html.Div("MARKET NEWS & SENTIMENT", className="metrics-section-title"),
                                html.Div(
                                    [
                                        html.Div(
                                            f"Overall Sentiment: {overall_sentiment['compound']:.2f}",
                                            style={"color": overall_sentiment_color, "fontSize": "14px", "fontWeight": "600", "marginBottom": "15px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(news_items),
                    ],
                    style={
                        "background": "linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%)",
                        "border": "1px solid rgba(125, 216, 125, 0.4)",
                        "borderRadius": "6px",
                        "padding": "20px",
                        "margin": "20px 0",
                        "maxHeight": "400px",
                        "overflowY": "auto",
                    },
                )
            
            # Economic Indicators Component
            econ_component = html.Div()
            if economic_data:
                econ_items = []
                for key, value in economic_data.items():
                    label = key.replace("_", " ")
                    color = "#00c853" if "VIX" not in key and value > 0 else "#ff1744" if "VIX" in key and value > 20 else "#667eea"
                    econ_items.append(
                        html.Div(
                            [
                                html.Div(f"{value:.2f}", className="metric-value", style={"color": color}),
                                html.Div(label, className="metric-label"),
                            ],
                            className="metric-card",
                        )
                    )
                
                econ_component = html.Div(
                    [
                        html.Div("ECONOMIC INDICATORS", className="metrics-section-title"),
                        html.Div(econ_items, className="metrics-grid"),
                    ],
                    className="metrics-section",
                    style={
                        "background": "linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%)",
                        "border": "1px solid rgba(138, 43, 226, 0.4)",
                        "borderRadius": "6px",
                        "padding": "20px",
                        "margin": "20px 0",
                    },
                )

            # Macro Research Component
            macro_component = html.Div()
            if MACRO_ENGINE_AVAILABLE and macro_engine:
                macro_result = macro_engine.run(
                    symbol,
                    df,
                    economic_data,
                    overall_sentiment if news_sentiment else None,
                )
                if macro_result:
                    history_df = macro_result.get("history")
                    macro_fig = go.Figure()
                    if history_df is not None and not history_df.empty:
                        macro_fig.add_trace(
                            go.Scatter(
                                x=history_df["Datetime"],
                                y=history_df["score"],
                                mode="lines",
                                line=dict(color="#7dd87d", width=3, shape="spline"),
                                name="Composite Score",
                            )
                        )
                        macro_fig.add_hline(y=25, line_dash="dot", line_color="#00c853", opacity=0.4)
                        macro_fig.add_hline(y=-25, line_dash="dot", line_color="#ff6b6b", opacity=0.4)
                        macro_fig.update_layout(
                            margin=dict(l=0, r=0, t=30, b=0),
                            height=260,
                            plot_bgcolor="#0a0a0a",
                            paper_bgcolor="#0a0a0a",
                            font=dict(color="#7dd87d", family="'Share Tech Mono', monospace"),
                            title=dict(
                                text="Adaptive Macro Factor Fusion",
                                font=dict(size=16),
                            ),
                            yaxis=dict(range=[-105, 105], gridcolor="rgba(125, 216, 125, 0.08)"),
                            xaxis=dict(gridcolor="rgba(125, 216, 125, 0.08)"),
                        )
                    factor_cards = []
                    for factor in macro_result.get("factors", []):
                        factor_cards.append(
                            html.Div(
                                [
                                    html.Div(f"{factor['value']:+.1f}", className="metric-value", style={"color": "#00bfff"}),
                                    html.Div(factor["name"].upper(), className="metric-label"),
                                    html.Div(
                                        f"Weight: {factor['weight']*100:.0f}%",
                                        style={"fontSize": "10px", "color": "rgba(125, 216, 125, 0.6)", "marginTop": "5px"},
                                    ),
                                ],
                                className="metric-card",
                            )
                        )
                    weights_badges = []
                    for name, value in (macro_result.get("weights") or {}).items():
                        weights_badges.append(
                            html.Div(
                                [
                                    html.Div(f"{name.upper()}", style={"fontSize": "10px", "fontWeight": "600"}),
                                    html.Div(f"{value*100:.0f}%", style={"fontSize": "12px", "color": "#7dd87d"}),
                                ],
                                className="metric-card",
                                style={"minWidth": "110px"},
                            )
                        )
                    scenario_cards = []
                    for scenario in macro_result.get("scenarios", []):
                        scenario_cards.append(
                            html.Div(
                                [
                                    html.Div(scenario.name, style={"fontWeight": "600", "marginBottom": "5px"}),
                                    html.Div(scenario.narrative, style={"fontSize": "12px", "color": "rgba(125, 216, 125, 0.7)", "marginBottom": "8px"}),
                                    html.Div(f"Projected Score: {scenario.projected_score:+.1f}", style={"color": "#7dd87d", "fontSize": "12px"}),
                                ],
                                className="metric-card",
                                style={"minHeight": "140px"},
                            )
                        )
                    backtest = macro_result.get("backtest")
                    backtest_fig = go.Figure()
                    backtest_stats_cards = []
                    if backtest and backtest.get("curve") is not None and not backtest["curve"].empty:
                        curve_df = backtest["curve"]
                        backtest_fig.add_trace(
                            go.Scatter(
                                x=curve_df["Datetime"],
                                y=curve_df["Strategy"],
                                mode="lines",
                                name="Macro Strategy",
                                line=dict(color="#7dd87d", width=2.5),
                            )
                        )
                        backtest_fig.add_trace(
                            go.Scatter(
                                x=curve_df["Datetime"],
                                y=curve_df["BuyHold"],
                                mode="lines",
                                name="Buy & Hold",
                                line=dict(color="#ff6b6b", width=1.5, dash="dot"),
                            )
                        )
                        backtest_fig.update_layout(
                            margin=dict(l=0, r=0, t=30, b=0),
                            height=260,
                            plot_bgcolor="#0a0a0a",
                            paper_bgcolor="#0a0a0a",
                            font=dict(color="#7dd87d", family="'Share Tech Mono', monospace"),
                            title=dict(text="Macro Strategy Backtest", font=dict(size=16)),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            yaxis=dict(gridcolor="rgba(125, 216, 125, 0.08)"),
                            xaxis=dict(gridcolor="rgba(125, 216, 125, 0.08)"),
                        )
                        stats = backtest.get("stats", {})
                        stat_mappings = [
                            ("Hit Ratio", f"{stats.get('hit_ratio', 0):.1f}%"),
                            ("Trades", str(stats.get("trades", 0))),
                            ("Strategy CAGR", f"{stats.get('strategy_cagr', 0):.1f}%"),
                            ("Sharpe", f"{stats.get('sharpe', 0):.2f}"),
                        ]
                        for label, value in stat_mappings:
                            backtest_stats_cards.append(
                                html.Div(
                                    [
                                        html.Div(value, className="metric-value", style={"color": "#7dd87d"}),
                                        html.Div(label.upper(), className="metric-label"),
                                    ],
                                    className="metric-card",
                                )
                            )

                    research_notes = macro_result.get("research_notes", [])
                    macro_component = html.Div(
                        [
                            html.Div("MACRO RESEARCH LAB", className="metrics-section-title"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(f"{macro_result['score']:+.1f}", className="metric-value", style={"color": "#7dd87d"}),
                                            html.Div("Composite Score", className="metric-label"),
                                            html.Div(
                                                macro_result["regime"],
                                                style={"fontSize": "12px", "color": "#00bfff", "marginTop": "5px"},
                                            ),
                                        ],
                                        className="metric-card",
                                    ),
                                ],
                                className="metrics-grid",
                            ),
                            html.Div(factor_cards, className="metrics-grid"),
                            html.Div(weights_badges, className="metrics-grid") if weights_badges else html.Div(),
                            dcc.Graph(figure=macro_fig) if history_df is not None and not history_df.empty else html.Div(),
                            html.Div(backtest_stats_cards, className="metrics-grid") if backtest_stats_cards else html.Div(),
                            dcc.Graph(figure=backtest_fig) if backtest_stats_cards else html.Div(),
                            html.Div(
                                scenario_cards,
                                className="metrics-grid",
                                style={"marginTop": "15px"},
                            ),
                            html.Div(
                                [
                                    html.Div("Research Notes", style={"fontWeight": "600", "marginBottom": "5px"}),
                                    html.Ul(
                                        [
                                            html.Li(note, style={"fontSize": "12px", "color": "rgba(125, 216, 125, 0.8)"})
                                            for note in research_notes
                                        ]
                                    ),
                                ],
                                style={"marginTop": "15px"},
                            ),
                        ],
                        className="metrics-section",
                        style={
                            "background": "linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%)",
                            "border": "1px solid rgba(0, 191, 255, 0.4)",
                            "borderRadius": "6px",
                            "padding": "20px",
                            "margin": "20px 0",
                        },
                    )
            
            # Create educational content for indicators
            rsi_education = html.Div()
            macd_education = html.Div()
            stoch_education = html.Div()
            
            if INDICATOR_EDUCATION_AVAILABLE and len(df) > 0:
                # Get latest RSI value
                latest_rsi = metrics.get('rsi', 50) if 'rsi' in metrics else 50
                rsi_education = create_indicator_link("RSI", latest_rsi)
                
                # Get latest MACD value
                latest_macd = metrics.get('macd', 0) if 'macd' in metrics else 0
                macd_education = create_indicator_link("MACD", latest_macd)
                
                # Get latest Stochastic value
                latest_stoch = metrics.get('stoch_k', 50) if 'stoch_k' in metrics else 50
                stoch_education = create_indicator_link("Stochastic", latest_stoch)
            
            # Check alerts if enhancements available
            if ENHANCEMENTS_AVAILABLE:
                # Check RSI alerts
                if 'rsi' in metrics:
                    alert_system.check_alert(">", metrics['rsi'], 70, symbol, "warning")
                    alert_system.check_alert("<", metrics['rsi'], 30, symbol, "info")
                
                # Check price change alerts
                if 'change_pct' in metrics:
                    alert_system.check_alert(">", abs(metrics['change_pct']), 2.0, symbol, "info")
                
                # Store data for export
                data_store = {"df": df.to_dict('records') if not df.empty else []}
                metrics_store = metrics
            else:
                data_store = {}
                metrics_store = {}
            
            # Return values matching the callback outputs
            base_returns = (
                fig_price,
                fig_rsi,
                fig_macd,
                fig_stoch,
                fig_volume,
                market_status_banner,
                trading_signal_display,
                metrics_cards,
                ml_component,
                news_component,
                econ_component,
                macro_component,
                rsi_education,
                macd_education,
                stoch_education,
            )
            
            if ENHANCEMENTS_AVAILABLE:
                return base_returns + (data_store, metrics_store)
            else:
                return base_returns
        except Exception as e:
            print(f"[ERROR] Exception in update_all: {e}")
            import traceback
            traceback.print_exc()
            error_fig = {
                "data": [],
                "layout": {
                    "title": f"Error loading data: {str(e)}",
                    "plot_bgcolor": "#0a0a0a",
                    "paper_bgcolor": "#0a0a0a",
                }
            }
            error_msg = html.Div(
                f"Error: {str(e)}",
                style={"color": "#ff6b6b", "textAlign": "center", "padding": "20px"}
            )
            # Error return values matching the callback outputs
            error_base_returns = (
                error_fig,
                error_fig,
                error_fig,
                error_fig,
                error_fig,
                html.Div(),
                error_msg,
                error_msg,
                error_msg,
                error_msg,
                error_msg,
                html.Div(),
                html.Div(),
                html.Div(),
                html.Div(),
            )
            
            if ENHANCEMENTS_AVAILABLE:
                return error_base_returns + ({}, {})
            else:
                return error_base_returns
    
    # Enhancement callbacks
    if ENHANCEMENTS_AVAILABLE:
        @dash_app.callback(
            Output("export-status", "children"),
            [Input("export-csv-btn", "n_clicks"),
             Input("export-json-btn", "n_clicks")],
            [State("current-data-store", "data"),
             State("current-metrics-store", "data")],
            prevent_initial_call=True,
        )
        def handle_export(csv_clicks, json_clicks, data_store, metrics_store):
            from dash import callback_context
            ctx = callback_context
            if not ctx.triggered:
                return ""
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            try:
                if button_id == "export-csv-btn" and csv_clicks:
                    if data_store and "df" in data_store:
                        import pandas as pd
                        df = pd.DataFrame(data_store["df"])
                        filepath = data_exporter.export_to_csv(df)
                        return f"Exported to: {filepath}"
                    return "No data available to export"
                
                elif button_id == "export-json-btn" and json_clicks:
                    if data_store and "df" in data_store:
                        import pandas as pd
                        df = pd.DataFrame(data_store["df"])
                        filepath = data_exporter.export_to_json(df)
                        return f"Exported to: {filepath}"
                    return "No data available to export"
            except Exception as e:
                return f"Export error: {str(e)}"
            
            return ""
        
        @dash_app.callback(
            Output("alerts-container", "children"),
            [Input("interval", "n_intervals")],
            [State("symbol-dropdown", "value"),
             State("metrics-container", "children")],
            prevent_initial_call=False,
        )
        def update_alerts(n_intervals, symbol, metrics_children):
            if not symbol:
                return html.Div()
            
            # Check common alert conditions
            # This would need access to current metrics - simplified for now
            active_alerts = alert_system.get_active_alerts(limit=5)
            
            if not active_alerts:
                return html.Div()
            
            alert_items = []
            for alert in active_alerts:
                alert_color = {
                    "info": "#00bfff",
                    "warning": "#ffa500",
                    "danger": "#ff6b6b",
                    "success": "#7dd87d"
                }.get(alert["type"], "#7dd87d")
                
                alert_items.append(
                    html.Div(
                        [
                            html.Div(alert["message"], style={"color": alert_color, "fontSize": "12px"}),
                            html.Div(
                                alert["timestamp"][:19],
                                style={"color": "rgba(125, 216, 125, 0.6)", "fontSize": "10px", "marginTop": "3px"}
                            ),
                        ],
                        style={
                            "padding": "10px",
                            "border": f"1px solid {alert_color}",
                            "borderRadius": "4px",
                            "marginBottom": "5px",
                            "backgroundColor": "rgba(10, 10, 10, 0.8)",
                        }
                    )
                )
            
            return html.Div(
                [
                    html.Div("ACTIVE ALERTS", className="metrics-section-title"),
                    html.Div(alert_items),
                ],
                className="metrics-section",
                style={
                    "background": "linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%)",
                    "border": "1px solid rgba(255, 165, 0, 0.4)",
                    "borderRadius": "6px",
                    "padding": "20px",
                    "margin": "20px 0",
                },
            )
        
        @dash_app.callback(
            Output("portfolio-container", "children"),
            [Input("interval", "n_intervals")],
            prevent_initial_call=False,
        )
        def update_portfolio(n_intervals):
            positions = portfolio_tracker.get_positions()
            
            if not positions:
                return html.Div(
                    [
                        html.Div("PORTFOLIO TRACKER", className="metrics-section-title"),
                        html.Div(
                            "No positions tracked. Add positions via API or manual entry.",
                            style={"color": "rgba(125, 216, 125, 0.6)", "textAlign": "center", "padding": "20px"},
                        ),
                    ],
                    className="metrics-section",
                    style={
                        "background": "linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%)",
                        "border": "1px solid rgba(138, 43, 226, 0.4)",
                        "borderRadius": "6px",
                        "padding": "20px",
                        "margin": "20px 0",
                    },
                )
            
            position_cards = []
            for symbol, pos in positions.items():
                position_cards.append(
                    html.Div(
                        [
                            html.Div(symbol, style={"fontWeight": "600", "color": "#7dd87d", "marginBottom": "5px"}),
                            html.Div(f"Qty: {pos['quantity']:.2f}", style={"fontSize": "12px", "color": "rgba(125, 216, 125, 0.8)"}),
                            html.Div(f"Entry: ${pos['entry_price']:.2f}", style={"fontSize": "12px", "color": "rgba(125, 216, 125, 0.8)"}),
                        ],
                        className="metric-card",
                    )
                )
            
            return html.Div(
                [
                    html.Div("PORTFOLIO TRACKER", className="metrics-section-title"),
                    html.Div(position_cards, className="metrics-grid"),
                ],
                className="metrics-section",
                style={
                    "background": "linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%)",
                    "border": "1px solid rgba(138, 43, 226, 0.4)",
                    "borderRadius": "6px",
                    "padding": "20px",
                    "margin": "20px 0",
                },
            )
    
    # Paper Trading Callbacks
    if PAPER_TRADING_AVAILABLE:
        @dash_app.callback(
            Output("paper-trading-container", "children"),
            [
                Input("interval", "n_intervals"),
                Input("place-order-btn", "n_clicks"),
            ],
            [
                State("symbol-dropdown", "value"),
                State("order-side", "value"),
                State("order-quantity", "value"),
                State("order-price", "value"),
                State("current-data-store", "data"),
            ],
            prevent_initial_call=False,
        )
        def update_paper_trading(n_intervals, place_order_clicks, symbol, order_side, order_quantity, order_price, data_store):
            from dash import callback_context
            
            ctx = callback_context
            error_msg = ""
            success_msg = ""
            
            # Handle order placement
            if ctx.triggered and "place-order-btn" in ctx.triggered[0]["prop_id"] and place_order_clicks:
                try:
                    if not symbol or not order_side or not order_quantity:
                        error_msg = "Please fill in all order fields"
                    else:
                        # Get current price from data store or use provided price
                        current_price = order_price
                        if not current_price and data_store and "df" in data_store:
                            import pandas as pd
                            df = pd.DataFrame(data_store["df"])
                            if not df.empty and "Close" in df.columns:
                                current_price = float(df["Close"].iloc[-1])
                        
                        if not current_price:
                            error_msg = "Unable to determine current price. Please enter a price."
                        else:
                            # Place order
                            order = paper_trading.place_order(
                                symbol=symbol,
                                side=order_side,
                                quantity=float(order_quantity),
                                price=current_price,
                                order_type="MARKET"
                            )
                            success_msg = f"Order {order.order_id} {order.side} {order.quantity} {symbol} @ ${order.filled_price:.2f}"
                except Exception as e:
                    error_msg = f"Order error: {str(e)}"
            
            # Get current prices for portfolio update
            current_prices = {}
            if data_store and "df" in data_store:
                import pandas as pd
                df = pd.DataFrame(data_store["df"])
                if not df.empty and "Close" in df.columns and symbol:
                    current_prices[symbol] = float(df["Close"].iloc[-1])
            
            # Update positions with current prices
            if current_prices:
                paper_trading.update_positions(current_prices)
            
            # Get portfolio value
            portfolio_value = paper_trading.get_portfolio_value(current_prices)
            positions = paper_trading.get_positions()
            recent_orders = paper_trading.get_recent_orders(limit=10)
            
            # Build UI
            portfolio_cards = [
                html.Div(
                    [
                        html.Div(f"${portfolio_value['cash']:,.2f}", className="metric-value", style={"color": "#7dd87d"}),
                        html.Div("Available Cash", className="metric-label"),
                    ],
                    className="metric-card",
                ),
                html.Div(
                    [
                        html.Div(f"${portfolio_value['total_value']:,.2f}", className="metric-value", 
                                style={"color": "#00bfff"}),
                        html.Div("Total Value", className="metric-label"),
                    ],
                    className="metric-card",
                ),
                html.Div(
                    [
                        html.Div(
                            f"${portfolio_value['total_pnl']:+,.2f} ({portfolio_value['total_pnl_pct']:+.2f}%)",
                            className="metric-value",
                            style={"color": "#00c853" if portfolio_value['total_pnl'] >= 0 else "#ff1744"},
                        ),
                        html.Div("Total P&L", className="metric-label"),
                    ],
                    className="metric-card",
                ),
                html.Div(
                    [
                        html.Div(f"{portfolio_value['num_positions']}", className="metric-value", style={"color": "#ff1493"}),
                        html.Div("Open Positions", className="metric-label"),
                    ],
                    className="metric-card",
                ),
            ]
            
            # Order Entry Form
            order_form = html.Div(
                [
                    html.Div("PLACE ORDER", style={"fontSize": "14px", "fontWeight": "600", "color": "#7dd87d", "marginBottom": "15px"}),
                    html.Div(
                        [
                            html.Label("Side:", style={"color": "rgba(125, 216, 125, 0.8)", "marginRight": "10px", "width": "80px", "display": "inline-block"}),
                            dcc.RadioItems(
                                id="order-side",
                                options=[
                                    {"label": "BUY", "value": "BUY"},
                                    {"label": "SELL", "value": "SELL"},
                                ],
                                value="BUY",
                                inline=True,
                                style={"display": "inline-block", "color": "#7dd87d"},
                            ),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    html.Div(
                        [
                            html.Label("Quantity:", style={"color": "rgba(125, 216, 125, 0.8)", "marginRight": "10px", "width": "80px", "display": "inline-block"}),
                            dcc.Input(
                                id="order-quantity",
                                type="number",
                                value=1,
                                min=0.01,
                                step=0.01,
                                style={"width": "120px", "backgroundColor": "#1a1a1a", "color": "#7dd87d", "border": "1px solid rgba(125, 216, 125, 0.3)", "padding": "5px"},
                            ),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    html.Div(
                        [
                            html.Label("Price (optional):", style={"color": "rgba(125, 216, 125, 0.8)", "marginRight": "10px", "width": "80px", "display": "inline-block"}),
                            dcc.Input(
                                id="order-price",
                                type="number",
                                value=None,
                                min=0,
                                step=0.01,
                                placeholder="Market Price",
                                style={"width": "120px", "backgroundColor": "#1a1a1a", "color": "#7dd87d", "border": "1px solid rgba(125, 216, 125, 0.3)", "padding": "5px"},
                            ),
                        ],
                        style={"marginBottom": "15px"},
                    ),
                    html.Button(
                        "Place Order",
                        id="place-order-btn",
                        n_clicks=0,
                        style={
                            "backgroundColor": "rgba(125, 216, 125, 0.2)",
                            "border": "1px solid rgba(125, 216, 125, 0.5)",
                            "color": "#7dd87d",
                            "padding": "10px 30px",
                            "borderRadius": "6px",
                            "cursor": "pointer",
                            "fontSize": "14px",
                            "fontWeight": "600",
                        },
                    ),
                    html.Div(success_msg, style={"color": "#7dd87d", "marginTop": "10px", "fontSize": "12px"}) if success_msg else html.Div(),
                    html.Div(error_msg, style={"color": "#ff6b6b", "marginTop": "10px", "fontSize": "12px"}) if error_msg else html.Div(),
                ],
                style={
                    "padding": "20px",
                    "border": "1px solid rgba(125, 216, 125, 0.3)",
                    "borderRadius": "6px",
                    "marginBottom": "20px",
                },
            )
            
            # Positions Display
            positions_display = html.Div()
            if positions:
                position_items = []
                for pos in positions:
                    pnl_color = "#00c853" if pos["unrealized_pnl"] >= 0 else "#ff1744"
                    position_items.append(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(pos["symbol"], style={"fontWeight": "600", "color": "#7dd87d", "fontSize": "14px"}),
                                        html.Div(f"Qty: {pos['quantity']:.2f}", style={"fontSize": "12px", "color": "rgba(125, 216, 125, 0.8)", "marginTop": "5px"}),
                                    ],
                                    style={"flex": "1"},
                                ),
                                html.Div(
                                    [
                                        html.Div(f"Entry: ${pos['avg_entry_price']:.2f}", style={"fontSize": "11px", "color": "rgba(125, 216, 125, 0.7)"}),
                                        html.Div(f"Current: ${pos['current_price']:.2f}", style={"fontSize": "11px", "color": "rgba(125, 216, 125, 0.7)", "marginTop": "3px"}),
                                    ],
                                    style={"flex": "1", "textAlign": "right"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            f"${pos['unrealized_pnl']:+,.2f}",
                                            style={"fontSize": "14px", "fontWeight": "600", "color": pnl_color},
                                        ),
                                        html.Div(
                                            f"({pos['unrealized_pnl_pct']:+.2f}%)",
                                            style={"fontSize": "11px", "color": pnl_color, "marginTop": "3px"},
                                        ),
                                    ],
                                    style={"flex": "1", "textAlign": "right"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "padding": "15px",
                                "borderBottom": "1px solid rgba(125, 216, 125, 0.2)",
                                "backgroundColor": "rgba(10, 10, 10, 0.5)",
                            },
                        )
                    )
                
                positions_display = html.Div(
                    [
                        html.Div("OPEN POSITIONS", className="metrics-section-title"),
                        html.Div(position_items),
                    ],
                    style={"marginTop": "20px"},
                )
            
            # Recent Orders
            orders_display = html.Div()
            if recent_orders:
                order_items = []
                for order in recent_orders[-5:]:  # Show last 5
                    side_color = "#00c853" if order["side"] == "BUY" else "#ff1744"
                    order_items.append(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(order["symbol"], style={"fontWeight": "600", "color": "#7dd87d"}),
                                        html.Div(
                                            order["timestamp"][:19],
                                            style={"fontSize": "10px", "color": "rgba(125, 216, 125, 0.6)", "marginTop": "3px"},
                                        ),
                                    ],
                                    style={"flex": "1"},
                                ),
                                html.Div(
                                    [
                                        html.Div(order["side"], style={"color": side_color, "fontWeight": "600"}),
                                        html.Div(f"{order['filled_quantity']:.2f} @ ${order['filled_price']:.2f}", 
                                                style={"fontSize": "11px", "color": "rgba(125, 216, 125, 0.8)", "marginTop": "3px"}),
                                    ],
                                    style={"flex": "1", "textAlign": "right"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "padding": "10px",
                                "borderBottom": "1px solid rgba(125, 216, 125, 0.1)",
                            },
                        )
                    )
                
                orders_display = html.Div(
                    [
                        html.Div("RECENT ORDERS", className="metrics-section-title"),
                        html.Div(order_items),
                    ],
                    style={"marginTop": "20px"},
                )
            
            return html.Div(
                [
                    html.Div("PAPER TRADING", className="metrics-section-title"),
                    html.Div(portfolio_cards, className="metrics-grid"),
                    html.Div(
                        [
                            html.Div([order_form], style={"flex": "1", "marginRight": "20px"}),
                            html.Div([positions_display, orders_display], style={"flex": "1"}),
                        ],
                        style={"display": "flex", "marginTop": "20px"},
                    ),
                ],
                className="metrics-section",
                style={
                    "background": "linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%)",
                    "border": "1px solid rgba(125, 216, 125, 0.4)",
                    "borderRadius": "6px",
                    "padding": "20px",
                    "margin": "20px 0",
                },
            )

    return dash_app

# ---------- MODAL ENTRY POINT ----------
if MODAL_AVAILABLE:
    @app.asgi_app(image=image)
    def dashapp():
        """ASGI entry point for Modal deployment."""
        # Create Dash app inside the function (not at module level)
        dash_instance = create_dash_app()
        
        # Convert WSGI (Dash) to ASGI using asgiref
        asgi_app = WsgiToAsgi(dash_instance.server)
        
        return asgi_app

# ---------- LOCAL TEST / RAILWAY DEPLOYMENT ----------
if __name__ == "__main__":
    import os
    dash_instance = create_dash_app()
    # Railway provides PORT environment variable, default to 8050 for local
    port = int(os.environ.get("PORT", 8050))
    # Disable debug in production (Railway sets RAILWAY_ENVIRONMENT)
    debug = os.environ.get("RAILWAY_ENVIRONMENT") != "production"
    dash_instance.run(host="0.0.0.0", port=port, debug=debug)

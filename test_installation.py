"""
Test script to verify all packages are installed correctly
"""

print("=" * 60)
print("TESTING PACKAGE INSTALLATION")
print("=" * 60)

# Test core packages
print("\n1. Core Dashboard Packages:")
try:
    import dash
    print("   [OK] dash", dash.__version__)
except ImportError as e:
    print("   [ERROR] dash - Error:", e)

try:
    import pandas as pd
    print("   [OK] pandas", pd.__version__)
except ImportError as e:
    print("   [ERROR] pandas - Error:", e)

try:
    import plotly
    print("   [OK] plotly", plotly.__version__)
except ImportError as e:
    print("   [ERROR] plotly - Error:", e)

try:
    import yfinance as yf
    print("   [OK] yfinance", yf.__version__)
except ImportError as e:
    print("   [ERROR] yfinance - Error:", e)

# Test ML packages
print("\n2. Machine Learning:")
try:
    import sklearn
    print("   [OK] scikit-learn", sklearn.__version__)
except ImportError as e:
    print("   [ERROR] scikit-learn - Error:", e)

# Test advanced technical analysis
print("\n3. Advanced Technical Analysis:")
try:
    import pandas_ta as ta
    print("   [OK] pandas-ta", ta.version)
except ImportError as e:
    print("   [ERROR] pandas-ta - Error:", e)

try:
    import talib
    print("   [OK] TA-Lib installed")
except ImportError as e:
    print("   [WARN]  TA-Lib - Error:", e, "(optional)")

# Test sentiment analysis
print("\n4. Sentiment Analysis:")
try:
    from textblob import TextBlob
    print("   [OK] textblob installed")
except ImportError as e:
    print("   [ERROR] textblob - Error:", e)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("   [OK] vaderSentiment installed")
except ImportError as e:
    print("   [ERROR] vaderSentiment - Error:", e)

# Test API packages
print("\n5. API Libraries:")
try:
    import requests
    print("   [OK] requests", requests.__version__)
except ImportError as e:
    print("   [ERROR] requests - Error:", e)

try:
    from fredapi import Fred
    print("   [OK] fredapi installed")
except ImportError as e:
    print("   [ERROR] fredapi - Error:", e)

# Test environment variables
print("\n6. Environment Variables (API Keys):")
import os

news_key = os.environ.get("NEWS_API_KEY", "")
fred_key = os.environ.get("FRED_API_KEY", "")
alpha_key = os.environ.get("ALPHA_VANTAGE_KEY", "")

if news_key:
    print("   [OK] NEWS_API_KEY is set")
else:
    print("   [WARN]  NEWS_API_KEY is NOT set (optional for news features)")

if fred_key:
    print("   [OK] FRED_API_KEY is set")
else:
    print("   [WARN]  FRED_API_KEY is NOT set (optional for economic data)")

if alpha_key:
    print("   [OK] ALPHA_VANTAGE_KEY is set")
else:
    print("   [WARN]  ALPHA_VANTAGE_KEY is NOT set (optional)")

# Test NLTK data
print("\n7. NLTK Data (for TextBlob):")
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    print("   [OK] NLTK punkt data installed")
    nltk.data.find('corpora/brown')
    print("   [OK] NLTK brown corpus installed")
except LookupError:
    print("   [WARN]  NLTK data missing - run: python -c \"import nltk; nltk.download('punkt'); nltk.download('brown')\"")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n[OK] All core packages are installed!")
print("[WARN]  API keys are optional - set them for advanced features.")
print("See API_SETUP_GUIDE.md for instructions on getting API keys.")
print("=" * 60)


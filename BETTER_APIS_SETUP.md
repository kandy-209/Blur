# Better Web APIs Integration Guide

## 🎯 Overview

This guide shows you how to integrate premium and free APIs for better, more reliable market data than relying solely on Yahoo Finance.

## 📊 Recommended APIs (Ranked)

### 1. **Polygon.io** ⭐⭐⭐⭐⭐
**Best for: Real-time data, professional traders**

- **Free Tier**: Limited (sign up required)
- **Paid**: $29-199/month
- **Features**:
  - Real-time and historical data
  - Options data
  - News and fundamentals
  - WebSocket streaming
- **Rate Limits**: Depends on plan
- **Sign up**: https://polygon.io/

### 2. **Finnhub** ⭐⭐⭐⭐⭐
**Best for: Free tier users**

- **Free Tier**: 60 calls/minute (excellent!)
- **Paid**: $9.99-39.99/month
- **Features**:
  - Real-time quotes
  - Historical candles
  - Company news
  - Financials
  - Options data
- **Rate Limits**: 60 calls/minute (free)
- **Sign up**: https://finnhub.io/

### 3. **Twelve Data** ⭐⭐⭐⭐
**Best for: Historical data, multiple symbols**

- **Free Tier**: 8 calls/minute, 800/day
- **Paid**: $7.99-99.99/month
- **Features**:
  - Real-time and historical
  - Technical indicators
  - Multiple exchanges
  - Good documentation
- **Rate Limits**: 8 calls/minute (free)
- **Sign up**: https://twelvedata.com/

### 4. **Alpha Vantage** ⭐⭐⭐⭐
**Best for: Technical indicators**

- **Free Tier**: 5 calls/minute, 500/day
- **Paid**: $49.99-149.99/month
- **Features**:
  - 100+ technical indicators
  - Historical data
  - Fundamental data
  - Economic indicators
- **Rate Limits**: 5 calls/minute (free)
- **Sign up**: https://www.alphavantage.co/

### 5. **Financial Modeling Prep** ⭐⭐⭐⭐
**Best for: Fundamentals and financials**

- **Free Tier**: 250 calls/day
- **Paid**: $14-29/month
- **Features**:
  - Financial statements
  - Key metrics
  - Historical data
  - Company profiles
- **Rate Limits**: 250 calls/day (free)
- **Sign up**: https://site.financialmodelingprep.com/

### 6. **IEX Cloud** ⭐⭐⭐⭐
**Best for: US market data**

- **Free Tier**: Limited
- **Paid**: $9-999/month
- **Features**:
  - Real-time quotes
  - Historical data
  - Options
  - News
- **Sign up**: https://iexcloud.io/

## 🚀 Quick Setup

### Step 1: Get API Keys

1. Sign up for APIs (start with free tiers)
2. Get your API keys from each provider
3. Store them securely

### Step 2: Set Environment Variables

**Windows (PowerShell):**
```powershell
$env:POLYGON_API_KEY="your-polygon-key"
$env:FINNHUB_API_KEY="your-finnhub-key"
$env:TWELVE_DATA_API_KEY="your-twelve-data-key"
$env:ALPHA_VANTAGE_KEY="your-alpha-vantage-key"
$env:FMP_API_KEY="your-fmp-key"
```

**Linux/Mac:**
```bash
export POLYGON_API_KEY="your-polygon-key"
export FINNHUB_API_KEY="your-finnhub-key"
export TWELVE_DATA_API_KEY="your-twelve-data-key"
export ALPHA_VANTAGE_KEY="your-alpha-vantage-key"
export FMP_API_KEY="your-fmp-key"
```

**Or create `.env` file:**
```env
POLYGON_API_KEY=your-polygon-key
FINNHUB_API_KEY=your-finnhub-key
TWELVE_DATA_API_KEY=your-twelve-data-key
ALPHA_VANTAGE_KEY=your-alpha-vantage-key
FMP_API_KEY=your-fmp-key
```

### Step 3: Test APIs

```python
from api_manager import APIManager

manager = APIManager()

# Test data retrieval
df = manager.get_market_data("ES=F", interval="1min")
print(f"Retrieved {len(df)} data points")
```

## 💡 Usage Examples

### Example 1: Get Market Data with Fallback

```python
from api_manager import APIManager

manager = APIManager()

# Automatically tries APIs in priority order
# Falls back to Yahoo Finance if all fail
data = manager.get_market_data(
    symbol="ES=F",
    interval="1min",
    period="1d"
)
```

### Example 2: Get Real-Time Quote

```python
quote = manager.get_real_time_quote("ES=F")
print(f"Current Price: ${quote.get('c', 0):.2f}")
print(f"Change: {quote.get('d', 0):.2f} ({quote.get('dp', 0):.2f}%)")
```

### Example 3: Get News

```python
news = manager.get_news("ES=F")
for article in news[:5]:
    print(f"{article.get('headline')}")
    print(f"  {article.get('url')}")
```

### Example 4: Get Technical Indicators

```python
from api_manager import AlphaVantageAPI

av = AlphaVantageAPI(api_key="your-key")
rsi = av.get_technical_indicators(
    symbol="ES=F",
    indicator="RSI",
    interval="1min",
    time_period=14,
    series_type="close"
)
```

## 🔄 Integration with Existing System

### Update Data Collection Service

```python
from api_manager import APIManager

class DataCollectionService:
    def __init__(self):
        self.api_manager = APIManager()
    
    def collect_snapshot(self, symbol: str):
        # Use API manager instead of yfinance
        data = self.api_manager.get_market_data(
            symbol=symbol,
            interval="1min",
            period="1d"
        )
        # ... rest of collection logic
```

### Update Advanced Trading System

```python
from api_manager import APIManager

class AdvancedTradingSystem:
    def __init__(self):
        self.api_manager = APIManager()
        # ... rest of initialization
    
    def generate_signal(self, symbol: str, current_data=None):
        # Get data from API manager
        if current_data is None:
            current_data = self.api_manager.get_market_data(symbol)
        # ... rest of signal generation
```

## 📊 API Comparison

| Feature | Polygon | Finnhub | Twelve Data | Alpha Vantage | FMP |
|---------|---------|---------|-------------|---------------|-----|
| Real-time | ✅ | ✅ | ✅ | ⚠️ | ❌ |
| Historical | ✅ | ✅ | ✅ | ✅ | ✅ |
| Free Tier | Limited | ✅ Good | ✅ Good | ✅ | ✅ |
| Technical Indicators | ✅ | ❌ | ✅ | ✅✅ | ❌ |
| News | ✅ | ✅ | ❌ | ❌ | ❌ |
| Options | ✅ | ✅ | ❌ | ❌ | ❌ |
| Fundamentals | ✅ | ✅ | ❌ | ✅ | ✅✅ |
| WebSocket | ✅ | ✅ | ❌ | ❌ | ❌ |

## 🎯 Recommended Setup

### For Free Users:
1. **Finnhub** (primary) - 60 calls/min, great free tier
2. **Twelve Data** (backup) - 8 calls/min
3. **Alpha Vantage** (indicators) - 5 calls/min
4. **Yahoo Finance** (fallback) - Unlimited but less reliable

### For Paid Users:
1. **Polygon.io** (primary) - Best overall
2. **Finnhub** (backup) - Still useful
3. **Alpha Vantage** (indicators) - Best technical indicators

## 🔧 Rate Limiting

The API manager automatically handles rate limiting:

```python
# Alpha Vantage: 5 calls/minute = 12 seconds between calls
# Finnhub: 60 calls/minute = 1 second between calls
# Twelve Data: 8 calls/minute = 7.5 seconds between calls
```

## 🚨 Best Practices

1. **Start with Free Tiers**: Test APIs before paying
2. **Use Fallbacks**: Always have Yahoo Finance as backup
3. **Cache Data**: Don't make redundant API calls
4. **Monitor Usage**: Track API calls to stay within limits
5. **Error Handling**: APIs can fail, handle gracefully
6. **Rate Limiting**: Respect API limits

## 📈 Cost Estimates

### Free Tier (Recommended Start):
- **Finnhub**: 60 calls/min = 86,400/day (free)
- **Twelve Data**: 8 calls/min = 11,520/day (free)
- **Alpha Vantage**: 5 calls/min = 7,200/day (free)
- **Total Cost**: $0/month

### Paid Tier (Professional):
- **Polygon Starter**: $29/month
- **Finnhub Pro**: $9.99/month
- **Total**: ~$39/month for professional-grade data

## ✅ Benefits Over Yahoo Finance

1. **More Reliable**: Professional APIs have better uptime
2. **Real-Time**: True real-time data (not 15-min delayed)
3. **More Data**: Options, fundamentals, news
4. **Better Support**: Professional support from providers
5. **Rate Limits**: Predictable limits vs. Yahoo's unpredictable blocking
6. **WebSocket**: Real-time streaming available
7. **Historical**: Better historical data quality

## 🎉 Result

You now have access to:
- ✅ Multiple reliable data sources
- ✅ Automatic fallback mechanisms
- ✅ Real-time data capabilities
- ✅ Professional-grade APIs
- ✅ Better data quality
- ✅ More features (news, fundamentals, options)

The system automatically uses the best available API and falls back gracefully if one fails!


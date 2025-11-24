# API Setup Guide

All packages have been successfully installed! Now you need to set up API keys for the advanced features.

## ✅ Installed Packages

All the following packages are now installed:
- ✅ dash, pandas, plotly, yfinance (core dashboard)
- ✅ scikit-learn (machine learning predictions)
- ✅ pandas-ta (advanced technical analysis)
- ✅ textblob, vaderSentiment (sentiment analysis)
- ✅ fredapi (economic indicators)
- ✅ requests, beautifulsoup4 (web scraping)
- ✅ TA-Lib (technical analysis)

## 🔑 Setting Up API Keys

### 1. NewsAPI (Free Tier Available)

**Purpose**: Market news and sentiment analysis

1. Go to: https://newsapi.org/register
2. Sign up for a free account (500 requests/day)
3. Copy your API key
4. Set environment variable:
   
   **Windows PowerShell:**
   ```powershell
   $env:NEWS_API_KEY="your_api_key_here"
   ```
   
   **Windows CMD:**
   ```cmd
   set NEWS_API_KEY=your_api_key_here
   ```
   
   **Permanent (Windows):**
   - Search for "Environment Variables" in Windows
   - Click "Edit the system environment variables"
   - Click "Environment Variables"
   - Under "User variables", click "New"
   - Variable name: `NEWS_API_KEY`
   - Variable value: `your_api_key_here`
   - Click OK

### 2. FRED API (Free)

**Purpose**: Economic indicators (VIX, Treasury rates, GDP, Unemployment)

1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html
2. Sign up for a free account
3. Request an API key (instant approval)
4. Copy your API key
5. Set environment variable:
   
   **Windows PowerShell:**
   ```powershell
   $env:FRED_API_KEY="your_api_key_here"
   ```
   
   **Windows CMD:**
   ```cmd
   set FRED_API_KEY=your_api_key_here
   ```
   
   **Permanent (Windows):**
   - Follow same steps as NEWS_API_KEY above
   - Variable name: `FRED_API_KEY`

### 3. Alpha Vantage API (Optional - Free Tier Available)

**Purpose**: Additional market data and indicators

1. Go to: https://www.alphavantage.co/support/#api-key
2. Fill out the form (no email verification needed for free tier)
3. Copy your API key
4. Set environment variable:
   
   **Windows PowerShell:**
   ```powershell
   $env:ALPHA_VANTAGE_KEY="your_api_key_here"
   ```
   
   **Windows CMD:**
   ```cmd
   set ALPHA_VANTAGE_KEY=your_api_key_here
   ```

## 🧪 Testing Your Setup

Create a test file `test_apis.py`:

```python
import os

print("Checking API Keys...")
print("-" * 50)

news_key = os.environ.get("NEWS_API_KEY", "")
fred_key = os.environ.get("FRED_API_KEY", "")
alpha_key = os.environ.get("ALPHA_VANTAGE_KEY", "")

if news_key:
    print("✅ NEWS_API_KEY is set")
else:
    print("❌ NEWS_API_KEY is NOT set")

if fred_key:
    print("✅ FRED_API_KEY is set")
else:
    print("❌ FRED_API_KEY is NOT set")

if alpha_key:
    print("✅ ALPHA_VANTAGE_KEY is set")
else:
    print("⚠️  ALPHA_VANTAGE_KEY is NOT set (optional)")

print("-" * 50)
print("\nNote: The dashboard will work without API keys,")
print("but advanced features like news and economic data won't be available.")
```

Run it:
```bash
python test_apis.py
```

## 📝 Quick Setup Script

You can also create a `.env` file in your project root (if you use python-dotenv):

```bash
NEWS_API_KEY=your_news_api_key_here
FRED_API_KEY=your_fred_api_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

And load it in your script:
```python
from dotenv import load_dotenv
load_dotenv()
```

## 🚀 Running Your Dashboard

After setting up API keys (optional but recommended):

```bash
python dashboard.py
# or
python dashboard_modal.py  # if you restore this file
```

## 📊 What Works Without API Keys?

✅ **Works without API keys:**
- All technical indicators
- Price charts and graphs
- Trading signals
- ML predictions
- Basic analytics

❌ **Requires API keys:**
- Market news & sentiment (needs NEWS_API_KEY)
- Economic indicators (needs FRED_API_KEY)
- Additional market data (needs ALPHA_VANTAGE_KEY - optional)

## 🆘 Troubleshooting

### "Module not found" errors
- Run: `pip install -r requirements.txt`
- Make sure you're using the correct Python environment

### API keys not working
- Check that environment variables are set correctly
- Restart your terminal/IDE after setting environment variables
- Use `python test_apis.py` to verify

### TA-Lib errors
- If you see C library errors, try: `pip uninstall TA-Lib` then `pip install TA-Lib`

## 📚 Additional Resources

- NewsAPI Docs: https://newsapi.org/docs
- FRED API Docs: https://fred.stlouisfed.org/docs/api/
- Alpha Vantage Docs: https://www.alphavantage.co/documentation/


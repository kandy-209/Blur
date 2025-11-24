"""
Script to set API keys permanently on Windows
Run this once to configure your API keys
"""
import os
import sys

# Your API keys
API_KEYS = {
    "NEWS_API_KEY": "8328eeace3f44ae29593cbbbb922cc75",
    "FRED_API_KEY": "fa61c07493160b2463ef2eccc5e880db",
    "ALPHA_VANTAGE_KEY": "VLPGUTTKQKS7UWBW"
}

def set_api_keys():
    """Set API keys as environment variables"""
    print("Setting API keys...")
    
    for key_name, key_value in API_KEYS.items():
        os.environ[key_name] = key_value
        print(f"  [OK] {key_name} set")
    
    print("\n" + "="*60)
    print("API keys set for current session!")
    print("\nTo set them permanently:")
    print("1. Press Windows key and search for 'Environment Variables'")
    print("2. Click 'Edit the system environment variables'")
    print("3. Click 'Environment Variables' button")
    print("4. Under 'User variables', click 'New'")
    print("5. Add each key:")
    for key_name in API_KEYS.keys():
        print(f"   - Name: {key_name}")
        print(f"     Value: (your key)")
    print("="*60)
    
    # Test the keys
    print("\nTesting API keys...")
    test_api_keys()

def test_api_keys():
    """Test if API keys are working"""
    import requests
    
    # Test NewsAPI
    news_key = os.environ.get("NEWS_API_KEY")
    if news_key:
        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {"apiKey": news_key, "q": "stock", "pageSize": 1}
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                print("  [OK] NewsAPI: Working!")
            else:
                print(f"  [ERROR] NewsAPI: Error {response.status_code}")
        except Exception as e:
            print(f"  [ERROR] NewsAPI: {e}")
    else:
        print("  [ERROR] NewsAPI: Key not set")
    
    # Test FRED
    fred_key = os.environ.get("FRED_API_KEY")
    if fred_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            # Try to fetch a simple series
            data = fred.get_series("VIXCLS", observation_start='2024-01-01', limit=1)
            if len(data) > 0:
                print("  [OK] FRED API: Working!")
            else:
                print("  [ERROR] FRED API: No data returned")
        except Exception as e:
            print(f"  [ERROR] FRED API: {e}")
    else:
        print("  [ERROR] FRED API: Key not set")
    
    # Test Alpha Vantage
    alpha_key = os.environ.get("ALPHA_VANTAGE_KEY")
    if alpha_key:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": "IBM",
                "interval": "1min",
                "apikey": alpha_key
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "Error Message" not in data and "Note" not in data:
                    print("  [OK] Alpha Vantage: Working!")
                else:
                    print(f"  [ERROR] Alpha Vantage: {data.get('Error Message', data.get('Note', 'Unknown error'))}")
            else:
                print(f"  [ERROR] Alpha Vantage: Error {response.status_code}")
        except Exception as e:
            print(f"  [ERROR] Alpha Vantage: {e}")
    else:
        print("  [ERROR] Alpha Vantage: Key not set")

if __name__ == "__main__":
    set_api_keys()


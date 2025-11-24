"""
Quick verification script to check if your trading dashboard is ready to run.
Run this before starting the dashboard to catch common issues early.
"""

import sys
import importlib

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"[OK] {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"[X] {package_name or module_name} - {str(e)}")
        return False

def main():
    print("=" * 60)
    print("TRADING DASHBOARD - SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    # Core dependencies
    print("Checking core dependencies...")
    core_deps = [
        ("dash", "Dash"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("yfinance", "yfinance"),
        ("plotly", "Plotly"),
        ("pytz", "pytz"),
    ]
    
    core_ok = all(check_import(dep[0], dep[1]) for dep in core_deps)
    print()
    
    # Optional dependencies
    print("Checking optional dependencies...")
    optional_deps = [
        ("dash_auth", "dash-auth"),
        ("pandas_ta", "pandas-ta"),
        ("sklearn", "scikit-learn"),
        ("textblob", "TextBlob"),
        ("vaderSentiment", "vaderSentiment"),
        ("fredapi", "fredapi"),
    ]
    
    optional_ok = []
    for dep in optional_deps:
        if check_import(dep[0], dep[1]):
            optional_ok.append(dep[1])
    print()
    
    # Check custom modules
    print("Checking custom modules...")
    custom_modules = [
        ("macro_research_algo", "Macro Research Algorithm"),
        ("indicator_education", "Indicator Education"),
    ]
    
    custom_ok = all(check_import(mod[0], mod[1]) for mod in custom_modules)
    print()
    
    # Check main dashboard file
    print("Checking dashboard file...")
    try:
        with open("dashbord_blur.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "def create_dash_app" in content:
                print("[OK] dashbord_blur.py structure looks good")
            else:
                print("[!] dashbord_blur.py may be incomplete")
    except FileNotFoundError:
        print("[X] dashbord_blur.py not found!")
        custom_ok = False
    print()
    
    # Environment variables check
    print("Checking environment variables...")
    import os
    env_vars = {
        "DASH_ADMIN_USERNAME": "Dashboard authentication (optional)",
        "DASH_ADMIN_PASSWORD": "Dashboard authentication (optional)",
        "FRED_API_KEY": "Economic indicators (optional)",
        "NEWS_API_KEY": "Market news (optional)",
        "MACRO_WEIGHTS": "Macro research tuning (optional)",
    }
    
    for var, desc in env_vars.items():
        if os.environ.get(var):
            print(f"[OK] {var} is set - {desc}")
        else:
            print(f"[ ] {var} not set - {desc}")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if core_ok and custom_ok:
        print("[OK] Core setup is complete! You can run the dashboard.")
        print()
        print("To start the dashboard:")
        print("  python dashbord_blur.py")
        print()
        if optional_ok:
            print(f"[OK] Optional features available: {', '.join(optional_ok)}")
        else:
            print("[!] Some optional features are missing (dashboard will still work)")
    else:
        print("[X] Setup incomplete. Please install missing dependencies:")
        print()
        if not core_ok:
            print("  pip install -r requirements.txt")
        if not custom_ok:
            print("  Make sure macro_research_algo.py and indicator_education.py exist")
        print()
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main()


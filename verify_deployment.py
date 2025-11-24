"""
Deployment Verification Script
==============================
Run this locally to verify your app is ready for Railway deployment.
"""

import os
import sys

def check_file_exists(filename):
    """Check if a required file exists."""
    if os.path.exists(filename):
        print(f"[OK] {filename} exists")
        return True
    else:
        print(f"[X] {filename} missing!")
        return False

def check_file_content(filename, required_strings):
    """Check if file contains required strings."""
    if not os.path.exists(filename):
        print(f"❌ {filename} not found")
        return False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='latin-1') as f:
            content = f.read()
    
    all_found = True
    for req in required_strings:
        if req in content:
            print(f"[OK] {filename} contains: {req}")
        else:
            print(f"[X] {filename} missing: {req}")
            all_found = False
    
    return all_found

def main():
    print("=" * 60)
    print("RAILWAY DEPLOYMENT VERIFICATION")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check required files
    print("Checking required files...")
    print("-" * 60)
    required_files = [
        "dashbord_blur.py",
        "Procfile",
        "railway.toml",
        "requirements.txt",
        ".gitignore"
    ]
    
    for file in required_files:
        if not check_file_exists(file):
            all_ok = False
    
    print()
    
    # Check Procfile
    print("Checking Procfile...")
    print("-" * 60)
    if not check_file_content("Procfile", ["web:", "dashbord_blur.py"]):
        all_ok = False
    
    print()
    
    # Check railway.toml
    print("Checking railway.toml...")
    print("-" * 60)
    if not check_file_content("railway.toml", ["startCommand", "dashbord_blur.py"]):
        all_ok = False
    
    print()
    
    # Check main app file
    print("Checking dashbord_blur.py...")
    print("-" * 60)
    if not check_file_content("dashbord_blur.py", ["PORT", "0.0.0.0"]):
        all_ok = False
    
    print()
    
    # Check requirements.txt
    print("Checking requirements.txt...")
    print("-" * 60)
    if not check_file_content("requirements.txt", ["dash", "pandas", "plotly", "yfinance"]):
        all_ok = False
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("[OK] ALL CHECKS PASSED!")
        print()
        print("Your app is ready for Railway deployment!")
        print()
        print("Next steps:")
        print("1. Go to https://railway.app")
        print("2. Create new project")
        print("3. Deploy from GitHub repo: kandy-209/Blur")
        print("4. Wait for deployment to complete")
        print("5. Visit your live URL!")
        print()
        print("See RAILWAY_DEPLOYMENT_STEPS.md for detailed instructions.")
    else:
        print("[X] SOME CHECKS FAILED!")
        print()
        print("Please fix the issues above before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()


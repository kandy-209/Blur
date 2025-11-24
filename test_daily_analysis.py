#!/usr/bin/env python3
"""
Test script for Daily Analysis Logger
Tests analysis generation, storage, and retrieval
"""

import sys
from datetime import datetime
from daily_analysis_logger import (
    generate_comprehensive_analysis,
    save_daily_analysis,
    get_daily_analyses,
    get_all_analysis_dates,
    is_during_analysis_hours,
    get_today_date_str
)

def test_analysis_generation():
    """Test analysis generation."""
    print("=" * 60)
    print("TESTING ANALYSIS GENERATION")
    print("=" * 60)
    
    print("\n1. Testing rule-based analysis generation...")
    try:
        analysis = generate_comprehensive_analysis(
            symbols=["ES=F", "NQ=F", "GC=F"],
            use_llm=False  # Use rule-based for testing
        )
        
        print(f"   [OK] Analysis generated successfully")
        print(f"   Date: {analysis.get('date')}")
        print(f"   Time: {analysis.get('time_est')}")
        print(f"   Type: {analysis.get('analysis_type')}")
        print(f"   Symbols analyzed: {len(analysis.get('market_data', {}).get('symbols', {}))}")
        
        # Check written analysis
        written = analysis.get('written_analysis', '')
        if written:
            print(f"   [OK] Written analysis generated ({len(written)} characters)")
            print(f"   Preview: {written[:200]}...")
        else:
            print(f"   [WARNING] No written analysis generated")
        
        return analysis
    except Exception as e:
        print(f"   [ERROR] Analysis generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_analysis_storage(analysis):
    """Test analysis storage."""
    print("\n2. Testing analysis storage...")
    
    if not analysis:
        print("   [SKIP] No analysis to save")
        return None
    
    try:
        filename = save_daily_analysis(analysis)
        print(f"   [OK] Analysis saved to: {filename}")
        return filename
    except Exception as e:
        print(f"   [ERROR] Storage failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_analysis_retrieval():
    """Test analysis retrieval."""
    print("\n3. Testing analysis retrieval...")
    
    try:
        today = get_today_date_str()
        analyses = get_daily_analyses(today)
        
        if analyses:
            print(f"   [OK] Retrieved {len(analyses)} analyses for {today}")
            for idx, anal in enumerate(analyses):
                print(f"      Analysis {idx+1}: {anal.get('time_est')} ({anal.get('analysis_type')})")
        else:
            print(f"   [INFO] No analyses found for {today}")
        
        # Test date listing
        all_dates = get_all_analysis_dates()
        print(f"   [OK] Found analyses for {len(all_dates)} dates")
        if all_dates:
            print(f"      Recent dates: {', '.join(all_dates[:5])}")
        
        return analyses
    except Exception as e:
        print(f"   [ERROR] Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_time_utilities():
    """Test time utility functions."""
    print("\n4. Testing time utilities...")
    
    try:
        is_during = is_during_analysis_hours()
        print(f"   [OK] During analysis hours: {is_during}")
        
        today = get_today_date_str()
        print(f"   [OK] Today's date string: {today}")
        
        return True
    except Exception as e:
        print(f"   [ERROR] Time utilities failed: {e}")
        return False

def test_llm_integration():
    """Test LLM integration (if API keys available)."""
    print("\n5. Testing LLM integration...")
    
    import os
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    
    if has_openai:
        print("   [INFO] OpenAI API key detected")
    if has_anthropic:
        print("   [INFO] Anthropic API key detected")
    
    if not (has_openai or has_anthropic):
        print("   [INFO] No LLM API keys found - will use rule-based analysis")
        return False
    
    print("   [INFO] LLM integration available - test with use_llm=True")
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DAILY ANALYSIS LOGGER TEST SUITE")
    print("=" * 60)
    
    # Test 1: Analysis generation
    analysis = test_analysis_generation()
    
    # Test 2: Storage
    filename = test_analysis_storage(analysis)
    
    # Test 3: Retrieval
    analyses = test_analysis_retrieval()
    
    # Test 4: Time utilities
    test_time_utilities()
    
    # Test 5: LLM integration
    test_llm_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if analysis:
        print("[OK] Analysis generation: PASSED")
    else:
        print("[FAIL] Analysis generation: FAILED")
    
    if filename:
        print("[OK] Analysis storage: PASSED")
    else:
        print("[FAIL] Analysis storage: FAILED")
    
    if analyses is not None:
        print("[OK] Analysis retrieval: PASSED")
    else:
        print("[FAIL] Analysis retrieval: FAILED")
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()


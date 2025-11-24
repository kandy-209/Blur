#!/usr/bin/env python3
"""
Background Scheduler for Automated Daily Analysis
Runs analysis during opening hours (9:30 AM - 11:30 AM EST)
"""

import schedule
import time
import threading
from datetime import datetime
import pytz
from daily_analysis_logger import run_automated_analysis, is_during_analysis_hours

def run_analysis_job():
    """Job to run market analysis."""
    print(f"[{datetime.now()}] Running scheduled market analysis...")
    try:
        analysis = run_automated_analysis(
            symbols=["ES=F", "NQ=F", "GC=F"],
            force=False
        )
        if analysis:
            print(f"[{datetime.now()}] Analysis completed and saved.")
        else:
            print(f"[{datetime.now()}] Analysis skipped (not during analysis hours).")
    except Exception as e:
        print(f"[{datetime.now()}] Error running analysis: {e}")

def start_scheduler():
    """Start the background scheduler."""
    # Schedule analysis every 15 minutes during market hours
    # This will only run if is_during_analysis_hours() returns True
    schedule.every(15).minutes.do(run_analysis_job)
    
    # Also run at market open (9:30 AM EST)
    schedule.every().monday.at("09:30").do(run_analysis_job)
    schedule.every().tuesday.at("09:30").do(run_analysis_job)
    schedule.every().wednesday.at("09:30").do(run_analysis_job)
    schedule.every().thursday.at("09:30").do(run_analysis_job)
    schedule.every().friday.at("09:30").do(run_analysis_job)
    
    print("Analysis scheduler started. Running every 15 minutes during 9:30 AM - 11:30 AM EST.")
    
    # Run scheduler in background thread
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    return scheduler_thread

if __name__ == "__main__":
    print("Starting analysis scheduler...")
    start_scheduler()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nScheduler stopped.")


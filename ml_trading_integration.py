#!/usr/bin/env python3
"""
Integration Script for ML Trading System
Connects data collection, ML training, and signal generation
"""

import asyncio
import schedule
import time
import threading
from datetime import datetime
from advanced_ml_trading_system import AdvancedTradingSystem, TradingSignal
from data_collection_service import DataCollectionService
from daily_analysis_logger import generate_comprehensive_analysis, save_daily_analysis
import yfinance as yf
import pandas as pd

class MLTradingIntegration:
    """Complete integration of all components."""
    
    def __init__(self):
        self.system = AdvancedTradingSystem()
        self.data_service = DataCollectionService()
        self.training_interval_days = 7  # Retrain weekly
        self.last_training_date = None
    
    def initialize_system(self):
        """Initialize the complete system."""
        print("Initializing ML Trading System...")
        
        # Check if we have enough data to train
        print("Checking data availability...")
        df = self.system.db.get_training_data(min_samples=2000)
        
        if len(df) >= 2000:
            print(f"Found {len(df)} samples. Training models...")
            success = self.system.train_models(min_samples=2000)
            if success:
                print("Models trained successfully!")
                self.system.is_trained = True
                self.last_training_date = datetime.now().date()
            else:
                print("Training failed. Will retry when more data is available.")
        else:
            print(f"Only {len(df)} samples available. Need at least 2000 for training.")
            print("Collecting more data...")
    
    def collect_data_job(self):
        """Job to collect market data."""
        print(f"[{datetime.now()}] Collecting market data...")
        self.data_service.collect_all_symbols()
    
    def generate_signals_job(self):
        """Job to generate trading signals."""
        if not self.system.is_trained:
            print("Models not trained yet. Skipping signal generation.")
            return
        
        print(f"[{datetime.now()}] Generating trading signals...")
        
        for symbol in self.data_service.symbols:
            try:
                # Get recent data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d", interval="1m")
                
                if data.empty or len(data) < 100:
                    continue
                
                # Convert to format expected by system
                data = data.reset_index()
                if 'Datetime' in data.columns:
                    data = data.rename(columns={'Datetime': 'timestamp'})
                    data = data.set_index('timestamp')
                
                # Generate signal
                signal = self.system.generate_signal(symbol, data)
                
                print(f"  {symbol}: {signal.signal} (Confidence: {signal.confidence:.2%})")
                print(f"    Entry: ${signal.entry_price:.2f}")
                if signal.stop_loss:
                    print(f"    Stop Loss: ${signal.stop_loss:.2f}")
                if signal.take_profit:
                    print(f"    Take Profit: ${signal.take_profit:.2f}")
                print(f"    Reasoning: {signal.reasoning[:100]}...")
                
            except Exception as e:
                print(f"Error generating signal for {symbol}: {e}")
    
    def retrain_models_job(self):
        """Job to retrain models periodically."""
        if self.last_training_date:
            days_since_training = (datetime.now().date() - self.last_training_date).days
            if days_since_training < self.training_interval_days:
                return
        
        print(f"[{datetime.now()}] Retraining models...")
        df = self.system.db.get_training_data(min_samples=2000)
        
        if len(df) >= 2000:
            success = self.system.train_models(min_samples=2000)
            if success:
                self.last_training_date = datetime.now().date()
                print("Models retrained successfully!")
            else:
                print("Retraining failed.")
        else:
            print(f"Insufficient data for retraining ({len(df)} samples)")
    
    def daily_analysis_job(self):
        """Job to generate daily analysis."""
        print(f"[{datetime.now()}] Generating daily analysis...")
        try:
            analysis = generate_comprehensive_analysis(
                symbols=self.data_service.symbols,
                use_llm=bool(self.system.rag.encoder is not None)
            )
            save_daily_analysis(analysis)
            print("Daily analysis saved.")
        except Exception as e:
            print(f"Error generating daily analysis: {e}")
    
    def start_scheduler(self):
        """Start all scheduled jobs."""
        # Data collection every minute during market hours
        schedule.every(1).minutes.do(self.collect_data_job)
        
        # Signal generation every 5 minutes
        schedule.every(5).minutes.do(self.generate_signals_job)
        
        # Retrain models weekly (Sunday night)
        schedule.every().sunday.at("20:00").do(self.retrain_models_job)
        
        # Daily analysis at market open
        schedule.every().monday.at("09:30").do(self.daily_analysis_job)
        schedule.every().tuesday.at("09:30").do(self.daily_analysis_job)
        schedule.every().wednesday.at("09:30").do(self.daily_analysis_job)
        schedule.every().thursday.at("09:30").do(self.daily_analysis_job)
        schedule.every().friday.at("09:30").do(self.daily_analysis_job)
        
        print("Scheduler started with jobs:")
        print("  - Data collection: Every 1 minute")
        print("  - Signal generation: Every 5 minutes")
        print("  - Model retraining: Weekly (Sunday 8 PM)")
        print("  - Daily analysis: Market open (9:30 AM EST)")
        
        # Run scheduler in background
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        return scheduler_thread
    
    def run(self):
        """Run the complete integrated system."""
        print("=" * 60)
        print("ML TRADING SYSTEM - INTEGRATED")
        print("=" * 60)
        
        # Initialize
        self.initialize_system()
        
        # Start data collection
        data_thread = threading.Thread(target=self.data_service.run_during_market_hours, daemon=True)
        data_thread.start()
        
        # Start scheduler
        self.start_scheduler()
        
        print("\nSystem running. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.data_service.running = False

if __name__ == "__main__":
    integration = MLTradingIntegration()
    integration.run()


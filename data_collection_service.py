#!/usr/bin/env python3
"""
Continuous Data Collection Service
Records market data in real-time for ML training and analysis
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
import asyncio
from typing import List, Dict, Optional
from advanced_ml_trading_system import AdvancedTradingSystem, MarketSnapshot
import pytz

class DataCollectionService:
    """Service for continuously collecting and recording market data."""
    
    def __init__(self, symbols: List[str] = None, interval_seconds: int = 60):
        self.symbols = symbols or ["ES=F", "NQ=F", "GC=F"]
        self.interval_seconds = interval_seconds
        self.system = AdvancedTradingSystem()
        self.running = False
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for a snapshot."""
        if df.empty or len(df) < 20:
            return {}
        
        latest = df.iloc[-1]
        
        indicators = {}
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        indicators['rsi'] = float((100 - (100 / (1 + rs))).iloc[-1]) if not rs.isna().iloc[-1] else None
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        indicators['macd'] = float(macd.iloc[-1]) if not macd.isna().iloc[-1] else None
        indicators['macd_signal'] = float(signal.iloc[-1]) if not signal.isna().iloc[-1] else None
        indicators['macd_hist'] = float((macd - signal).iloc[-1]) if not (macd - signal).isna().iloc[-1] else None
        
        # Moving Averages
        indicators['ma_20'] = float(df['Close'].rolling(20).mean().iloc[-1]) if len(df) >= 20 else None
        indicators['ma_50'] = float(df['Close'].rolling(50).mean().iloc[-1]) if len(df) >= 50 else None
        
        # Bollinger Bands
        if len(df) >= 20:
            ma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            indicators['bb_upper'] = float((ma_20 + (std_20 * 2)).iloc[-1])
            indicators['bb_lower'] = float((ma_20 - (std_20 * 2)).iloc[-1])
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = float(tr.rolling(14).mean().iloc[-1]) if len(df) >= 14 else None
        
        # VWAP (for current day)
        today = df[df.index.date == df.index[-1].date()]
        if not today.empty:
            typical_price = (today['High'] + today['Low'] + today['Close']) / 3
            vwap = (typical_price * today['Volume']).sum() / today['Volume'].sum()
            indicators['vwap'] = float(vwap)
        
        # Price change
        if len(df) >= 2:
            indicators['price_change_pct'] = float(((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100)
        
        # Volume change
        if len(df) >= 2:
            indicators['volume_change_pct'] = float(((df['Volume'].iloc[-1] / df['Volume'].iloc[-2]) - 1) * 100) if df['Volume'].iloc[-2] > 0 else 0
        
        return indicators
    
    def collect_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        """Collect a single market snapshot."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get recent data (last 100 minutes for indicators)
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                return None
            
            latest = data.iloc[-1]
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Calculate indicators
            indicators = self.calculate_indicators(data)
            
            snapshot = MarketSnapshot(
                timestamp=timestamp,
                symbol=symbol,
                open=float(latest['Open']),
                high=float(latest['High']),
                low=float(latest['Low']),
                close=float(latest['Close']),
                volume=int(latest['Volume']),
                rsi=indicators.get('rsi'),
                macd=indicators.get('macd'),
                macd_signal=indicators.get('macd_signal'),
                macd_hist=indicators.get('macd_hist'),
                ma_20=indicators.get('ma_20'),
                ma_50=indicators.get('ma_50'),
                bb_upper=indicators.get('bb_upper'),
                bb_lower=indicators.get('bb_lower'),
                atr=indicators.get('atr'),
                vwap=indicators.get('vwap'),
                price_change_pct=indicators.get('price_change_pct'),
                volume_change_pct=indicators.get('volume_change_pct')
            )
            
            return snapshot
        
        except Exception as e:
            print(f"Error collecting snapshot for {symbol}: {e}")
            return None
    
    def collect_all_symbols(self):
        """Collect snapshots for all symbols."""
        for symbol in self.symbols:
            snapshot = self.collect_snapshot(symbol)
            if snapshot:
                self.system.record_data(snapshot)
                print(f"[{datetime.now()}] Recorded {symbol}: ${snapshot.close:.2f}")
            time.sleep(0.5)  # Small delay between symbols
    
    def run_continuous(self):
        """Run continuous data collection."""
        self.running = True
        print(f"Starting continuous data collection for {self.symbols}")
        print(f"Collection interval: {self.interval_seconds} seconds")
        
        while self.running:
            try:
                self.collect_all_symbols()
                time.sleep(self.interval_seconds)
            except KeyboardInterrupt:
                print("\nStopping data collection...")
                self.running = False
                break
            except Exception as e:
                print(f"Error in data collection: {e}")
                time.sleep(self.interval_seconds)
    
    def run_during_market_hours(self):
        """Run data collection only during market hours."""
        est = pytz.timezone('America/New_York')
        
        while True:
            now_est = datetime.now(est)
            
            # Check if market is open (9:30 AM - 4:00 PM EST, Mon-Fri)
            if now_est.weekday() < 5:  # Monday-Friday
                market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
                
                if market_open <= now_est <= market_close:
                    if not self.running:
                        print(f"Market open. Starting data collection at {now_est.strftime('%H:%M:%S EST')}")
                        self.running = True
                    
                    self.collect_all_symbols()
                    time.sleep(self.interval_seconds)
                else:
                    if self.running:
                        print(f"Market closed. Stopping data collection at {now_est.strftime('%H:%M:%S EST')}")
                        self.running = False
                    time.sleep(60)  # Check every minute when market is closed
            else:
                if self.running:
                    print("Weekend. Stopping data collection.")
                    self.running = False
                time.sleep(3600)  # Check every hour on weekends

if __name__ == "__main__":
    service = DataCollectionService(
        symbols=["ES=F", "NQ=F", "GC=F"],
        interval_seconds=60  # Collect every minute
    )
    
    # Run during market hours
    service.run_during_market_hours()


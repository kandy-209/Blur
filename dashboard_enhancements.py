"""
Dashboard Enhancement Modules
=============================
Additional features for the trading dashboard:
- Data export functionality
- Alert system
- Portfolio tracking
- Multi-asset correlation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os


class DataExporter:
    """Handle data export to CSV/JSON formats."""
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: str = None) -> str:
        """Export DataFrame to CSV format."""
        if filename is None:
            filename = f"trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Ensure filename has .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        # Create exports directory if it doesn't exist
        os.makedirs('data/exports', exist_ok=True)
        filepath = os.path.join('data/exports', filename)
        
        df.to_csv(filepath, index=True)
        return filepath
    
    @staticmethod
    def export_to_json(df: pd.DataFrame, filename: str = None) -> str:
        """Export DataFrame to JSON format."""
        if filename is None:
            filename = f"trading_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        os.makedirs('data/exports', exist_ok=True)
        filepath = os.path.join('data/exports', filename)
        
        # Convert to dict with datetime as string
        df_dict = df.reset_index().to_dict(orient='records')
        for record in df_dict:
            if 'Datetime' in record and pd.notna(record['Datetime']):
                if isinstance(record['Datetime'], pd.Timestamp):
                    record['Datetime'] = record['Datetime'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(df_dict, f, indent=2, default=str)
        
        return filepath
    
    @staticmethod
    def export_metrics(metrics: Dict, filename: str = None) -> str:
        """Export metrics dictionary to JSON."""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        os.makedirs('data/exports', exist_ok=True)
        filepath = os.path.join('data/exports', filename)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        return filepath


class AlertSystem:
    """Real-time alert system for trading signals."""
    
    def __init__(self):
        self.alerts = []
        self.alert_history = []
        self.max_history = 100
    
    def check_alert(self, condition: str, value: float, threshold: float, 
                   symbol: str, alert_type: str = "info") -> Optional[Dict]:
        """Check if an alert condition is met."""
        triggered = False
        
        if ">" in condition:
            triggered = value > threshold
        elif "<" in condition:
            triggered = value < threshold
        elif ">=" in condition:
            triggered = value >= threshold
        elif "<=" in condition:
            triggered = value <= threshold
        elif "==" in condition:
            triggered = abs(value - threshold) < 0.01
        
        if triggered:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "type": alert_type,
                "condition": condition,
                "value": value,
                "threshold": threshold,
                "message": f"{symbol}: {condition.replace('>', 'above').replace('<', 'below')} {threshold}"
            }
            self.alerts.append(alert)
            self.alert_history.append(alert)
            
            # Keep history limited
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
            
            return alert
        return None
    
    def get_active_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent active alerts."""
        return self.alerts[-limit:]
    
    def clear_alerts(self):
        """Clear active alerts."""
        self.alerts = []
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get alert history."""
        return self.alert_history[-limit:]


class PortfolioTracker:
    """Track portfolio positions and performance."""
    
    def __init__(self):
        self.positions = {}  # {symbol: {quantity, entry_price, entry_time}}
        self.trades = []
    
    def add_position(self, symbol: str, quantity: float, entry_price: float):
        """Add a new position."""
        if symbol in self.positions:
            # Average the entry price
            old_qty = self.positions[symbol]['quantity']
            old_price = self.positions[symbol]['entry_price']
            total_cost = (old_qty * old_price) + (quantity * entry_price)
            total_qty = old_qty + quantity
            self.positions[symbol] = {
                'quantity': total_qty,
                'entry_price': total_cost / total_qty,
                'entry_time': self.positions[symbol]['entry_time']
            }
        else:
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': datetime.now().isoformat()
            }
        
        self.trades.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': entry_price
        })
    
    def close_position(self, symbol: str, quantity: float, exit_price: float):
        """Close part or all of a position."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        close_qty = min(quantity, pos['quantity'])
        
        pnl = (exit_price - pos['entry_price']) * close_qty
        pnl_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
        
        pos['quantity'] -= close_qty
        if pos['quantity'] <= 0:
            del self.positions[symbol]
        
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': 'SELL',
            'quantity': close_qty,
            'price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }
        self.trades.append(trade)
        return trade
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate current portfolio value and P&L."""
        total_cost = 0
        total_value = 0
        
        for symbol, pos in self.positions.items():
            current_price = current_prices.get(symbol, pos['entry_price'])
            cost = pos['quantity'] * pos['entry_price']
            value = pos['quantity'] * current_price
            
            total_cost += cost
            total_value += value
        
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'total_cost': total_cost,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'positions': len(self.positions)
        }
    
    def get_positions(self) -> Dict:
        """Get all current positions."""
        return self.positions.copy()
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get recent trade history."""
        return self.trades[-limit:]


class CorrelationAnalyzer:
    """Analyze correlations between multiple assets."""
    
    @staticmethod
    def calculate_correlation_matrix(price_data: Dict[str, pd.Series], 
                                    method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets."""
        # Align all series to common index
        df = pd.DataFrame(price_data)
        df = df.dropna()
        
        if len(df) < 10:
            return pd.DataFrame()
        
        # Calculate returns
        returns = df.pct_change().dropna()
        
        if len(returns) < 5:
            return pd.DataFrame()
        
        # Calculate correlation
        corr_matrix = returns.corr(method=method)
        return corr_matrix
    
    @staticmethod
    def find_high_correlation_pairs(corr_matrix: pd.DataFrame, 
                                   threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find asset pairs with high correlation."""
        pairs = []
        
        for i, symbol1 in enumerate(corr_matrix.columns):
            for j, symbol2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr = corr_matrix.loc[symbol1, symbol2]
                    if abs(corr) >= threshold:
                        pairs.append((symbol1, symbol2, corr))
        
        # Sort by absolute correlation
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs


# Global instances
_alert_system = AlertSystem()
_portfolio_tracker = PortfolioTracker()

def get_alert_system() -> AlertSystem:
    """Get global alert system instance."""
    return _alert_system

def get_portfolio_tracker() -> PortfolioTracker:
    """Get global portfolio tracker instance."""
    return _portfolio_tracker


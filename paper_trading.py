"""
Paper Trading System
====================
A complete paper trading module for practice trading without real money.
Features:
- Buy/Sell order placement
- Position tracking
- Order history
- Real-time P&L calculation
- Portfolio management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import os


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: str
    order_type: str = "MARKET"  # MARKET, LIMIT, STOP
    status: str = "FILLED"  # PENDING, FILLED, CANCELLED
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: str
    last_update: str


class PaperTradingSystem:
    """Complete paper trading system."""
    
    def __init__(self, initial_capital: float = 100000.0, data_dir: str = "data/paper_trading"):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.order_history: List[Order] = []
        self.trade_history: List[Dict] = []
        self.data_dir = data_dir
        self.order_counter = 0
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data if available
        self._load_data()
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"ORD{datetime.now().strftime('%Y%m%d')}{self.order_counter:04d}"
    
    def _save_data(self):
        """Save trading data to disk."""
        try:
            # Save orders
            orders_data = [asdict(order) for order in self.order_history[-1000:]]  # Keep last 1000
            with open(os.path.join(self.data_dir, "orders.json"), "w") as f:
                json.dump(orders_data, f, indent=2, default=str)
            
            # Save positions
            positions_data = {k: {
                "symbol": v.symbol,
                "quantity": v.quantity,
                "avg_entry_price": v.avg_entry_price,
                "entry_time": v.entry_time,
            } for k, v in self.positions.items()}
            with open(os.path.join(self.data_dir, "positions.json"), "w") as f:
                json.dump(positions_data, f, indent=2, default=str)
            
            # Save account state
            account_data = {
                "cash": self.cash,
                "initial_capital": self.initial_capital,
                "order_counter": self.order_counter,
            }
            with open(os.path.join(self.data_dir, "account.json"), "w") as f:
                json.dump(account_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving paper trading data: {e}")
    
    def _load_data(self):
        """Load trading data from disk."""
        try:
            # Load account
            account_file = os.path.join(self.data_dir, "account.json")
            if os.path.exists(account_file):
                with open(account_file, "r") as f:
                    account_data = json.load(f)
                    self.cash = account_data.get("cash", self.initial_capital)
                    self.order_counter = account_data.get("order_counter", 0)
            
            # Load positions
            positions_file = os.path.join(self.data_dir, "positions.json")
            if os.path.exists(positions_file):
                with open(positions_file, "r") as f:
                    positions_data = json.load(f)
                    for symbol, pos_data in positions_data.items():
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            quantity=pos_data["quantity"],
                            avg_entry_price=pos_data["avg_entry_price"],
                            current_price=pos_data["avg_entry_price"],  # Will be updated
                            unrealized_pnl=0.0,
                            unrealized_pnl_pct=0.0,
                            entry_time=pos_data["entry_time"],
                            last_update=datetime.now().isoformat(),
                        )
            
            # Load order history
            orders_file = os.path.join(self.data_dir, "orders.json")
            if os.path.exists(orders_file):
                with open(orders_file, "r") as f:
                    orders_data = json.load(f)
                    for order_data in orders_data:
                        order = Order(**order_data)
                        self.order_history.append(order)
        except Exception as e:
            print(f"Error loading paper trading data: {e}")
    
    def place_order(self, symbol: str, side: str, quantity: float, 
                   price: Optional[float] = None, order_type: str = "MARKET") -> Order:
        """Place a buy or sell order."""
        if side not in ["BUY", "SELL"]:
            raise ValueError("Side must be 'BUY' or 'SELL'")
        
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        # Use current market price if not specified
        if price is None:
            price = 0.0  # Will be filled at market price
        
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=datetime.now().isoformat(),
            order_type=order_type,
            status="FILLED",  # For paper trading, market orders fill immediately
            filled_price=price,
            filled_quantity=quantity,
        )
        
        # Execute the order
        self._execute_order(order)
        
        # Save data
        self._save_data()
        
        return order
    
    def _execute_order(self, order: Order):
        """Execute an order and update positions."""
        cost = order.filled_price * order.filled_quantity
        
        if order.side == "BUY":
            # Check if we have enough cash
            if self.cash < cost:
                order.status = "REJECTED"
                raise ValueError(f"Insufficient cash. Available: ${self.cash:.2f}, Required: ${cost:.2f}")
            
            # Deduct cash
            self.cash -= cost
            
            # Update or create position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                # Average the entry price
                total_cost = (pos.quantity * pos.avg_entry_price) + cost
                total_quantity = pos.quantity + order.filled_quantity
                pos.avg_entry_price = total_cost / total_quantity
                pos.quantity = total_quantity
                pos.last_update = datetime.now().isoformat()
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.filled_quantity,
                    avg_entry_price=order.filled_price,
                    current_price=order.filled_price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    entry_time=datetime.now().isoformat(),
                    last_update=datetime.now().isoformat(),
                )
        
        elif order.side == "SELL":
            # Check if we have the position
            if order.symbol not in self.positions:
                order.status = "REJECTED"
                raise ValueError(f"No position in {order.symbol} to sell")
            
            pos = self.positions[order.symbol]
            
            # Check if we have enough quantity
            if pos.quantity < order.filled_quantity:
                order.status = "REJECTED"
                raise ValueError(f"Insufficient quantity. Available: {pos.quantity}, Requested: {order.filled_quantity}")
            
            # Calculate P&L
            pnl = (order.filled_price - pos.avg_entry_price) * order.filled_quantity
            pnl_pct = ((order.filled_price - pos.avg_entry_price) / pos.avg_entry_price) * 100
            
            # Add cash
            self.cash += order.filled_price * order.filled_quantity
            
            # Update position
            pos.quantity -= order.filled_quantity
            if pos.quantity <= 0:
                del self.positions[order.symbol]
            else:
                pos.last_update = datetime.now().isoformat()
            
            # Record trade
            self.trade_history.append({
                "timestamp": order.timestamp,
                "symbol": order.symbol,
                "side": "SELL",
                "quantity": order.filled_quantity,
                "entry_price": pos.avg_entry_price,
                "exit_price": order.filled_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            })
        
        # Add to orders and history
        self.orders.append(order)
        self.order_history.append(order)
    
    def update_positions(self, current_prices: Dict[str, float]):
        """Update positions with current market prices."""
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                pos.current_price = current_prices[symbol]
                pos.unrealized_pnl = (pos.current_price - pos.avg_entry_price) * pos.quantity
                pos.unrealized_pnl_pct = ((pos.current_price - pos.avg_entry_price) / pos.avg_entry_price) * 100
                pos.last_update = datetime.now().isoformat()
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate total portfolio value."""
        self.update_positions(current_prices)
        
        positions_value = sum(
            pos.current_price * pos.quantity 
            for pos in self.positions.values()
        )
        
        total_value = self.cash + positions_value
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        return {
            "cash": self.cash,
            "positions_value": positions_value,
            "total_value": total_value,
            "initial_capital": self.initial_capital,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "num_positions": len(self.positions),
        }
    
    def get_positions(self) -> List[Dict]:
        """Get all current positions as dictionaries."""
        return [
            {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "avg_entry_price": pos.avg_entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "entry_time": pos.entry_time,
            }
            for pos in self.positions.values()
        ]
    
    def get_recent_orders(self, limit: int = 20) -> List[Dict]:
        """Get recent orders."""
        return [
            asdict(order) for order in self.order_history[-limit:]
        ]
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get trade history (closed positions)."""
        return self.trade_history[-limit:]
    
    def close_position(self, symbol: str, quantity: Optional[float] = None):
        """Close a position (or part of it)."""
        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol}")
        
        pos = self.positions[symbol]
        close_qty = quantity if quantity else pos.quantity
        
        if close_qty > pos.quantity:
            close_qty = pos.quantity
        
        # Place a sell order
        order = self.place_order(
            symbol=symbol,
            side="SELL",
            quantity=close_qty,
            price=pos.current_price,
            order_type="MARKET"
        )
        
        return order
    
    def reset_account(self):
        """Reset account to initial state (for testing)."""
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = []
        self.order_history = []
        self.trade_history = []
        self.order_counter = 0
        
        # Clear saved data
        try:
            for filename in ["orders.json", "positions.json", "account.json"]:
                filepath = os.path.join(self.data_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
        except Exception as e:
            print(f"Error resetting account: {e}")


# Global instance
_paper_trading = None

def get_paper_trading_system(initial_capital: float = 100000.0) -> PaperTradingSystem:
    """Get global paper trading system instance."""
    global _paper_trading
    if _paper_trading is None:
        _paper_trading = PaperTradingSystem(initial_capital=initial_capital)
    return _paper_trading


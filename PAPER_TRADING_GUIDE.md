# Paper Trading System Guide

## 🎯 Overview

The Paper Trading section allows users to practice trading futures without risking real money. You start with $100,000 in virtual cash and can place buy/sell orders to build and manage a portfolio.

## 💰 Features

### 1. **Order Placement**
- **Buy Orders**: Purchase contracts at current market price
- **Sell Orders**: Sell existing positions or short sell
- **Market Orders**: Execute immediately at current price
- **Price Entry**: Optional price field (uses market price if not specified)

### 2. **Portfolio Tracking**
- **Real-time P&L**: See unrealized profit/loss on open positions
- **Total Portfolio Value**: Cash + positions value
- **Position Details**: Entry price, current price, quantity, P&L percentage

### 3. **Order History**
- **Recent Orders**: View last 10 executed orders
- **Order Details**: Symbol, side, quantity, price, timestamp
- **Trade History**: Closed positions with realized P&L

### 4. **Account Management**
- **Starting Capital**: $100,000 virtual cash
- **Cash Balance**: Available funds for new orders
- **Position Tracking**: All open positions automatically tracked

## 📊 How to Use

### Placing a Buy Order

1. **Select Symbol**: Choose the futures contract from the dropdown (already selected from main chart)
2. **Choose Side**: Select "BUY"
3. **Enter Quantity**: Type the number of contracts (e.g., 1, 2, 0.5)
4. **Price (Optional)**: Leave blank to use current market price, or enter a specific price
5. **Click "Place Order"**: Order executes immediately

### Placing a Sell Order

1. **Select Symbol**: Choose the contract you want to sell
2. **Choose Side**: Select "SELL"
3. **Enter Quantity**: Amount to sell (must not exceed position size)
4. **Click "Place Order"**: Order executes and closes position (or part of it)

### Viewing Your Portfolio

- **Portfolio Summary**: Shows at the top with:
  - Available Cash
  - Total Portfolio Value
  - Total P&L (profit/loss)
  - Number of Open Positions

- **Open Positions**: Lists all current positions with:
  - Symbol
  - Quantity
  - Entry Price
  - Current Price
  - Unrealized P&L (in $ and %)

- **Recent Orders**: Shows last 5 executed orders

## 💡 Trading Tips

1. **Start Small**: Begin with small position sizes to learn
2. **Watch P&L**: Monitor your unrealized P&L to see how positions perform
3. **Use Current Prices**: The system automatically uses the latest price from the chart
4. **Manage Risk**: Don't use all your cash on one position
5. **Practice Strategies**: Test different trading strategies risk-free

## 🔧 Technical Details

### Data Persistence
- Orders and positions are saved to `data/paper_trading/`
- Data persists between dashboard restarts
- Files: `orders.json`, `positions.json`, `account.json`

### Price Updates
- Positions update automatically every 30 seconds
- Uses current market price from the selected symbol
- P&L recalculates in real-time

### Order Execution
- Market orders execute immediately
- No slippage simulation (executes at exact price)
- Insufficient cash/quantity orders are rejected

## 📈 Example Trading Session

1. **Start**: $100,000 cash, 0 positions
2. **Buy ES=F**: 1 contract @ $4,500
   - Cash: $95,500
   - Position: 1 ES=F @ $4,500
3. **Price Moves**: ES=F now at $4,510
   - Unrealized P&L: +$10 (0.22%)
   - Total Value: $100,010
4. **Sell ES=F**: 1 contract @ $4,510
   - Cash: $100,010
   - Realized P&L: +$10
   - Position: Closed

## ⚠️ Important Notes

- **Paper Trading Only**: This is for practice only, no real money involved
- **Market Hours**: Prices update based on market data availability
- **No Commissions**: Orders don't include trading fees (for simplicity)
- **Data Reset**: To reset account, delete files in `data/paper_trading/` directory

## 🚀 Advanced Features

### Position Management
- Partial closes: Sell part of a position
- Multiple positions: Hold different symbols simultaneously
- Average entry: Multiple buys average the entry price

### Portfolio Analytics
- Total return calculation
- Win/loss tracking (via trade history)
- Position sizing recommendations

## 🔮 Future Enhancements

Potential additions:
- Limit orders
- Stop-loss orders
- Order cancellation
- Trade journal/notes
- Performance analytics dashboard
- Strategy backtesting integration
- Multi-account support
- Commission simulation

## 📝 Code Example

```python
from paper_trading import get_paper_trading_system

# Get system instance
pt = get_paper_trading_system()

# Place a buy order
order = pt.place_order(
    symbol="ES=F",
    side="BUY",
    quantity=1,
    price=4500.0
)

# Get portfolio value
portfolio = pt.get_portfolio_value({"ES=F": 4510.0})
print(f"Total Value: ${portfolio['total_value']:,.2f}")

# Get positions
positions = pt.get_positions()
for pos in positions:
    print(f"{pos['symbol']}: {pos['quantity']} @ ${pos['avg_entry_price']:.2f}")
```

## 🎓 Learning Resources

Use paper trading to:
- Practice entry/exit timing
- Test technical analysis strategies
- Learn position sizing
- Understand P&L calculation
- Build trading confidence

Happy Trading! 📈


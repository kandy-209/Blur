import modal
import pandas as pd
import os
from datetime import datetime

# Build image with required packages
image = (
    modal.Image.debian_slim()
    .pip_install("yfinance", "pandas", "numpy")
)

# Create Modal app
app = modal.App("market-analysis")

# Persistent volume name (will be created if missing)
VOLUME_NAME = "futures_analysis"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
MOUNT_PATH = "/data"  # inside the remote container

# Symbols for futures
SYMBOLS = ["NQ=F", "ES=F", "GC=F"]  # Nasdaq, S&P, Gold

# Helper to compute RSI (returns series)
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Remote function: fetch + analyze + append to CSV on volume
@app.function(image=image, volumes={MOUNT_PATH: volume}, timeout=600)
def get_market_data(symbol: str) -> dict:
    import yfinance as yf
    import pandas as pd
    from datetime import datetime

    # Fetch
    df = yf.download(symbol, period="5d", interval="1h", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        # flatten multiindex if yfinance returns multi-column
        df.columns = [c[0] for c in df.columns]

    # Indicators
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = compute_rsi(df["Close"], window=14)

    # drop NaNs so last row is valid
    df = df.dropna()
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    last = df.iloc[-1]
    last_price = float(last["Close"])
    last_rsi = float(last["RSI"])
    ma20 = float(last["MA_20"])
    ma50 = float(last["MA_50"])

    # Simple trade signal logic (RSI + MA crossover)
    buy_signals = 0
    sell_signals = 0
    if last_rsi < 30:
        buy_signals += 1
    elif last_rsi > 70:
        sell_signals += 1

    if ma20 > ma50:
        buy_signals += 1
    elif ma20 < ma50:
        sell_signals += 1

    trade_signal = "HOLD"
    if buy_signals > sell_signals:
        trade_signal = "BUY"
    elif sell_signals > buy_signals:
        trade_signal = "SELL"

    # Prepare row to append
    row = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "close": last_price,
        "rsi_14": last_rsi,
        "ma_20": ma20,
        "ma_50": ma50,
        "trade_signal": trade_signal,
    }

    # Persist CSV to the mounted volume path
    csv_path = os.path.join(MOUNT_PATH, "market_summary.csv")
    row_df = pd.DataFrame([row])

    # If file exists, append, otherwise write header
    if os.path.exists(csv_path):
        row_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        # Ensure directory exists (should, but be safe)
        os.makedirs(MOUNT_PATH, exist_ok=True)
        row_df.to_csv(csv_path, mode="w", header=True, index=False)

    print(f"✅ {symbol} - price={last_price:.2f}, rsi={last_rsi:.2f}, signal={trade_signal}  (saved to {csv_path})")

    # Return the summary dict (so local process can also consume it)
    return row

# Local entrypoint orchestrates the three-symbol run and prints colored summary
@app.local_entrypoint()
def main():
    COLOR_RESET = "\033[0m"
    COLOR_GREEN = "\033[32m"
    COLOR_RED = "\033[31m"
    COLOR_YELLOW = "\033[33m"
    COLOR_CYAN = "\033[36m"

    def color_signal(signal: str):
        if signal == "BUY":
            return f"{COLOR_GREEN}{signal}{COLOR_RESET}"
        if signal == "SELL":
            return f"{COLOR_RED}{signal}{COLOR_RESET}"
        return f"{COLOR_YELLOW}{signal}{COLOR_RESET}"

    print(f"{COLOR_CYAN}Fetching market data for {', '.join(SYMBOLS)} and appending to Modal volume '{VOLUME_NAME}'...{COLOR_RESET}\n")

    results = []
    for sym in SYMBOLS:
        print(f"→ Fetching {sym} ...")
        summary = get_market_data.remote(sym)
        results.append(summary)

    print(f"\n{COLOR_CYAN}===== SUMMARY ====={COLOR_RESET}\n")
    for r in results:
        # r is a dict containing the row
        sig_colored = color_signal(r["trade_signal"])
        print(f"{r['symbol']}: price={r['close']:.2f}, rsi={r['rsi_14']:.2f}, ma20={r['ma_20']:.2f}, ma50={r['ma_50']:.2f} → {sig_colored}")

    print(f"\nCSV file persisted to Modal Volume '{VOLUME_NAME}' as /data/market_summary.csv")
    print("Run complete.")

# ---------- IMPORTS ----------
import pandas as pd
import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Modal and ASGI imports (only needed for deployment)
try:
    import modal
    from asgiref.wsgi import WsgiToAsgi
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    # Create a dummy modal for local testing
    class DummyModal:
        class App:
            def __init__(self, name):
                self.name = name
            def asgi_app(self, image=None):
                def decorator(func):
                    return func
                return decorator
        class Image:
            @staticmethod
            def debian_slim():
                return DummyImage()
    class DummyImage:
        def pip_install(self, packages):
            return self
    modal = DummyModal()

# ---------- MODAL SETUP ----------
if MODAL_AVAILABLE:
    app = modal.App("live_futures_dashboard")
    image = (
        modal.Image.debian_slim()
        .pip_install([
            "dash",
            "pandas",
            "plotly",
            "yfinance",
            "asgiref",  # For WSGI-to-ASGI conversion
        ])
    )
else:
    app = None
    image = None

# ---------- DATA FETCHING ----------
symbols = {"ES=F": "S&P 500", "NQ=F": "Nasdaq", "GC=F": "Gold", "YM=F": "Dow", "CL=F": "Crude Oil"}

# Period and interval mapping
PERIOD_INTERVALS = {
    "1m": ("1d", "1m"),
    "5m": ("5d", "5m"),
    "15m": ("5d", "15m"),
    "1h": ("1mo", "1h"),
    "1d": ("1y", "1d"),
}

def fetch_data(sym, period="1d", interval="1m"):
    """Fetch data from Yahoo Finance with specified period and interval."""
    try:
        data = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        
        # Handle MultiIndex columns (yfinance sometimes returns these)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        
        data = data.reset_index()
        
        # Find datetime column
        datetime_col = None
        for col in ["Datetime", "Date", "index"]:
            if col in data.columns:
                datetime_col = col
                break
        
        if datetime_col:
            data["Datetime"] = pd.to_datetime(data[datetime_col], utc=True)
        else:
            # If no datetime column found, create one from index
            data["Datetime"] = pd.to_datetime(data.index, utc=True)
        
        # Ensure we have all required columns (case-insensitive)
        required_cols_lower = {col.lower(): col for col in ["Open", "High", "Low", "Close", "Volume"]}
        data_cols_lower = {col.lower(): col for col in data.columns}
        
        result_data = {"Datetime": data["Datetime"]}
        
        for req_lower, req_orig in required_cols_lower.items():
            if req_lower in data_cols_lower:
                result_data[req_orig] = data[data_cols_lower[req_lower]]
            else:
                # Default values if column missing
                if req_orig == "Volume":
                    result_data[req_orig] = 0
                elif req_orig == "Close":
                    # Try to get Close from other columns
                    if "close" in data_cols_lower:
                        result_data[req_orig] = data[data_cols_lower["close"]]
                    else:
                        result_data[req_orig] = 0
                else:
                    # For Open, High, Low, use Close if available
                    if "close" in data_cols_lower:
                        result_data[req_orig] = data[data_cols_lower["close"]]
                    else:
                        result_data[req_orig] = 0
        
        return pd.DataFrame(result_data)
    except Exception as e:
        print(f"Error fetching {sym}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])

def calculate_indicators(df):
    """Calculate technical indicators."""
    if df.empty or len(df) < 14:
        return df
    
    # Moving Averages
    df["MA_20"] = df["Close"].rolling(window=min(20, len(df))).mean()
    df["MA_50"] = df["Close"].rolling(window=min(50, len(df))).mean()
    
    # RSI Calculation
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=min(14, len(df))).mean()
    avg_loss = loss.rolling(window=min(14, len(df))).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=min(20, len(df))).mean()
    bb_std = df["Close"].rolling(window=min(20, len(df))).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    
    return df

def calculate_metrics(df, symbol):
    """Calculate key metrics for display."""
    if df.empty:
        return {
            "current_price": 0,
            "change": 0,
            "change_pct": 0,
            "high": 0,
            "low": 0,
            "volume": 0,
            "rsi": 0,
            "volatility": 0,
        }
    
    current_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[0]) if len(df) > 1 else current_price
    change = current_price - prev_price
    change_pct = (change / prev_price * 100) if prev_price != 0 else 0
    
    high = float(df["High"].max())
    low = float(df["Low"].min())
    volume = int(df["Volume"].sum())
    
    rsi = float(df["RSI"].iloc[-1]) if "RSI" in df.columns and not pd.isna(df["RSI"].iloc[-1]) else 0
    
    # Volatility (standard deviation of returns)
    returns = df["Close"].pct_change().dropna()
    volatility = float(returns.std() * 100) if len(returns) > 0 else 0
    
    return {
        "current_price": current_price,
        "change": change,
        "change_pct": change_pct,
        "high": high,
        "low": low,
        "volume": volume,
        "rsi": rsi,
        "volatility": volatility,
    }

# ---------- DASH APP ----------
def create_dash_app():
    dash_app = dash.Dash(__name__, requests_pathname_prefix="/")
    
    # Custom CSS styling - Dark theme with Fallout/TopStepX vibes
    dash_app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Futures Trading Terminal</title>
            {%favicon%}
            {%css%}
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&display=swap');
                
                * {
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Rajdhani', 'Courier New', monospace;
                    background: #0a0a0a;
                    background-image: 
                        radial-gradient(circle at 20% 50%, rgba(125, 216, 125, 0.08) 0%, transparent 50%),
                        radial-gradient(circle at 80% 80%, rgba(255, 20, 147, 0.06) 0%, transparent 50%),
                        radial-gradient(circle at 50% 20%, rgba(0, 191, 255, 0.06) 0%, transparent 50%),
                        linear-gradient(135deg, rgba(138, 43, 226, 0.03) 0%, transparent 50%);
                    margin: 0;
                    padding: 20px;
                    color: #7dd87d;
                    min-height: 100vh;
                    filter: blur(0.3px);
                }
                
                .metric-card {
                    background: linear-gradient(135deg, rgba(26, 26, 26, 0.95) 0%, rgba(15, 15, 15, 0.95) 100%);
                    border: 1px solid rgba(125, 216, 125, 0.3);
                    border-top: 2px solid rgba(125, 216, 125, 0.5);
                    border-radius: 6px;
                    padding: 20px;
                    margin: 10px;
                    box-shadow: 
                        0 0 20px rgba(125, 216, 125, 0.12),
                        0 0 40px rgba(255, 20, 147, 0.05),
                        inset 0 0 10px rgba(125, 216, 125, 0.03);
                    text-align: center;
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                    backdrop-filter: blur(3px);
                }
                
                .metric-card::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(90deg, 
                        transparent, 
                        rgba(125, 216, 125, 0.1), 
                        rgba(0, 191, 255, 0.08),
                        rgba(255, 20, 147, 0.08),
                        transparent);
                    transition: left 0.6s;
                    filter: blur(4px);
                }
                
                .metric-card:hover::before {
                    left: 100%;
                }
                
                .metric-card:hover {
                    border-color: rgba(125, 216, 125, 0.6);
                    border-top-color: rgba(0, 191, 255, 0.6);
                    box-shadow: 
                        0 0 30px rgba(125, 216, 125, 0.25),
                        0 0 60px rgba(255, 20, 147, 0.15),
                        inset 0 0 20px rgba(125, 216, 125, 0.08);
                    transform: translateY(-3px);
                }
                
                .metric-value {
                    font-size: 28px;
                    font-weight: 600;
                    margin: 8px 0;
                    font-family: 'Share Tech Mono', monospace;
                    text-shadow: 0 0 8px currentColor, 0 0 15px currentColor;
                    letter-spacing: 1px;
                    filter: blur(0.2px);
                }
                
                .metric-label {
                    font-size: 11px;
                    color: rgba(125, 216, 125, 0.7);
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    font-weight: 500;
                    opacity: 0.8;
                    filter: blur(0.3px);
                }
                
                .positive { 
                    color: #7dd87d; 
                    text-shadow: 0 0 10px #7dd87d, 0 0 20px rgba(125, 216, 125, 0.4), 0 0 30px rgba(0, 191, 255, 0.2);
                }
                
                .negative { 
                    color: #ff6b6b; 
                    text-shadow: 0 0 10px #ff6b6b, 0 0 20px rgba(255, 107, 107, 0.4), 0 0 30px rgba(255, 20, 147, 0.2);
                }
                
                .neutral { 
                    color: #00bfff; 
                    text-shadow: 0 0 10px #00bfff, 0 0 20px rgba(0, 191, 255, 0.4), 0 0 30px rgba(138, 43, 226, 0.2);
                }
                
                h1, h2, h3 {
                    font-family: 'Share Tech Mono', monospace;
                    text-shadow: 
                        0 0 15px currentColor, 
                        0 0 25px rgba(125, 216, 125, 0.5),
                        0 0 40px rgba(0, 191, 255, 0.3),
                        0 0 60px rgba(255, 20, 147, 0.2);
                    filter: blur(0.4px);
                }
                
                /* Dropdown styling */
                .Select-control {
                    background-color: rgba(26, 26, 26, 0.95) !important;
                    border: 1px solid rgba(125, 216, 125, 0.4) !important;
                    border-top: 2px solid rgba(0, 191, 255, 0.5) !important;
                    border-radius: 6px !important;
                    color: #7dd87d !important;
                    backdrop-filter: blur(3px);
                    box-shadow: 0 0 15px rgba(125, 216, 125, 0.1), 0 0 30px rgba(0, 191, 255, 0.05) !important;
                }
                
                .Select-menu-outer {
                    background-color: rgba(26, 26, 26, 0.98) !important;
                    border: 1px solid rgba(125, 216, 125, 0.4) !important;
                    border-top: 2px solid rgba(255, 20, 147, 0.5) !important;
                    backdrop-filter: blur(3px);
                    box-shadow: 0 0 20px rgba(255, 20, 147, 0.2) !important;
                }
                
                .Select-option {
                    background-color: transparent !important;
                    color: #7dd87d !important;
                    transition: all 0.2s ease;
                }
                
                .Select-option.is-focused {
                    background: linear-gradient(90deg, rgba(125, 216, 125, 0.15), rgba(0, 191, 255, 0.15)) !important;
                    color: #00bfff !important;
                    text-shadow: 0 0 10px #00bfff;
                }
                
                .Select-value-label {
                    color: #7dd87d !important;
                    filter: blur(0.2px);
                }
                
                .Select-placeholder {
                    color: rgba(125, 216, 125, 0.6) !important;
                    opacity: 0.6;
                }
                
                label {
                    color: rgba(125, 216, 125, 0.8) !important;
                    text-shadow: 0 0 8px rgba(125, 216, 125, 0.5), 0 0 15px rgba(0, 191, 255, 0.3);
                    font-weight: 600;
                    letter-spacing: 1.5px;
                    filter: blur(0.3px);
                }
                
                /* Scrollbar styling */
                ::-webkit-scrollbar {
                    width: 10px;
                    height: 10px;
                }
                
                ::-webkit-scrollbar-track {
                    background: #0a0a0a;
                }
                
                ::-webkit-scrollbar-thumb {
                    background: linear-gradient(180deg, rgba(125, 216, 125, 0.6), rgba(0, 191, 255, 0.6));
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(125, 216, 125, 0.3);
                }
                
                ::-webkit-scrollbar-thumb:hover {
                    background: linear-gradient(180deg, rgba(125, 216, 125, 0.8), rgba(255, 20, 147, 0.8));
                    box-shadow: 0 0 15px rgba(125, 216, 125, 0.5), 0 0 25px rgba(255, 20, 147, 0.3);
                }
                
                /* Live indicator pulse animation */
                @keyframes pulse {
                    0%, 100% {
                        opacity: 1;
                        text-shadow: 0 0 10px #00ff41, 0 0 20px #00ff41, 0 0 30px #00ff41;
                    }
                    50% {
                        opacity: 0.4;
                        text-shadow: 0 0 5px #00ff41, 0 0 10px #00ff41;
                    }
                }
                
                #live-dot {
                    animation: pulse 1.5s ease-in-out infinite;
                    display: inline-block;
                }
                
                /* Smooth chart transitions */
                .js-plotly-plot {
                    transition: opacity 0.4s ease-in-out;
                }
                
                /* Fade in animation for new data */
                @keyframes fadeIn {
                    from {
                        opacity: 0;
                        transform: translateY(-5px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                
                .metric-card {
                    animation: fadeIn 0.6s ease-out;
                }
                
                /* Subtle glow pulse on data updates */
                @keyframes dataUpdate {
                    0% {
                        box-shadow: 0 0 20px rgba(125, 216, 125, 0.12), 0 0 40px rgba(255, 20, 147, 0.05);
                    }
                    50% {
                        box-shadow: 0 0 30px rgba(125, 216, 125, 0.2), 0 0 60px rgba(0, 191, 255, 0.1);
                    }
                    100% {
                        box-shadow: 0 0 20px rgba(125, 216, 125, 0.12), 0 0 40px rgba(255, 20, 147, 0.05);
                    }
                }
                
                .metric-card:has(.metric-value) {
                    animation: fadeIn 0.6s ease-out, dataUpdate 2s ease-in-out infinite;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    dash_app.layout = html.Div(
        style={"maxWidth": "1600px", "margin": "0 auto", "backgroundColor": "transparent"},
        children=[
            html.Div(
                [
                    html.H1(
                        "FUTURES TRADING TERMINAL",
                        style={
                            "textAlign": "center",
                            "color": "#7dd87d",
                            "marginBottom": "10px",
                            "textShadow": 
                                "0 0 20px rgba(125, 216, 125, 0.8), "
                                "0 0 40px rgba(0, 191, 255, 0.5), "
                                "0 0 60px rgba(255, 20, 147, 0.3), "
                                "0 0 80px rgba(138, 43, 226, 0.2)",
                            "fontFamily": "'Share Tech Mono', monospace",
                            "fontSize": "38px",
                            "letterSpacing": "4px",
                            "fontWeight": "bold",
                            "borderBottom": "3px solid",
                            "borderImage": "linear-gradient(90deg, rgba(125, 216, 125, 0.6), rgba(0, 191, 255, 0.6), rgba(255, 20, 147, 0.6), rgba(138, 43, 226, 0.6)) 1",
                            "paddingBottom": "15px",
                            "marginBottom": "30px",
                            "boxShadow": 
                                "0 4px 30px rgba(125, 216, 125, 0.3), "
                                "0 4px 50px rgba(0, 191, 255, 0.2), "
                                "0 4px 70px rgba(255, 20, 147, 0.1)",
                            "filter": "blur(0.5px)",
                        },
                    ),
                    html.Div(
                        [
                            html.Span("SYSTEM ONLINE | DATA STREAM ACTIVE | MARKET LIVE", id="live-indicator"),
                            html.Span(" ●", id="live-dot", style={"color": "#00ff41", "marginLeft": "10px", "fontSize": "16px"}),
                        ],
                        style={
                            "textAlign": "center",
                            "color": "rgba(0, 191, 255, 0.8)",
                            "fontSize": "13px",
                            "letterSpacing": "5px",
                            "fontFamily": "'Share Tech Mono', monospace",
                            "marginBottom": "30px",
                            "opacity": "0.9",
                            "filter": "blur(0.4px)",
                            "textShadow": "0 0 10px rgba(0, 191, 255, 0.5), 0 0 20px rgba(255, 20, 147, 0.3)",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("SYMBOL:", style={"color": "rgba(125, 216, 125, 0.7)", "fontWeight": "600", "marginRight": "10px", "letterSpacing": "1px", "fontSize": "13px"}),
                                    dcc.Dropdown(
                                        id="symbol-dropdown",
                                        options=[{"label": f"{v} ({k})", "value": k} for k, v in symbols.items()],
                                        value="ES=F",
                                        clearable=False,
                                        style={"width": "250px", "backgroundColor": "#1a1a1a"},
                                    ),
                                ],
                                style={"display": "inline-block", "marginRight": "20px"},
                            ),
                            html.Div(
                                [
                                    html.Label("INTERVAL:", style={"color": "rgba(125, 216, 125, 0.7)", "fontWeight": "600", "marginRight": "10px", "letterSpacing": "1px", "fontSize": "13px"}),
                                    dcc.Dropdown(
                                        id="interval-dropdown",
                                        options=[
                                            {"label": "1 MIN", "value": "1m"},
                                            {"label": "5 MIN", "value": "5m"},
                                            {"label": "15 MIN", "value": "15m"},
                                            {"label": "1 HOUR", "value": "1h"},
                                            {"label": "1 DAY", "value": "1d"},
                                        ],
                                        value="1m",
                                        clearable=False,
                                        style={"width": "150px", "backgroundColor": "#1a1a1a"},
                                    ),
                                ],
                                style={"display": "inline-block", "marginRight": "20px"},
                            ),
                            html.Div(
                                [
                                    html.Label("CHART:", style={"color": "rgba(125, 216, 125, 0.7)", "fontWeight": "600", "marginRight": "10px", "letterSpacing": "1px", "fontSize": "13px"}),
                                    dcc.Dropdown(
                                        id="chart-type-dropdown",
                                        options=[
                                            {"label": "CANDLESTICK", "value": "candlestick"},
                                            {"label": "LINE", "value": "line"},
                                            {"label": "OHLC", "value": "ohlc"},
                                        ],
                                        value="candlestick",
                                        clearable=False,
                                        style={"width": "150px", "backgroundColor": "#1a1a1a"},
                                    ),
                                ],
                                style={"display": "inline-block"},
                            ),
                        ],
                        style={"textAlign": "center", "marginBottom": "20px"},
                    ),
                ]
            ),
            
            # Metrics Cards
            html.Div(
                id="metrics-container",
                style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center", "marginBottom": "20px"},
            ),
            
            # Main Chart
            html.Div(
                [
                    dcc.Graph(id="price-graph", style={"height": "500px", "backgroundColor": "#0a0a0a", "border": "1px solid rgba(125, 216, 125, 0.4)", "borderRadius": "4px", "padding": "15px", "boxShadow": "0 0 20px rgba(125, 216, 125, 0.15)", "backdropFilter": "blur(2px)"}),
                ],
                style={"marginBottom": "20px"},
            ),
            
            # RSI Chart
            html.Div(
                [
                    dcc.Graph(id="rsi-graph", style={"height": "300px", "backgroundColor": "rgba(10, 10, 10, 0.95)", "border": "1px solid rgba(125, 216, 125, 0.4)", "borderTop": "3px solid rgba(255, 20, 147, 0.6)", "borderRadius": "6px", "padding": "15px", "boxShadow": "0 0 25px rgba(255, 20, 147, 0.2), 0 0 50px rgba(138, 43, 226, 0.1)", "backdropFilter": "blur(3px)"}),
                ],
                style={"marginBottom": "20px"},
            ),
            
            # Volume Chart
            html.Div(
                [
                    dcc.Graph(id="volume-graph", style={"height": "250px", "backgroundColor": "rgba(10, 10, 10, 0.95)", "border": "1px solid rgba(125, 216, 125, 0.4)", "borderTop": "3px solid rgba(138, 43, 226, 0.6)", "borderRadius": "6px", "padding": "15px", "boxShadow": "0 0 25px rgba(138, 43, 226, 0.2), 0 0 50px rgba(0, 191, 255, 0.1)", "backdropFilter": "blur(3px)"}),
                ],
            ),
            
            dcc.Interval(id="interval", interval=30 * 1000, n_intervals=0),  # Update every 30 seconds for more live feel
        ]
    )

    @dash_app.callback(
        [
            Output("price-graph", "figure"),
            Output("rsi-graph", "figure"),
            Output("volume-graph", "figure"),
            Output("metrics-container", "children"),
        ],
        [
            Input("symbol-dropdown", "value"),
            Input("interval-dropdown", "value"),
            Input("chart-type-dropdown", "value"),
            Input("interval", "n_intervals"),
        ],
        prevent_initial_call=False,  # Allow initial call to load data immediately
    )
    def update_all(symbol, interval_key, chart_type, n_intervals):
        try:
            period, interval = PERIOD_INTERVALS.get(interval_key, ("1d", "1m"))
            print(f"[UPDATE] Fetching data for {symbol} - Period: {period}, Interval: {interval}, Update #{n_intervals}")
            
            df = fetch_data(symbol, period, interval)
            
            if df.empty:
                print(f"[WARNING] No data returned for {symbol}")
                empty_fig = {
                    "data": [], 
                    "layout": {
                        "title": f"No data for {symbol}",
                        "plot_bgcolor": "#0a0a0a",
                        "paper_bgcolor": "#0a0a0a",
                    }
                }
                return empty_fig, empty_fig, empty_fig, html.Div(
                    f"No data available for {symbol}. Market may be closed or symbol unavailable.",
                    style={"color": "#ff6b6b", "textAlign": "center", "padding": "20px"}
                )
            
            print(f"[SUCCESS] Fetched {len(df)} rows for {symbol}")
            
            df = calculate_indicators(df)
            metrics = calculate_metrics(df, symbol)
        
            # Main Price Chart
            fig_price = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[1],
        )
        
            if chart_type == "candlestick":
                fig_price.add_trace(
                    go.Candlestick(
                        x=df["Datetime"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="PRICE",
                        increasing_line_color="#7dd87d",
                        decreasing_line_color="#ff6b6b",
                        increasing_fillcolor="rgba(125, 216, 125, 0.8)",
                        decreasing_fillcolor="rgba(255, 107, 107, 0.8)",
                    ),
                    row=1, col=1,
                )
            elif chart_type == "ohlc":
                fig_price.add_trace(
                    go.Ohlc(
                        x=df["Datetime"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="PRICE",
                        increasing_line_color="#7dd87d",
                        decreasing_line_color="#ff6b6b",
                    ),
                    row=1, col=1,
                )
        else:  # line
            fig_price.add_trace(
                go.Scatter(
                    x=df["Datetime"], 
                    y=df["Close"], 
                    mode="lines+markers", 
                    name="CLOSE", 
                    line=dict(color="#7dd87d", width=2.5, shape="spline"),
                    marker=dict(size=3, color="#7dd87d", opacity=0.6),
                    fill="tozeroy",
                    fillcolor="rgba(125, 216, 125, 0.1)",
                ),
                row=1, col=1,
            )
            
            # Add Moving Averages
            if "MA_20" in df.columns:
                fig_price.add_trace(
                    go.Scatter(x=df["Datetime"], y=df["MA_20"], mode="lines", name="MA 20", line=dict(color="#00bfff", width=2.5, dash="dot"), opacity=0.85),
                    row=1, col=1,
                )
            if "MA_50" in df.columns:
                fig_price.add_trace(
                    go.Scatter(x=df["Datetime"], y=df["MA_50"], mode="lines", name="MA 50", line=dict(color="#ff1493", width=2.5, dash="dash"), opacity=0.85),
                    row=1, col=1,
                )
            
            # Add Bollinger Bands
            if "BB_Upper" in df.columns:
                fig_price.add_trace(
                    go.Scatter(x=df["Datetime"], y=df["BB_Upper"], mode="lines", name="BB UPPER", line=dict(color="rgba(125, 216, 125, 0.5)", width=1.5, dash="dash"), showlegend=False, opacity=0.4),
                    row=1, col=1,
                )
                fig_price.add_trace(
                    go.Scatter(x=df["Datetime"], y=df["BB_Lower"], mode="lines", name="BB LOWER", line=dict(color="rgba(125, 216, 125, 0.5)", width=1.5, dash="dash"), fill="tonexty", fillcolor="rgba(125, 216, 125, 0.03)", showlegend=False, opacity=0.4),
                    row=1, col=1,
                )
            
            fig_price.update_layout(
                title=dict(
                    text=f"{symbols[symbol]} ({symbol}) — {interval_key.upper()}",
                    font=dict(color="#7dd87d", size=18, family="'Share Tech Mono', monospace"),
                ),
                xaxis=dict(
                    title=dict(text="TIME", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                    rangeslider=dict(visible=False),
                ),
                yaxis=dict(
                    title=dict(text="PRICE", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                ),
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                hovermode="x unified",
                height=500,
                font=dict(color="#7dd87d", family="'Share Tech Mono', monospace"),
                # Add smooth transitions
                transition=dict(duration=500, easing="cubic-in-out"),
                # Auto-scroll to latest data
                xaxis_rangeslider_visible=False,
            )
            
            # Add smooth animation by updating x-axis range to show latest data
            if len(df) > 0:
                # Show last 100 data points for better live feel
                max_points = min(100, len(df))
                fig_price.update_xaxes(range=[df["Datetime"].iloc[-max_points], df["Datetime"].iloc[-1]])
            
            # RSI Chart
            fig_rsi = go.Figure()
            if "RSI" in df.columns:
                fig_rsi.add_trace(
                    go.Scatter(
                        x=df["Datetime"], 
                        y=df["RSI"], 
                        mode="lines+markers", 
                        name="RSI", 
                        line=dict(color="#00bfff", width=3, shape="spline"),
                        marker=dict(size=4, color="#00bfff", opacity=0.7),
                        fill="tozeroy", 
                        fillcolor="rgba(0, 191, 255, 0.2)",
                    ),
                )
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="rgba(255, 20, 147, 0.8)", line_width=2.5, annotation_text="OVERBOUGHT (70)", annotation_font=dict(color="#ff1493", size=11, family="'Share Tech Mono', monospace"))
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="rgba(125, 216, 125, 0.8)", line_width=2.5, annotation_text="OVERSOLD (30)", annotation_font=dict(color="#7dd87d", size=11, family="'Share Tech Mono', monospace"))
                fig_rsi.add_hline(y=50, line_dash="dot", line_color="rgba(138, 43, 226, 0.5)", opacity=0.5, line_width=1.5)
            
            fig_rsi.update_layout(
                title=dict(
                    text="RSI (RELATIVE STRENGTH INDEX)",
                    font=dict(color="#7dd87d", size=16, family="'Share Tech Mono', monospace"),
                ),
                xaxis=dict(
                    title=dict(text="TIME", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                ),
                yaxis=dict(
                    title=dict(text="RSI", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                    range=[0, 100],
                ),
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                height=300,
                font=dict(color="#7dd87d", family="'Share Tech Mono', monospace"),
                # Add smooth transitions
                transition=dict(duration=500, easing="cubic-in-out"),
            )
            
            # Auto-scroll RSI chart to latest data
            if len(df) > 0:
                max_points = min(100, len(df))
                fig_rsi.update_xaxes(range=[df["Datetime"].iloc[-max_points], df["Datetime"].iloc[-1]])
            
            # Volume Chart
            fig_volume = go.Figure()
            if len(df) > 0:
                # Color code volume bars: red for down, green for up
                colors = []
                for i in range(len(df)):
                    if i < len(df) and pd.notna(df["Close"].iloc[i]) and pd.notna(df["Open"].iloc[i]):
                        colors.append("#ff6b6b" if df["Close"].iloc[i] < df["Open"].iloc[i] else "#7dd87d")
                    else:
                        colors.append("rgba(125, 216, 125, 0.6)")
                fig_volume.add_trace(
                    go.Bar(x=df["Datetime"], y=df["Volume"], name="VOLUME", marker_color=colors if colors else "#7dd87d", marker_line_width=0, opacity=0.8),
                )
            else:
                fig_volume.add_trace(
                    go.Bar(x=[], y=[], name="VOLUME"),
                )
            fig_volume.update_layout(
                title=dict(
                    text="VOLUME",
                    font=dict(color="#7dd87d", size=16, family="'Share Tech Mono', monospace"),
                ),
                xaxis=dict(
                    title=dict(text="TIME", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                ),
                yaxis=dict(
                    title=dict(text="VOLUME", font=dict(color="rgba(125, 216, 125, 0.7)", size=12, family="'Share Tech Mono', monospace")),
                    tickfont=dict(color="rgba(125, 216, 125, 0.6)", size=10, family="'Share Tech Mono', monospace"),
                    gridcolor="rgba(125, 216, 125, 0.08)",
                    linecolor="rgba(125, 216, 125, 0.4)",
                ),
                plot_bgcolor="#0a0a0a",
                paper_bgcolor="#0a0a0a",
                height=250,
                font=dict(color="#7dd87d", family="'Share Tech Mono', monospace"),
                # Add smooth transitions
                transition=dict(duration=500, easing="cubic-in-out"),
            )
            
            # Auto-scroll volume chart to latest data
            if len(df) > 0:
                max_points = min(100, len(df))
                fig_volume.update_xaxes(range=[df["Datetime"].iloc[-max_points], df["Datetime"].iloc[-1]])
            
            # Metrics Cards
            change_class = "positive" if metrics["change"] >= 0 else "negative"
            rsi_class = "positive" if 30 <= metrics["rsi"] <= 70 else "negative" if metrics["rsi"] > 70 else "positive"
            
            metrics_cards = [
                html.Div(
                    [
                    html.Div(metrics["current_price"], className="metric-value", style={"color": "#667eea"}),
                        html.Div("Current Price", className="metric-label"),
                    ],
                    className="metric-card",
                    style={"flex": "1", "minWidth": "150px"},
                ),
                html.Div(
                [
                    html.Div(
                        f"{metrics['change']:+.2f} ({metrics['change_pct']:+.2f}%)",
                        className="metric-value",
                        style={"color": "#00c853" if metrics["change"] >= 0 else "#ff1744"},
                    ),
                    html.Div("Change", className="metric-label"),
                ],
                className="metric-card",
                style={"flex": "1", "minWidth": "150px"},
            ),
            html.Div(
                [
                    html.Div(f"{metrics['high']:.2f}", className="metric-value", style={"color": "#00c853"}),
                    html.Div("High", className="metric-label"),
                ],
                className="metric-card",
                style={"flex": "1", "minWidth": "150px"},
            ),
            html.Div(
                [
                    html.Div(f"{metrics['low']:.2f}", className="metric-value", style={"color": "#ff1744"}),
                    html.Div("Low", className="metric-label"),
                ],
                className="metric-card",
                style={"flex": "1", "minWidth": "150px"},
            ),
            html.Div(
                [
                    html.Div(
                        f"{metrics['rsi']:.1f}",
                        className="metric-value",
                        style={"color": "#ff1744" if metrics["rsi"] > 70 else "#00c853" if metrics["rsi"] < 30 else "#666"},
                    ),
                    html.Div("RSI", className="metric-label"),
                ],
                className="metric-card",
                style={"flex": "1", "minWidth": "150px"},
            ),
            html.Div(
                [
                    html.Div(f"{metrics['volatility']:.2f}%", className="metric-value", style={"color": "#666"}),
                    html.Div("Volatility", className="metric-label"),
                ],
                className="metric-card",
                style={"flex": "1", "minWidth": "150px"},
            ),
            html.Div(
                [
                    html.Div(f"{metrics['volume']:,}", className="metric-value", style={"color": "#666"}),
                    html.Div("Total Volume", className="metric-label"),
                ],
                className="metric-card",
                style={"flex": "1", "minWidth": "150px"},
            ),
            ]
            
            return fig_price, fig_rsi, fig_volume, metrics_cards
        except Exception as e:
            print(f"[ERROR] Exception in update_all: {e}")
            import traceback
            traceback.print_exc()
            error_fig = {
                "data": [],
                "layout": {
                    "title": f"Error loading data: {str(e)}",
                    "plot_bgcolor": "#0a0a0a",
                    "paper_bgcolor": "#0a0a0a",
                }
            }
            error_msg = html.Div(
                f"Error: {str(e)}",
                style={"color": "#ff6b6b", "textAlign": "center", "padding": "20px"}
            )
            return error_fig, error_fig, error_fig, error_msg

    return dash_app

# ---------- MODAL ENTRY POINT ----------
if MODAL_AVAILABLE:
    @app.asgi_app(image=image)
    def dashapp():
        """ASGI entry point for Modal deployment."""
        # Create Dash app inside the function (not at module level)
        dash_instance = create_dash_app()
        
        # Convert WSGI (Dash) to ASGI using asgiref
        asgi_app = WsgiToAsgi(dash_instance.server)
        
        return asgi_app

# ---------- LOCAL TEST / RAILWAY DEPLOYMENT ----------
if __name__ == "__main__":
    import os
    dash_instance = create_dash_app()
    # Railway provides PORT environment variable, default to 8050 for local
    port = int(os.environ.get("PORT", 8050))
    # Disable debug in production (Railway sets RAILWAY_ENVIRONMENT)
    debug = os.environ.get("RAILWAY_ENVIRONMENT") != "production"
    dash_instance.run(host="0.0.0.0", port=port, debug=debug)

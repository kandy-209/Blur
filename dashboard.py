# dashboard_modal.py

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import asyncio
from fastapi import FastAPI
import modal

# ---------- CONFIG ----------
SYMBOLS = ["ES=F", "NQ=F", "GC=F"]
UPDATE_INTERVAL = 60  # seconds
HISTORY_HOURS = 24
RSI_PERIOD = 14

# ---------- DATA ----------
df = pd.DataFrame(columns=["timestamp_utc", "symbol", "price", "RSI"])

# ---------- RSI CALCULATION ----------
def compute_rsi(prices, period=RSI_PERIOD):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------- DATA FETCH ----------
async def fetch_historical_data(symbols=SYMBOLS, retries=3):
    global df
    new_rows = []
    for sym in symbols:
        attempt = 0
        while attempt < retries:
            try:
                data = yf.download(sym, period="1d", interval="1m", progress=False)
                if data.empty:
                    break
                data = data.reset_index()
                data["RSI"] = compute_rsi(data["Close"])
                for _, row in data.iterrows():
                    ts = row["Datetime"]
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    new_rows.append({
                        "timestamp_utc": ts,
                        "symbol": sym,
                        "price": row["Close"],
                        "RSI": row["RSI"]
                    })
                break
            except Exception as e:
                print(f"Error fetching {sym} attempt {attempt+1}: {e}")
                attempt += 1
                await asyncio.sleep(2 ** attempt)
    if new_rows:
        df = pd.DataFrame(new_rows)
        df = df[df["timestamp_utc"] >= datetime.now(timezone.utc) - timedelta(hours=HISTORY_HOURS)]

async def fetch_latest_data(symbols=SYMBOLS):
    global df
    new_rows = []
    for sym in symbols:
        try:
            data = yf.download(sym, period="1d", interval="1m", progress=False)
            if data.empty:
                continue
            last_row = data.reset_index().iloc[-1]
            ts = last_row["Datetime"]
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            sym_df = df[df["symbol"] == sym].copy()
            prices = pd.concat([sym_df["price"], pd.Series([last_row["Close"]])], ignore_index=True)
            rsi_series = compute_rsi(prices)
            new_rows.append({
                "timestamp_utc": ts,
                "symbol": sym,
                "price": last_row["Close"],
                "RSI": rsi_series.iloc[-1]
            })
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
    if new_rows:
        return pd.DataFrame(new_rows)
    return pd.DataFrame(columns=["timestamp_utc", "symbol", "price", "RSI"])

async def background_fetch():
    global df
    while True:
        latest = await fetch_latest_data(SYMBOLS)
        if not latest.empty:
            df = pd.concat([df, latest])
            df = df[df["timestamp_utc"] >= datetime.now(timezone.utc) - timedelta(hours=HISTORY_HOURS)]
        await asyncio.sleep(UPDATE_INTERVAL)

# ---------- DASH APP ----------
def create_dash_app():
    global df
    app = Dash(__name__)
    app.layout = html.Div([
        html.H1("Live Futures Dashboard", style={"textAlign": "center"}),
        html.Div([
            html.Label("Select Symbol(s):"),
            dcc.Dropdown(
                id="symbol-dropdown",
                options=[{"label": s, "value": s} for s in SYMBOLS],
                value=list(SYMBOLS),
                multi=True
            )
        ], style={"width": "45%", "display": "inline-block", "padding": "10px"}),
        dcc.Graph(id="price-chart"),
        dcc.Graph(id="rsi-chart"),
        html.Div(id="summary", style={"padding": "10px", "fontWeight": "bold"}),
        dcc.Interval(id="interval-component", interval=UPDATE_INTERVAL*1000, n_intervals=0)
    ])

    @app.callback(
        [Output("price-chart", "figure"),
         Output("rsi-chart", "figure"),
         Output("summary", "children")],
        [Input("symbol-dropdown", "value"),
         Input("interval-component", "n_intervals")]
    )
    def update_charts(selected_symbols, n):
        global df
        if not selected_symbols:
            return {}, {}, "No symbols selected."

        price_fig = go.Figure()
        rsi_fig = go.Figure()
        summary_lines = []

        for sym in selected_symbols:
            sym_df = df[df["symbol"] == sym]
            if sym_df.empty:
                continue

            price_fig.add_trace(go.Scatter(
                x=sym_df["timestamp_utc"],
                y=sym_df["price"],
                mode="lines+markers",
                name=sym
            ))

            colors = ["red" if r > 70 else "green" if r < 30 else "blue" for r in sym_df["RSI"]]
            rsi_fig.add_trace(go.Bar(
                x=sym_df["timestamp_utc"],
                y=sym_df["RSI"],
                marker_color=colors,
                name=f"{sym} RSI"
            ))

            latest_price = sym_df["price"].iloc[-1]
            latest_rsi = sym_df["RSI"].iloc[-1]
            summary_lines.append(f"{sym}: Latest={latest_price:.2f}, RSI={latest_rsi:.2f}")

        price_fig.update_layout(title="Price Chart", xaxis_title="Time", yaxis_title="Price")
        rsi_fig.update_layout(title="RSI Chart", xaxis_title="Time", yaxis_title="RSI")

        return price_fig, rsi_fig, " | ".join(summary_lines)

    return app

# ---------- MODAL DEPLOY ----------
stub = modal.App(name="live_futures_dashboard")
image = modal.Image.debian_slim().pip_install([
    "dash", "pandas", "plotly", "fastapi", "yfinance"
])

@stub.asgi(image=image)
async def dashapp():
    global df
    await fetch_historical_data()
    asyncio.create_task(background_fetch())
    dash_instance = create_dash_app()
    fastapi_app = FastAPI()
    fastapi_app.mount("/", dash_instance.server)
    return fastapi_app

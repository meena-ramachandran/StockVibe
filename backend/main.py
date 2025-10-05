import os
import httpx

# backend/main.py
import os
import time
import json
import asyncio
from collections import defaultdict
from typing import Dict, Any, List

import httpx
import yfinance as yf
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import dotenv 
from textblob import TextBlob

dotenv.load_dotenv()  # Load environment variables from .env file

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    print("Warning: FINNHUB_API_KEY not set. /search will return an error until you set it.")

app = FastAPI(title="Stock Dashboard API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default curated list (you can expand)
CURATED = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOG": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "TSLA": "Tesla, Inc.",
    "NVDA": "NVIDIA Corporation",
    "NFLX": "Netflix, Inc."
}

# Subscriptions: symbol -> set of WebSocket connections
subscriptions: Dict[str, set] = defaultdict(set)
# Reverse map: websocket -> symbol
ws_to_symbol: Dict[WebSocket, str] = {}

POLL_INTERVAL = 5  # seconds (be mindful of rate limits)

# --- News Sentiment Analysis Endpoint ---
@app.get("/news-sentiment/{symbol}")
async def get_news_sentiment(symbol: str, limit: int = 10):
    """
    Fetch recent news headlines about the stock symbol and analyze sentiment.
    Uses NewsAPI (or similar) and TextBlob for sentiment analysis.
    """
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    if not NEWSAPI_KEY:
        return {"error": "NEWSAPI_KEY not set. Get one at https://newsapi.org/ and set NEWSAPI_KEY env var."}
    url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&language=en&pageSize={limit}&apiKey={NEWSAPI_KEY}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return {"error": f"NewsAPI returned status {r.status_code}"}
            data = r.json()
            articles = data.get("articles", [])
            if not articles:
                return {"symbol": symbol, "count": 0, "error": "No news found."}
            headlines = [a["title"] for a in articles if a.get("title")]
            sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            # Find most positive/negative headline
            # for i,h in enumerate(headlines):
            #     print(h)
            #     print(sentiments[i])
            #     print()
            if sentiments:
                max_idx = sentiments.index(max(sentiments))
                min_idx = sentiments.index(min(sentiments))
                most_positive = headlines[max_idx]
                most_negative = headlines[min_idx]
            else:
                most_positive = most_negative = None
            # Classify
            if avg_sentiment > 0.1:
                summary = "positive"
            elif avg_sentiment < -0.1:
                summary = "negative"
            else:
                summary = "neutral"
            return {
                "symbol": symbol,
                "count": len(headlines),
                "average_sentiment": avg_sentiment,
                "summary": summary,
                "most_positive": most_positive,
                "most_negative": most_negative
            }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

@app.get("/curated")
def get_curated():
    """Return a small curated list of symbols (for quick UI choices)."""
    return CURATED

@app.get("/search")
async def search_symbols(q: str = Query(..., min_length=1)):
    """
    Search symbols using Finnhub. Requires FINNHUB_API_KEY set as env var.
    Returns list of matches with symbol and description.
    """
    if not FINNHUB_API_KEY:
        return {"error": "FINNHUB_API_KEY not set. Get one at https://finnhub.io and set FINNHUB_API_KEY env var."}

    url = "https://finnhub.io/api/v1/search"
    params = {"q": q, "token": FINNHUB_API_KEY}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            return {"error": f"finnhub returned status {r.status_code}"}
        data = r.json()
        # data.get("result") is a list of matches
        results = []
        for item in data.get("result", []):
            results.append({
                "symbol": item.get("symbol"),
                "description": item.get("description"),
                "type": item.get("type"),
                "currency": item.get("currency")
            })
        return {"count": len(results), "results": results}

def compute_analytics_sync(symbol: str, period_days: int = 60) -> Dict[str, Any]:
    """
    Blocking computation: fetch historical daily close prices using yfinance
    and compute moving averages + volatility + percent change.
    """
    ticker = yf.Ticker(symbol)
    # request daily data for the window
    hist = ticker.history(period=f"{period_days}d", interval="1d", auto_adjust=False)
    hist = hist.dropna()
    if hist.empty:
        raise ValueError("No historical data available for symbol: " + symbol)

    close = hist["Close"]
    last_price = float(close.iloc[-1])
    first_price = float(close.iloc[0]) if len(close) > 0 else last_price

    ma_short = float(close.rolling(window=5).mean().iloc[-1]) if len(close) >= 5 else None
    ma_long = float(close.rolling(window=20).mean().iloc[-1]) if len(close) >= 20 else None

    returns = close.pct_change().dropna()
    volatility = float(returns.std() * (252 ** 0.5)) if not returns.empty else None  # annualized

    percent_change = ((last_price - first_price) / first_price) * 100 if first_price != 0 else None

    labels = [d.strftime("%Y-%m-%d") for d in hist.index]
    prices = [float(x) for x in close.round(4).tolist()]

    return {
        "symbol": symbol,
        "last_price": round(last_price, 4),
        "ma_short": round(ma_short, 4) if ma_short is not None else None,
        "ma_long": round(ma_long, 4) if ma_long is not None else None,
        "volatility_annualized": round(volatility, 6) if volatility is not None else None,
        "percent_change_period": round(percent_change, 4) if percent_change is not None else None,
        "history": {
            "labels": labels,
            "prices": prices
        }
    }

@app.get("/analytics/{symbol}")
async def get_analytics(symbol: str, period_days: int = 60):
    """
    Return analytics (MA, volatility, history) for a symbol.
    This runs blocking yfinance calls in a threadpool so it won't block the event loop.
    """
    loop = asyncio.get_event_loop()
    try:
        out = await loop.run_in_executor(None, compute_analytics_sync, symbol, period_days)
        return out
    except Exception as e:
        return {"error": str(e)}

# Helper to synchronous yfinance price fetch (used in background poll)
def get_latest_price_sync(symbol: str):
    ticker = yf.Ticker(symbol)
    # try 1-minute intraday first
    try:
        intraday = ticker.history(period="1d", interval="1m")
        if not intraday.empty:
            return float(intraday["Close"].iloc[-1])
    except Exception:
        pass
    # fallback to info
    try:
        info = ticker.info
        return float(info.get("currentPrice") or info.get("regularMarketPrice"))
    except Exception:
        return None

async def get_latest_price(symbol: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_latest_price_sync, symbol)

async def poll_loop():
    """Background task: poll latest prices for subscribed symbols and broadcast."""
    while True:
        try:
            symbols = list(subscriptions.keys())
            if symbols:
                # fetch prices for each symbol serially (could be parallelized)
                for sym in symbols:
                    price = await get_latest_price(sym)
                    payload = {"type": "price", "symbol": sym, "price": price, "ts": int(time.time())}
                    conns = list(subscriptions[sym])
                    for ws in conns:
                        try:
                            await ws.send_text(json.dumps(payload))
                        except Exception:
                            # remove dead connections
                            subscriptions[sym].discard(ws)
                            ws_to_symbol.pop(ws, None)
        except Exception as e:
            print("Poll loop error:", e)
        await asyncio.sleep(POLL_INTERVAL)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(poll_loop())

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        # default subscription (none until client subscribes)
        ws_to_symbol[ws] = None
        while True:
            data = await ws.receive_text()
            try:
                obj = json.loads(data)
            except Exception:
                await ws.send_text(json.dumps({"error": "invalid json"}))
                continue

            action = obj.get("action")
            if action == "subscribe":
                symbol = obj.get("symbol")
                if not symbol:
                    await ws.send_text(json.dumps({"error": "no symbol provided"}))
                    continue
                # remove from previous symbol set
                prev = ws_to_symbol.get(ws)
                if prev:
                    subscriptions[prev].discard(ws)
                # add to new symbol
                subscriptions[symbol].add(ws)
                ws_to_symbol[ws] = symbol
                await ws.send_text(json.dumps({"type": "subscribed", "symbol": symbol}))
            else:
                await ws.send_text(json.dumps({"error": "unknown action"}))

    except WebSocketDisconnect:
        # cleanup
        symbol = ws_to_symbol.pop(ws, None)
        if symbol:
            subscriptions[symbol].discard(ws)
        # nothing else needed

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
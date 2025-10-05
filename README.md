# StockVibe

StockVibe is an analytics dashboard for real-time stock tracking and news sentiment analysis. It leverages a FastAPI backend and a vanilla JavaScript frontend to display live stock prices and analyze the market sentiment of related financial news.

## Features

- **Stock Search & Directory**: Search for stocks by symbol or company name, and browse popular pre-configured selections.
- **Real-Time Price Analytics**: Live price updates pushed over WebSockets. Interactive historical charts display moving averages and volatility indicators.
- **Sentiment Analysis**: Fetches news headlines for target stock symbols and performs sentiment analysis using Natural Language Processing (NLP) to classify news context.
- **Visual Dashboard**: Client interface built with HTML, CSS, and Chart.js for visualization.

## Architecture

- **Frontend**: Single-page application using HTML, CSS (vanilla), and Chart.js.
- **Backend**: Python-based FastAPI server providing REST endpoints and WebSocket connections.
  - Data ingestion via `yfinance` for price statistics and metadata.
  - Sentiment modeling via `TextBlob` for linguistic analysis.
  - News headlines queried from `NewsAPI`.

## Setup Instructions

### 1. Backend Setup

Prerequisites: Python 3.11+

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. Create a `.env` file in the project root containing your API credentials:
   ```env
   NEWSAPI_KEY=your_newsapi_key_here
   FINNHUB_API_KEY=your_finnhub_key_here
   ```
   *Note: A news API key can be obtained at [newsapi.org](https://newsapi.org).*

4. Run the FastAPI development server:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

### 2. Frontend Setup

1. Serve the frontend assets using any HTTP file server (e.g., Python's built-in server):
   ```bash
   cd frontend
   python -m http.server 3000
   ```

2. Open the dashboard in your browser at `http://localhost:3000`.

## License

This project is licensed under the MIT License.

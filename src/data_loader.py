"""Data Loader Module
Handles downloading, validating, and caching financial market data from Yahoo Finance.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf
import streamlit as st


# Pre-configured popular tickers for easy user selection
POPULAR_TICKERS = {
    "US Tech & Equities": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Cryptocurrencies": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD", "ADA-USD"],
    "Indian Equities (NSE)": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "TATAMOTORS.NS"],
    "Commodities & Indices": ["SPY", "QQQ", "GLD", "SLV", "^NSEI"],
}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetch historical OHLCV data for a ticker with in-memory caching (TTL: 15 mins).

    Args:
        ticker: Symbol string (e.g. 'AAPL', 'BTC-USD').
        period: Data duration ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max').
        interval: Data granularity ('1d', '1wk', '1mo').

    Returns:
        Tuple of (DataFrame, error_message). If successful, error_message is None.
    """
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        return None, "Ticker symbol cannot be empty."

    try:
        t = yf.Ticker(clean_ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)

        if df is None or df.empty:
            return None, f"No market data found for ticker '{clean_ticker}'. Please verify the symbol."

        # Clean index and timezone
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.index.name = "Date"
        df.reset_index(inplace=True)

        # Standardize column naming
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                return None, f"Malformed data returned: missing column '{col}'"

        df = df[required_cols].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df, None

    except Exception as exc:
        return None, f"Failed to fetch data for {clean_ticker}: {str(exc)}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_metadata(ticker: str) -> Dict:
    """Fetch fundamental details and profile metadata for a given ticker."""
    clean_ticker = ticker.strip().upper()
    metadata = {
        "symbol": clean_ticker,
        "name": clean_ticker,
        "currency": "USD",
        "market_cap": None,
        "summary": "No company description available.",
        "pe_ratio": None,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
        "sector": "N/A",
    }

    try:
        info = yf.Ticker(clean_ticker).info
        if info:
            metadata["name"] = info.get("longName") or info.get("shortName") or clean_ticker
            metadata["currency"] = info.get("currency", "USD")
            metadata["market_cap"] = info.get("marketCap")
            metadata["summary"] = info.get("longBusinessSummary", "No company description available.")
            metadata["pe_ratio"] = info.get("trailingPE")
            metadata["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh")
            metadata["fifty_two_week_low"] = info.get("fiftyTwoWeekLow")
            metadata["sector"] = info.get("sector", "N/A")
    except Exception:
        # Fallback to defaults on rate limit or incomplete info
        pass

    return metadata


@st.cache_data(ttl=900, show_spinner=False)
def fetch_multi_asset_data(
    tickers: List[str],
    period: str = "6mo"
) -> pd.DataFrame:
    """Fetch closing prices for multiple assets to compute cumulative returns and correlations."""
    closes = {}
    for ticker in tickers:
        df, err = fetch_ticker_data(ticker, period=period)
        if df is not None and not df.empty:
            closes[ticker] = df.set_index("Date")["Close"]

    if not closes:
        return pd.DataFrame()

    combined = pd.DataFrame(closes).dropna(how="all")
    return combined

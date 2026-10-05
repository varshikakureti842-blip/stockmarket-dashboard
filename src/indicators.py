"""Technical Indicators Engine
Computes core quantitative indicators: SMA, EMA, Bollinger Bands, RSI, and MACD.
"""

from typing import List
import numpy as np
import pandas as pd


def compute_sma(df: pd.DataFrame, windows: List[int] = [20, 50, 200], price_col: str = "Close") -> pd.DataFrame:
    """Compute Simple Moving Averages for specified window lengths."""
    data = df.copy()
    for w in windows:
        col_name = f"SMA_{w}"
        data[col_name] = data[price_col].rolling(window=w, min_periods=1).mean()
    return data


def compute_ema(df: pd.DataFrame, spans: List[int] = [12, 26], price_col: str = "Close") -> pd.DataFrame:
    """Compute Exponential Moving Averages for specified spans."""
    data = df.copy()
    for s in spans:
        col_name = f"EMA_{s}"
        data[col_name] = data[price_col].ewm(span=s, adjust=False).mean()
    return data


def compute_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_col: str = "Close"
) -> pd.DataFrame:
    """Compute Bollinger Bands (Middle SMA, Upper Band, Lower Band, and Bandwidth)."""
    data = df.copy()
    rolling = data[price_col].rolling(window=window, min_periods=window // 2)
    middle = rolling.mean()
    std = rolling.std()

    data[f"BB_Middle_{window}"] = middle
    data[f"BB_Upper_{window}"] = middle + (num_std * std)
    data[f"BB_Lower_{window}"] = middle - (num_std * std)
    data[f"BB_Width_{window}"] = (data[f"BB_Upper_{window}"] - data[f"BB_Lower_{window}"]) / middle
    return data


def compute_rsi(df: pd.DataFrame, window: int = 14, price_col: str = "Close") -> pd.DataFrame:
    """Compute Relative Strength Index (RSI) using Wilder's Exponential Smoothing."""
    data = df.copy()
    delta = data[price_col].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing equivalent: alpha = 1 / window
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    # Prevent division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Fill initial NaNs with neutral 50
    data[f"RSI_{window}"] = rsi.fillna(50.0)
    return data


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "Close"
) -> pd.DataFrame:
    """Compute MACD Line, Signal Line, and MACD Histogram."""
    data = df.copy()
    ema_fast = data[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = data[price_col].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    data["MACD_Line"] = macd_line
    data["MACD_Signal"] = signal_line
    data["MACD_Hist"] = hist
    return data


def enrich_with_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience pipeline to calculate all indicators in one call."""
    enriched = compute_sma(df, windows=[20, 50, 200])
    enriched = compute_ema(enriched, spans=[12, 26])
    enriched = compute_bollinger_bands(enriched, window=20, num_std=2.0)
    enriched = compute_rsi(enriched, window=14)
    enriched = compute_macd(enriched, fast=12, slow=26, signal=9)

    # Calculate returns and rolling volatility
    enriched["Daily_Return"] = enriched["Close"].pct_change()
    enriched["Volatility_20d"] = enriched["Daily_Return"].rolling(20).std() * np.sqrt(252)

    return enriched

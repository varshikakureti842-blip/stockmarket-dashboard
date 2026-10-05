"""Visualization Engine
Generates interactive Plotly charts: Candlesticks with Volume, RSI, MACD,
ML Forecast Projections, and Normalized Multi-Asset Performance Comparisons.
"""

from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CHART_THEME = "plotly_dark"
COLOR_PALETTE = {
    "up_candle": "#26a69a",
    "down_candle": "#ef5350",
    "volume_up": "rgba(38, 166, 154, 0.4)",
    "volume_down": "rgba(239, 83, 80, 0.4)",
    "sma_20": "#ffeb3b",
    "sma_50": "#ff9800",
    "sma_200": "#00e5ff",
    "bb_upper": "rgba(100, 181, 246, 0.6)",
    "bb_lower": "rgba(100, 181, 246, 0.6)",
    "bb_fill": "rgba(33, 150, 243, 0.12)",
    "forecast": "#e040fb",
    "forecast_fill": "rgba(224, 64, 251, 0.18)",
}


def plot_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    active_indicators: List[str]
) -> go.Figure:
    """Create interactive candlestick chart with volume subplot and technical overlays."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # 1. Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=f"{ticker} Price",
            increasing_line_color=COLOR_PALETTE["up_candle"],
            decreasing_line_color=COLOR_PALETTE["down_candle"],
        ),
        row=1,
        col=1,
    )

    # 2. Moving Average Overlays
    if "SMA 20" in active_indicators and "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["SMA_20"], name="SMA 20", line=dict(color=COLOR_PALETTE["sma_20"], width=1.5)),
            row=1, col=1,
        )

    if "SMA 50" in active_indicators and "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["SMA_50"], name="SMA 50", line=dict(color=COLOR_PALETTE["sma_50"], width=1.5)),
            row=1, col=1,
        )

    if "SMA 200" in active_indicators and "SMA_200" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["Date"], y=df["SMA_200"], name="SMA 200", line=dict(color=COLOR_PALETTE["sma_200"], width=1.8)),
            row=1, col=1,
        )

    # 3. Bollinger Bands Overlay
    if "Bollinger Bands" in active_indicators and "BB_Upper_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["BB_Upper_20"],
                name="BB Upper (20,2)",
                line=dict(color=COLOR_PALETTE["bb_upper"], width=1, dash="dash"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["BB_Lower_20"],
                name="BB Lower (20,2)",
                line=dict(color=COLOR_PALETTE["bb_lower"], width=1, dash="dash"),
                fill="tonexty",
                fillcolor=COLOR_PALETTE["bb_fill"],
            ),
            row=1, col=1,
        )

    # 4. Volume Bar Subplot
    vol_colors = [
        COLOR_PALETTE["volume_up"] if c >= o else COLOR_PALETTE["volume_down"]
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Volume",
            marker_color=vol_colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template=CHART_THEME,
        title=f"<b>{ticker}</b> Price Action & Volume",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", showgrid=False),
    )
    return fig


def plot_rsi(df: pd.DataFrame, window: int = 14) -> go.Figure:
    """Create Relative Strength Index (RSI) chart with overbought/oversold bands."""
    fig = go.Figure()
    col = f"RSI_{window}"

    fig.add_trace(
        go.Scatter(x=df["Date"], y=df[col], name=f"RSI ({window})", line=dict(color="#ab47bc", width=2))
    )

    # Overbought threshold (70)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought (70)")
    # Oversold threshold (30)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="Oversold (30)")

    fig.update_layout(
        template=CHART_THEME,
        title=f"<b>Relative Strength Index (RSI {window})</b>",
        yaxis=dict(range=[0, 100], title="RSI"),
        margin=dict(l=40, r=40, t=50, b=30),
        height=260,
    )
    return fig


def plot_macd(df: pd.DataFrame) -> go.Figure:
    """Create MACD line, signal line, and color-coded momentum histogram."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["MACD_Line"], name="MACD Line", line=dict(color="#29b6f6", width=1.8))
    )
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="Signal Line", line=dict(color="#ffa726", width=1.8))
    )

    hist_colors = [
        "#26a69a" if val >= 0 else "#ef5350" for val in df["MACD_Hist"]
    ]
    fig.add_trace(
        go.Bar(x=df["Date"], y=df["MACD_Hist"], name="Histogram", marker_color=hist_colors)
    )

    fig.update_layout(
        template=CHART_THEME,
        title="<b>MACD (12, 26, 9)</b>",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=30),
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_ml_forecast(
    df: pd.DataFrame,
    forecast_results: Dict,
    ticker: str,
    recent_days: int = 90
) -> go.Figure:
    """Visualize actual historical prices, test set predictions, and future projection cone."""
    fig = go.Figure()

    # Slice recent historical data for visual clarity
    recent_df = df.tail(recent_days)

    # 1. Historical Actual Close
    fig.add_trace(
        go.Scatter(
            x=recent_df["Date"],
            y=recent_df["Close"],
            name="Actual Price",
            line=dict(color="#90caf9", width=2.5),
        )
    )

    # 2. Test Set Model Fit
    test_dates = forecast_results["test_dates"]
    test_preds = forecast_results["y_test_pred"]
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=test_preds,
            name="Out-of-Sample Test Fit",
            line=dict(color="#ffd54f", width=1.8, dash="dot"),
        )
    )

    # 3. Future Forecast
    fut_dates = forecast_results["future_dates"]
    fut_preds = forecast_results["future_preds"]
    upper = forecast_results["upper_bound"]
    lower = forecast_results["lower_bound"]

    # Connect seamlessly from the last actual point to the first future point
    last_actual_date = recent_df["Date"].iloc[-1]
    last_actual_price = recent_df["Close"].iloc[-1]

    plot_fut_dates = [last_actual_date] + list(fut_dates)
    plot_fut_preds = [last_actual_price] + list(fut_preds)
    plot_upper = [last_actual_price] + list(upper)
    plot_lower = [last_actual_price] + list(lower)

    # Confidence Interval Ribbon
    fig.add_trace(
        go.Scatter(
            x=plot_fut_dates,
            y=plot_upper,
            name="Upper 95% Confidence Bound",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_fut_dates,
            y=plot_lower,
            name="95% Confidence Interval",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=COLOR_PALETTE["forecast_fill"],
        )
    )

    # Future Expected Trajectory
    fig.add_trace(
        go.Scatter(
            x=plot_fut_dates,
            y=plot_fut_preds,
            name=f"Projected Forecast ({forecast_results['model_name']})",
            line=dict(color=COLOR_PALETTE["forecast"], width=3, dash="dash"),
        )
    )

    model_name = forecast_results["model_name"]
    mae = forecast_results["metrics"]["MAE"]
    mape = forecast_results["metrics"]["MAPE"]

    fig.update_layout(
        template=CHART_THEME,
        title=f"<b>{ticker} AI Price Projection</b> ({model_name} | Test MAE: ${mae} | MAPE: {mape}%)",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_multi_asset_comparison(combined_df: pd.DataFrame) -> go.Figure:
    """Plot cumulative percentage returns normalized to 0% at start."""
    fig = go.Figure()

    # Normalize to percentage gain from day 0
    norm_df = ((combined_df / combined_df.iloc[0]) - 1.0) * 100.0

    for col in norm_df.columns:
        fig.add_trace(
            go.Scatter(x=norm_df.index, y=norm_df[col], name=col, line=dict(width=2))
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        template=CHART_THEME,
        title="<b>Cumulative Relative Returns (%)</b> (Normalized to Day 0)",
        xaxis_title="Date",
        yaxis_title="Percentage Return (%)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Plot interactive heatmap for asset correlation matrix."""
    import numpy as np
    z_vals = np.round(corr_df.values, 3)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale="RdBu",
            zmin=-1.0,
            zmax=1.0,
            text=z_vals,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "white"},
            colorbar=dict(title="Pearson r"),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        template=CHART_THEME,
        title="<b>Asset Return Correlation Matrix</b>",
        margin=dict(l=50, r=40, t=50, b=40),
        height=380,
    )
    return fig


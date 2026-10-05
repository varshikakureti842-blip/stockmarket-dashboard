"""FinMetrics - AI Stock & Crypto Analytics & Forecasting Terminal
Main Streamlit Application.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import (
    POPULAR_TICKERS,
    fetch_ticker_data,
    fetch_ticker_metadata,
    fetch_multi_asset_data,
)
from src.indicators import enrich_with_all_indicators
from src.model import train_and_evaluate_forecaster
from src.visualizer import (
    plot_candlestick_chart,
    plot_rsi,
    plot_macd,
    plot_ml_forecast,
    plot_multi_asset_comparison,
    plot_correlation_heatmap,
)

# Set page configuration
st.set_page_config(
    page_title="FinMetrics | AI Stock & Crypto Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a professional dark-terminal aesthetic
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        padding: 12px 18px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1e28;
        border-radius: 6px 6px 0px 0px;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2962ff !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


def format_currency(val, currency="USD"):
    if val is None or np.isnan(val):
        return "N/A"
    symbol = "$" if currency == "USD" else (currency + " ")
    if abs(val) >= 1e12:
        return f"{symbol}{val/1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"{symbol}{val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"{symbol}{val/1e6:.2f}M"
    return f"{symbol}{val:,.2f}"


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("📈 FinMetrics Terminal")
st.sidebar.caption("Real-Time Analytics & Machine Learning Forecaster")

# 1. Asset Selection Mode
category = st.sidebar.selectbox("Market Category", list(POPULAR_TICKERS.keys()) + ["Custom Symbol"])

if category == "Custom Symbol":
    selected_ticker = st.sidebar.text_input("Enter Ticker (e.g. NVDA, BTC-USD, RELIANCE.NS):", value="NVDA").strip().upper()
else:
    options = POPULAR_TICKERS[category]
    selected_ticker = st.sidebar.selectbox("Select Asset", options, index=0)

# 2. Historical Lookback Period
PERIOD_MAP = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "Max": "max",
}
selected_period_label = st.sidebar.selectbox("Historical Lookback", list(PERIOD_MAP.keys()), index=3)
selected_period = PERIOD_MAP[selected_period_label]

st.sidebar.markdown("---")
st.sidebar.subheader("Technical Overlays")
show_sma20 = st.sidebar.checkbox("SMA 20 (Short-term)", value=True)
show_sma50 = st.sidebar.checkbox("SMA 50 (Medium-term)", value=True)
show_sma200 = st.sidebar.checkbox("SMA 200 (Long-term)", value=False)
show_bb = st.sidebar.checkbox("Bollinger Bands (20, 2)", value=True)
show_rsi = st.sidebar.checkbox("RSI (14-period)", value=True)
show_macd = st.sidebar.checkbox("MACD (12, 26, 9)", value=True)

active_indicators = []
if show_sma20: active_indicators.append("SMA 20")
if show_sma50: active_indicators.append("SMA 50")
if show_sma200: active_indicators.append("SMA 200")
if show_bb: active_indicators.append("Bollinger Bands")

if st.sidebar.button("Clear Cache & Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("💡 **Interview Tip:** All data ingestion uses `@st.cache_data(ttl=900)` to optimize API traffic and eliminate redundant network overhead.")


# ==============================================================================
# DATA INGESTION & PROCESSING
# ==============================================================================
with st.spinner(f"Ingesting market data for {selected_ticker}..."):
    raw_df, error_msg = fetch_ticker_data(selected_ticker, period=selected_period)
    metadata = fetch_ticker_metadata(selected_ticker)

if error_msg or raw_df is None or raw_df.empty:
    st.error(f"❌ Error: {error_msg}")
    st.stop()

# Enrich data with quantitative indicators
df = enrich_with_all_indicators(raw_df)

# ==============================================================================
# TOP SUMMARY & KPI BANNER
# ==============================================================================
latest_row = df.iloc[-1]
prev_row = df.iloc[-2] if len(df) > 1 else latest_row

curr_price = latest_row["Close"]
delta_val = curr_price - prev_row["Close"]
pct_change = (delta_val / prev_row["Close"]) * 100.0 if prev_row["Close"] != 0 else 0.0

col_meta, col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns([2.5, 1.2, 1.2, 1.2, 1.2])

with col_meta:
    st.title(f"{metadata['name']} ({selected_ticker})")
    st.caption(f"Sector: **{metadata['sector']}** | Currency: **{metadata['currency']}** | Records: **{len(df):,} trading days**")

with col_kpi1:
    st.metric(
        label="Latest Close",
        value=format_currency(curr_price, metadata["currency"]),
        delta=f"{delta_val:+.2f} ({pct_change:+.2f}%)",
    )

with col_kpi2:
    st.metric(
        label="Day Range (H / L)",
        value=f"{format_currency(latest_row['High'], metadata['currency'])}",
        delta=f"Low: {format_currency(latest_row['Low'], metadata['currency'])}",
        delta_color="off",
    )

with col_kpi3:
    st.metric(
        label="Volume",
        value=f"{int(latest_row['Volume']):,}",
    )

with col_kpi4:
    mcap_str = format_currency(metadata["market_cap"], metadata["currency"]) if metadata["market_cap"] else "N/A"
    st.metric(
        label="Market Cap",
        value=mcap_str,
    )

st.markdown("---")

# ==============================================================================
# MAIN DASHBOARD TABS
# ==============================================================================
tab_charts, tab_forecast, tab_compare, tab_interview = st.tabs([
    "📊 Market & Technicals",
    "🤖 Machine Learning Forecast",
    "🔄 Multi-Asset Comparison",
    "🎓 Interview & Architecture Dossier"
])

# ------------------------------------------------------------------------------
# TAB 1: MARKET CHARTS & TECHNICAL ANALYSIS
# ------------------------------------------------------------------------------
with tab_charts:
    chart_col, info_col = st.columns([3.5, 1.2])

    with chart_col:
        # Main Candlestick Chart
        candlestick_fig = plot_candlestick_chart(df, selected_ticker, active_indicators)
        st.plotly_chart(candlestick_fig, use_container_width=True)

        # Oscillators (RSI & MACD)
        if show_rsi:
            rsi_fig = plot_rsi(df, window=14)
            st.plotly_chart(rsi_fig, use_container_width=True)

        if show_macd:
            macd_fig = plot_macd(df)
            st.plotly_chart(macd_fig, use_container_width=True)

    with info_col:
        st.subheader("Asset Profile")
        with st.expander("Company Description", expanded=True):
            st.write(metadata["summary"][:600] + ("..." if len(metadata["summary"]) > 600 else ""))

        st.subheader("Technical Gauge")
        latest_rsi = df["RSI_14"].iloc[-1]
        rsi_status = "Overbought (>=70)" if latest_rsi >= 70 else ("Oversold (<=30)" if latest_rsi <= 30 else "Neutral")
        rsi_color = "red" if latest_rsi >= 70 else ("green" if latest_rsi <= 30 else "blue")

        st.markdown(f"**RSI (14):** `{latest_rsi:.2f}` — :{rsi_color}[**{rsi_status}**]")

        latest_sma20 = df["SMA_20"].iloc[-1]
        latest_sma50 = df["SMA_50"].iloc[-1]
        trend = "Bullish (SMA 20 > SMA 50)" if latest_sma20 > latest_sma50 else "Bearish (SMA 20 < SMA 50)"
        st.markdown(f"**Short/Medium Trend:** **{trend}**")

        st.subheader("Volatility & Risk")
        vol_20d = df["Volatility_20d"].iloc[-1]
        st.markdown(f"**Annualized 20D Volatility:** `{vol_20d*100:.1f}%`" if not np.isnan(vol_20d) else "N/A")

        if metadata["pe_ratio"]:
            st.markdown(f"**Trailing P/E Ratio:** `{metadata['pe_ratio']:.2f}`")

    # Data Table Expander
    with st.expander("📋 View Cleaned Historical OHLCV Dataset"):
        st.dataframe(df.sort_values("Date", ascending=False), use_container_width=True)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Historical CSV",
            data=csv_data,
            file_name=f"{selected_ticker}_historical_data.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------------------------
# TAB 2: MACHINE LEARNING PRICE FORECASTING
# ------------------------------------------------------------------------------
with tab_forecast:
    st.subheader("🤖 Time-Series Machine Learning Forecaster")
    st.caption("Autoregressive multi-step projection with strict chronological train/test split to prevent lookahead bias.")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        model_choice = st.selectbox("Estimator Algorithm", ["Ridge Regression (L2)", "Random Forest Regressor"])
    with f_col2:
        train_split = st.slider("Chronological Train Split Ratio", min_value=0.70, max_value=0.90, value=0.80, step=0.05)
    with f_col3:
        forecast_days = st.slider("Future Forecast Horizon (Days)", min_value=5, max_value=30, value=14, step=1)

    algo_name = "Ridge" if "Ridge" in model_choice else "Random Forest"

    with st.spinner("Engineering lag features and training model..."):
        forecast_res, ml_err = train_and_evaluate_forecaster(
            df=df,
            model_type=algo_name,
            train_ratio=train_split,
            forecast_horizon=forecast_days,
        )

    if ml_err:
        st.warning(f"⚠️ Unable to generate forecast: {ml_err}")
    else:
        # Show evaluation metrics
        m = forecast_res["metrics"]
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Test MAE (Mean Absolute Error)", f"${m['MAE']}")
        m_col2.metric("Test RMSE (Root Mean Sq Error)", f"${m['RMSE']}")
        m_col3.metric("Test MAPE", f"{m['MAPE']}%")
        m_col4.metric("Test R² Score", f"{m['R2']}")

        # Render forecast chart
        forecast_fig = plot_ml_forecast(df, forecast_res, selected_ticker)
        st.plotly_chart(forecast_fig, use_container_width=True)

        # Forecast Projection Table
        st.subheader("📅 Projected Price Trajectory & Confidence Bounds")
        proj_df = pd.DataFrame({
            "Forecast Date": [d.strftime("%Y-%m-%d") for d in forecast_res["future_dates"]],
            "Projected Close ($)": np.round(forecast_res["future_preds"], 2),
            "Lower 95% Bound ($)": np.round(forecast_res["lower_bound"], 2),
            "Upper 95% Bound ($)": np.round(forecast_res["upper_bound"], 2),
        })
        st.dataframe(proj_df, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: MULTI-ASSET PERFORMANCE COMPARISON
# ------------------------------------------------------------------------------
with tab_compare:
    st.subheader("🔄 Multi-Asset Cumulative Performance Comparison")
    st.caption("Normalize multiple equities or cryptos to 0% at origin to compare relative alpha and beta.")

    default_compare = ["AAPL", "MSFT", "NVDA", "BTC-USD"]
    compare_tickers = st.multiselect(
        "Select Assets to Compare:",
        options=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD"],
        default=[t for t in default_compare if t in ["AAPL", "MSFT", "NVDA", "BTC-USD"]],
    )

    if len(compare_tickers) >= 2:
        with st.spinner("Fetching comparative asset histories..."):
            comp_df = fetch_multi_asset_data(compare_tickers, period=selected_period)

        if not comp_df.empty:
            comp_fig = plot_multi_asset_comparison(comp_df)
            st.plotly_chart(comp_fig, use_container_width=True)

            # Correlation Matrix Heatmap
            corr = comp_df.pct_change().corr()
            corr_fig = plot_correlation_heatmap(corr)
            st.plotly_chart(corr_fig, use_container_width=True)

            with st.expander("📊 View Correlation Data Table"):
                st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None).format("{:.3f}"), use_container_width=True)
        else:
            st.warning("Could not fetch data for the selected assets.")
    else:
        st.info("Please select at least 2 assets to view performance comparison and correlation.")


# ------------------------------------------------------------------------------
# TAB 4: INTERVIEW & ARCHITECTURE DOSSIER
# ------------------------------------------------------------------------------
with tab_interview:
    st.subheader("🎓 Technical Architecture & Interview Talking Points")

    st.markdown(r"""
    ### 1. System Architecture
    ```
    Yahoo Finance API (yfinance)
             │
             ▼
    Caching Layer (@st.cache_data, TTL=900s)
             │
             ▼
    Feature & Indicators Engine (Pandas, NumPy)
      - SMA (20, 50, 200), EMA (12, 26)
      - Bollinger Bands (20, 2σ)
      - Wilder's Smoothed RSI (14)
      - MACD (12, 26, 9)
             │
             ├────────────────────────────────┐
             ▼                                ▼
    Plotly Graph Engine              Time-Series ML Pipeline
      - Candlestick + Volume Subplot   - Autoregressive Lag Features (t-1, ..., t-10)
      - Oscillators (RSI / MACD)       - Chronological 80/20 Partition
      - Performance Normalization      - Ridge / Random Forest Regressors
             │                         - Out-of-sample Horizon Projection
             └────────────────────────────────┘
                             │
                             ▼
                     Streamlit Front-End UI
    ```

    ---

    ### 2. Crucial Design Decisions (Why did you build it this way?)
    * **Preventing Data Leakage in Time Series:** Random train/test split (e.g. `shuffle=True`) causes catastrophic lookahead bias because future prices leak into past training points. This system enforces strict chronological ordering.
    * **L2 Regularized Ridge Regression:** Lag features in financial time-series are highly collinear ($t-1$ correlates strongly with $t-2$). Ridge regression applies an L2 penalty ($\lambda \sum w_i^2$) to stabilize weight coefficients and reduce variance.
    * **Wilder's RSI Smoothing:** Rather than standard arithmetic moving averages, this engine implements Wilder's Exponential Smoothing ($\alpha = 1 / 14$) matching the institutional standard used by Bloomberg and TradingView.
    * **Performance Optimization:** Implemented Streamlit caching with Time-To-Live expiration, allowing sub-second UI interactivity and slider reactivity without triggering Yahoo Finance rate limits.

    ---

    ### 3. Model Evaluation Metrics
    * **MAE (Mean Absolute Error):** Measures average dollar error: $\\text{MAE} = \\frac{1}{n} \\sum |y_i - \\hat{y}_i|$.
    * **RMSE (Root Mean Squared Error):** Penalizes extreme deviations: $\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum (y_i - \\hat{y}_i)^2}$.
    * **MAPE (Mean Absolute Percentage Error):** Normalized percentage error across differing asset price scales: $\\text{MAPE} = \\frac{100\\%}{n} \\sum |\\frac{y_i - \\hat{y}_i}{y_i}|$.
    """)

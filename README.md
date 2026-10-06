# FinMetrics: AI Stock & Crypto Analytics & Forecasting Terminal

A real-time financial market analytics dashboard and machine learning forecasting engine for global equities, ETFs, and cryptocurrencies. Built entirely in Python with zero external paid APIs.

> 📖 **Full System Architecture & Setup Guide:** See [`docs/COMPLETE_PROJECT_ARCH_AND_SETUP.md`](docs/COMPLETE_PROJECT_ARCH_AND_SETUP.md) for complete end-to-end documentation, terminal setup instructions, and chronological project records.
> 
> 🎓 **Zero-to-Hero Learning Guide (Terminology & Math):** See [`docs/ZERO_TO_HERO_LEARNING_GUIDE.md`](docs/ZERO_TO_HERO_LEARNING_GUIDE.md) to learn all financial, machine learning, and engineering terms from scratch.

---

## 🌟 Key Features

1. **Real-Time & Historical Data Ingestion**
   - Live and historical OHLCV data for US tech giants (AAPL, NVDA, MSFT), Cryptos (BTC, ETH, SOL), Indian Equities (RELIANCE, TCS), and commodities via Yahoo Finance.
   - Built-in TTL caching (`@st.cache_data`) to prevent rate-limiting and accelerate dashboard response times.

2. **Quantitative Technical Indicators**
   - **Trend:** Simple Moving Averages (SMA 20, 50, 200) and Exponential Moving Averages (EMA 12, 26).
   - **Volatility:** Bollinger Bands (20-period baseline, 2σ upper/lower envelopes, bandwidth) and annualized 20-day volatility.
   - **Momentum:** Institutional-grade Relative Strength Index (RSI 14) with Wilder's exponential smoothing and overbought/oversold bands.
   - **Oscillators:** Moving Average Convergence Divergence (MACD 12, 26, 9) with signal line and color-coded momentum histogram.

3. **Time-Series Machine Learning Forecaster**
   - **Zero Lookahead Bias:** Implements strict chronological 80/20 train-test splitting to eliminate data leakage.
   - **Autoregressive Feature Engineering:** Generates multi-step lag features ($t-1, \dots, t-10$), rolling statistics, and calendar variables.
   - **Models:** Ridge Regression with L2 regularization (handles multicollinear lag features) and Random Forest Regressor.
   - **Out-of-Sample Forecasting:** Projects multi-day trajectories with expanding 95% confidence intervals.
   - **Quantitative Metrics:** Evaluates out-of-sample performance using MAE, RMSE, MAPE, and $R^2$.

4. **Multi-Asset Performance & Correlation Engine**
   - Normalized cumulative percentage return comparison across multiple assets from day zero.
   - Live Pearson correlation matrix heatmap for asset diversification analysis.

---

## 🛠 Tech Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Core implementation |
| **Frontend / UI** | Streamlit | Reactive dashboard with state management |
| **Data Ingestion** | `yfinance` | Free historical and live market API |
| **Data Wrangling** | Pandas, NumPy | Vectorized time-series calculations |
| **Visualization** | Plotly (`plotly.graph_objects`) | Hardware-accelerated interactive candlestick charting |
| **Machine Learning** | Scikit-Learn | Time-series regression, lag features, and evaluation |
| **Testing** | `unittest` | Automated mathematical correctness unit tests |

---

## 🚀 How to Run Locally

### 1. Activate the Virtual Environment
```powershell
.\stock_crypto_dashboard\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies (if not already installed)
```bash
pip install -r stock_crypto_dashboard/requirements.txt
```

### 3. Run the Unit Tests
```powershell
$env:PYTHONPATH="stock_crypto_dashboard"
python -m unittest discover -s stock_crypto_dashboard/tests
```

### 4. Launch the Streamlit Terminal
```powershell
streamlit run stock_crypto_dashboard/app.py
```
The terminal will automatically launch in your browser at `http://localhost:8501`.

---

## 🎓 Interview Talking Points & Architecture

### System Architecture Flow:
```
Yahoo Finance API  --->  Caching Layer (TTL=15m)  --->  Indicator & Feature Engine
                                                                 │
                                ┌────────────────────────────────┴────────────────────────┐
                                ▼                                                         ▼
                     Plotly Visualization Layer                              Machine Learning Pipeline
                     - Interactive Candlesticks                              - Autoregressive Lag Creation
                     - Volume Subplots                                       - Chronological Train/Test Partition
                     - RSI / MACD Oscillators                                - Ridge / Random Forest Regressor
                     - Multi-Asset Performance                               - Confidence Interval Projection
                                │                                                         │
                                └────────────────────────────────┬────────────────────────┘
                                                                 ▼
                                                       Streamlit Web Terminal
```

### Top 3 Interview Questions:
1. **Why avoid random train/test split?**
   * *Answer:* Random shuffling leaks future price points into the past training partition (lookahead bias). Time series models must strictly be trained on chronological historical slices $[0, T]$ and tested on $(T, T+N]$.
2. **Why Ridge Regression for financial lags?**
   * *Answer:* Autoregressive lag features ($t-1, t-2$) are inherently collinear. Ordinary Least Squares (OLS) can produce unstable weights; Ridge's L2 penalty shrinks coefficients and prevents overfitting on noise.
3. **How is Wilder's RSI different from standard SMA RSI?**
   * *Answer:* Wilder's smoothing uses an exponential smoothing factor ($\alpha = 1/N$), retaining historical decay and matching institutional platforms like Bloomberg Terminal and TradingView.

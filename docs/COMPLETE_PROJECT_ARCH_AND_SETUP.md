# FinMetrics: Complete Architecture, Terminal Setup & Conversation Record

> **Project Folder Absolute Path:**  
> `C:\Users\varsh.VARSHI\stock_crypto_dashboard`
> 
> **GitHub Repository:**  
> `https://github.com/varshikakureti842-blip/stockmarket-dashboard`
> 
> **Local App URL:**  
> `http://localhost:8501`

---

## 1. Directory Tree & Project Structure

```
C:\Users\varsh.VARSHI\stock_crypto_dashboard\
│
├── .gitignore                      # Git ignore rules (.venv, __pycache__, logs)
├── README.md                       # High-level overview & quickstart guide
├── requirements.txt                # Frozen Python dependencies
├── app.py                          # Streamlit UI & Master Controller
│
├── src\                            # Core Application Modules
│   ├── __init__.py                 # Package declaration
│   ├── data_loader.py              # Yahoo Finance ingestion, caching & sanitization
│   ├── indicators.py               # Pure quantitative calculations (SMA, EMA, BB, RSI, MACD)
│   ├── model.py                    # Time-series ML forecaster, lag features, Ridge/RF models
│   └── visualizer.py               # Plotly interactive candlestick, indicator & heatmap charts
│
├── tests\                          # Automated Test Suite
│   └── test_indicators.py          # Unit tests verifying indicator math & feature pipelines
│
├── docs\                           # Documentation & Conversation History
│   └── COMPLETE_PROJECT_ARCH_AND_SETUP.md  # This document
│
└── .venv\                          # Python 3.14 Virtual Environment (Ignored by Git)
```

---

## 2. Complete Run & Terminal Setup Architecture

### Local Environment Specifications
* **Operating System:** Windows 11 (64-bit)
* **Python Runtime:** Python 3.14
* **Web Framework:** Streamlit
* **Local Web Server:** Uvicorn / Tornado on `http://localhost:8501`

### Exact Terminal Commands to Run the Application

#### Option A: Quick Launch (From Any PowerShell Prompt)
```powershell
cd C:\Users\varsh.VARSHI\stock_crypto_dashboard
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

#### Option B: Launch Without Activating the Environment
```powershell
C:\Users\varsh.VARSHI\stock_crypto_dashboard\.venv\Scripts\streamlit.exe run C:\Users\varsh.VARSHI\stock_crypto_dashboard\app.py
```

#### Option C: Run Automated Unit Tests
```powershell
cd C:\Users\varsh.VARSHI\stock_crypto_dashboard
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

---

## 3. End-to-End System & Data Flow Architecture

```
                       ┌─────────────────────────────────────┐
                       │  User Browser / Streamlit Frontend  │
                       │       (http://localhost:8501)       │
                       └──────────────────┬──────────────────┘
                                          │
                   User Selects Ticker, Date Range, Indicators, ML Horizon
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │  Caching Layer (@st.cache_data)     │
                       │         (TTL = 900 seconds)         │
                       └──────────┬───────────────────────┬──┘
                                  │                       │
                            Cache Miss                Cache Hit
                                  │                       │
                                  ▼                       ▼
      ┌───────────────────────────────────┐    ┌─────────────────────┐
      │ Yahoo Finance API (via yfinance) │    │ Instant In-Memory   │
      │  - Raw OHLCV Historical Quotes   │    │ Response (Cached)   │
      │  - Asset Fundamentals & Profile   │    └──────────┬──────────┘
      └──────────────────┬────────────────┘               │
                         │                                │
                         └────────────────┬───────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │ Quantitative Indicator Engine       │
                       │ (src/indicators.py)                 │
                       │  - SMA (20, 50, 200)                │
                       │  - EMA (12, 26)                     │
                       │  - Bollinger Bands (20, 2σ)         │
                       │  - Wilder's Smoothed RSI (14)       │
                       │  - MACD (12, 26, 9)                 │
                       └──────────┬───────────────────────┬──┘
                                  │                       │
                                  ▼                       ▼
          ┌──────────────────────────────┐   ┌──────────────────────────────┐
          │ Interactive Chart Engine     │   │ Machine Learning Forecaster  │
          │ (src/visualizer.py)          │   │ (src/model.py)               │
          │  - Candlestick + Volume      │   │  - Autoregressive Lags       │
          │  - Technical Overlays        │   │  - Chronological 80/20 Split │
          │  - RSI / MACD Oscillators    │   │  - Ridge / Random Forest Fit │
          │  - Pearson Corr Heatmap      │   │  - Out-of-sample Projection  │
          └──────────────┬───────────────┘   └──────────────┬───────────────┘
                         │                                  │
                         └────────────────┬─────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │ Streamlit UI Rendering (app.py)     │
                       │  - Tab 1: Market & Technicals       │
                       │  - Tab 2: AI Forecaster             │
                       │  - Tab 3: Multi-Asset Compare       │
                       │  - Tab 4: Interview Dossier         │
                       └─────────────────────────────────────┘
```

---

## 4. Chronological Step-by-Step History of the Conversation

### Step 1: Project Conceptualization & Selection
* **User Request:** Wanted to work on an AI / Python / Data Science project prototype completely free of cost.
* **Options Explored:** 
  1. AI Fitness Rep Counter (OpenCV / MediaPipe)
  2. Resume / Job Description Matcher (NLP)
  3. Document Q&A RAG Chatbot (Gemini / ChromaDB)
  4. Interactive Stock & Crypto Analytics Dashboard (Yahoo Finance / Streamlit / Scikit-Learn)
* **User Decision:** Selected **Option 4** (Stock & Crypto Analytics Dashboard).

### Step 2: Environment Setup & Package Installation
* Discovered local Python runtime: `Python 3.14.5`.
* Created virtual environment: `python -m venv stock_crypto_dashboard\.venv`.
* Installed core dependencies via pip:
  * `streamlit` (UI framework)
  * `yfinance` (Financial data ingestion)
  * `plotly` (Interactive graphing)
  * `pandas` & `numpy` (Vectorized calculations)
  * `scikit-learn` (Time-series predictive regression)

### Step 3: Architecture & Module Engineering
* Engineered clean modular architecture:
  * `src/data_loader.py`: Handles API queries, error checking, and TTL caching.
  * `src/indicators.py`: Implements mathematical formulas for SMA, EMA, Bollinger Bands, Wilder's RSI, and MACD.
  * `src/model.py`: Generates autoregressive lag features ($t-1 \dots t-10$), implements strict chronological train/test splitting (zero lookahead bias), trains Ridge / Random Forest models, and computes MAE, RMSE, and MAPE.
  * `src/visualizer.py`: Generates Plotly candlestick charts with volume bars, oscillator panels, forecast ribbons, and multi-asset comparisons.
  * `app.py`: Glues the components into a responsive 4-tab Streamlit web application.
  * `tests/test_indicators.py`: Created 6 automated unit tests with standard `unittest` (all 6 tests passing).

### Step 4: Troubleshooting & Polishing
* **Issue 1 (`ModuleNotFoundError: No module named 'matplotlib'`):**
  * *Root Cause:* Pandas `Styler.background_gradient()` internally calls matplotlib colormaps.
  * *Fix:* Installed `matplotlib` in the virtual environment and created a native Plotly `Heatmap` in `visualizer.py` for a vastly superior dark-theme experience.
* **Issue 2 (Streamlit Module Reload Delay):**
  * *Fix:* Restarted the Streamlit server process cleanly to reload freshly exported functions.

### Step 5: Git Installation & Elevation Resolution
* Discovered `git` was not pre-installed on the machine.
* Attempted automated `winget` installation, which paused due to Windows UAC elevation requirements.
* Terminated background locked installer processes (`taskkill` / `Stop-Process`).
* Directed user to run Administrator Command Prompt:
  ```cmd
  cd /d C:\Users\varsh.VARSHI\stock_crypto_dashboard && git push -u origin main --force
  ```
* User successfully executed elevation, authenticated with GitHub, and pushed the entire codebase to:
  `https://github.com/varshikakureti842-blip/stockmarket-dashboard`.

### Step 6: Streamlit Community Cloud Deployment Setup
* Connected local Git repository tracking to `origin/main`.
* Provided direct 1-click cloud deployment pipeline via:
  `https://share.streamlit.io/deploy?repository=varshikakureti842-blip/stockmarket-dashboard&branch=main&mainModule=app.py`.
* Guided the user on the 1-click GitHub authorization for free hosting.

---

## 5. Technical Interview Cheat Sheet (Talking Points)

1. **Why avoid random train/test split?**
   * *Answer:* Shuffling time-series data causes lookahead bias (data leakage). Future prices leak into the past. We strictly enforce chronological partitioning (first 80% train, last 20% test).
2. **Why Ridge Regression for financial lag features?**
   * *Answer:* Autoregressive lag features are collinear ($t-1$ correlates at ~0.98 with $t-2$). Ordinary Least Squares (OLS) produces unstable weights. Ridge regression adds an L2 regularization penalty ($\lambda \sum w_i^2$) to stabilize coefficients and prevent overfitting.
3. **How are API rate limits handled?**
   * *Answer:* In-memory caching via `@st.cache_data(ttl=900)` stores retrieved quotes for 15 minutes, allowing instantaneous widget re-renders without external HTTP overhead.
4. **Wilder's RSI vs. Standard SMA RSI:**
   * *Answer:* Wilder's smoothing employs exponential decay ($\alpha = 1/14$), matching the institutional formulation used by Bloomberg and TradingView.

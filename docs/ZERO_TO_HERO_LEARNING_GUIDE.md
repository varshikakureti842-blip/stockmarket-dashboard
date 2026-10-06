# The Zero-to-Hero Learning Guide: FinMetrics AI Terminal

> **Purpose:** This guide explains **every single term, concept, formula, and tool** used in this project in plain, simple English. Even if you had zero prior knowledge in finance or machine learning, studying this guide will give you the confidence to explain and discuss this project fluently in any technical interview or presentation.

---

# Table of Contents
1. [The 2-Minute Project Story (How to explain it without being nervous)](#1-the-2-minute-project-story)
2. [Domain 1: Financial & Stock Market Concepts](#2-domain-1-financial--stock-market-concepts)
3. [Domain 2: Machine Learning & Time-Series Concepts](#3-domain-2-machine-learning--time-series-concepts)
4. [Domain 3: Python & Data Engineering Stack](#4-domain-3-python--data-engineering-stack)
5. [Domain 4: Interactive Web & Visualization Stack](#5-domain-4-interactive-web--visualization-stack)
6. [Domain 5: Git, Version Control & Cloud Deployment](#6-domain-5-git-version-control--cloud-deployment)
7. [Domain 6: Flashcard Glossary (Quick Recall)](#7-domain-6-flashcard-glossary-quick-recall)

---

# 1. The 2-Minute Project Story

When an interviewer or colleague asks: *"Tell me about your project"*, here is your structured, stress-free script:

> **"I built a real-time web application called FinMetrics that combines quantitative finance with machine learning.**
>
> **The Problem:** Most people either pay thousands of dollars for Bloomberg or TradingView, or build ML models that make a rookie mistake: shuffling time-series data, which cheats by peeking into the future.
>
> **What My App Does:**
> 1. It pulls live, free market data for stocks and cryptos using the Yahoo Finance API.
> 2. It calculates key mathematical indicators like Wilder's RSI, MACD, and Bollinger Bands using Pandas and NumPy.
> 3. It trains a time-series machine learning model (Ridge Regression / Random Forest) using historical lag features without any future data leakage to project price trends for the next 7 to 30 days.
> 4. Everything is presented in an interactive dark-mode dashboard built with Streamlit and Plotly, deployed live on Streamlit Community Cloud."*

---

# 2. Domain 1: Financial & Stock Market Concepts

### 2.1 Ticker Symbol
* **Plain English:** A short nickname or abbreviation used to uniquely identify a publicly traded asset on an exchange.
* **In Our Project:** `AAPL` (Apple), `NVDA` (Nvidia), `BTC-USD` (Bitcoin in US Dollars), `RELIANCE.NS` (Reliance on the National Stock Exchange of India).

### 2.2 OHLCV (Open, High, Low, Close, Volume)
* **Plain English:** The 5 core numbers recorded for every single trading day:
  * **Open:** The price when the market opened in the morning.
  * **High:** The highest price reached during the day.
  * **Low:** The lowest price reached during the day.
  * **Close:** The final price when the market closed.
  * **Volume:** The total number of shares or coins traded that day.
* **In Our Project:** Stored in a Pandas DataFrame and used to draw candlesticks and compute indicators.

### 2.3 Candlestick Chart
* **Plain English:** A visual representation of OHLCV:
  * **Green (Bullish) Candle:** Close was higher than Open (price went UP).
  * **Red (Bearish) Candle:** Close was lower than Open (price went DOWN).
  * **Wicks (Shadows):** The thin lines sticking out of the top and bottom showing the High and Low extremes of that day.

### 2.4 Market Capitalization (Market Cap)
* **Plain English:** The total value of all of a company's shares:
  $$\text{Market Cap} = \text{Current Stock Price} \times \text{Total Shares Outstanding}$$
* **In Our Project:** Displayed in the top KPI card (e.g., Apple $\approx \$3.5\text{ Trillion}$).

### 2.5 P/E Ratio (Price-to-Earnings Ratio)
* **Plain English:** How much investors are willing to pay for $\$1$ of the company's annual profit. A high P/E (e.g., 50+) means investors expect high future growth; a low P/E (e.g., 15) means it is considered a value stock.

### 2.6 Technical Analysis vs. Fundamental Analysis
* **Fundamental Analysis:** Looking at business health—earnings reports, balance sheets, revenue growth, and management.
* **Technical Analysis:** Studying price patterns, trends, volume, and mathematical momentum indicators to forecast future price movements.
* **In Our Project:** We focus on Technical Analysis and quantitative indicators.

### 2.7 Moving Averages: SMA vs. EMA
* **Simple Moving Average (SMA):** The regular average price over the last $N$ days.
  * *Example:* 20-day SMA adds the last 20 closing prices and divides by 20.
* **Exponential Moving Average (EMA):** An average that gives **more mathematical weight to recent days**, so it reacts faster to sudden market changes.
* **Trading Insight:** If a short-term average (SMA 20) crosses **above** a long-term average (SMA 50), it is a bullish signal (uptrend). If it crosses below, it is a bearish signal (downtrend).

### 2.8 Bollinger Bands
* **Plain English:** A 20-day moving average in the middle, surrounded by an upper band and a lower band.
* **How It's Calculated:** 
  $$\text{Upper Band} = \text{SMA}_{20} + (2 \times \text{Standard Deviation})$$
  $$\text{Lower Band} = \text{SMA}_{20} - (2 \times \text{Standard Deviation})$$
* **Why It Matters:** Statistically, $\sim 95\%$ of price movements stay inside these bands. If the price touches the Upper Band, the asset is stretched high; if it touches the Lower Band, it is stretched low. When the bands pinch together (squeeze), a huge breakout is often imminent.

### 2.9 Relative Strength Index (RSI)
* **Plain English:** A momentum speed gauge that scores price strength on a scale from **0 to 100**.
* **The 70 / 30 Rule:**
  * **$\ge 70$ (Overbought):** Buyers pushed the price up too fast; a pullback or drop is likely.
  * **$\le 30$ (Oversold):** Sellers dumped the price too aggressively; a bounce or recovery is likely.
* **Wilder's Smoothing:** In `src/indicators.py`, we implement Wilder's exponential smoothing factor ($\alpha = 1/14$), which matches institutional platforms like Bloomberg Terminal.

### 2.10 MACD (Moving Average Convergence Divergence)
* **Plain English:** Measures how fast a short-term trend is pulling away from a long-term trend.
* **Three Components:**
  1. **MACD Line:** $\text{EMA}_{12} - \text{EMA}_{26}$.
  2. **Signal Line:** 9-day EMA of the MACD Line.
  3. **Histogram:** The difference between the MACD line and the Signal line. Green bars above zero mean growing upward momentum; red bars below zero mean growing downward momentum.

### 2.11 Pearson Correlation Matrix
* **Plain English:** A number between **-1.0 and +1.0** measuring how two assets move in relation to each other:
  * **+1.0 (Positive Correlation):** Both assets rise and fall together (e.g., Apple and Microsoft).
  * **0.0 (Uncorrelated):** The movement of one has no relationship to the other.
  * **-1.0 (Inverse Correlation):** When one rises, the other falls (e.g., Stocks vs. Gold during crashes).
* **In Our Project:** Rendered as an interactive colored heatmap in Tab 3 to help investors build diversified portfolios.

---

# 3. Domain 2: Machine Learning & Time-Series Concepts

### 3.1 What is Machine Learning Regression?
* **Classification:** Predicting a label or category (e.g., "Cat" vs. "Dog", "Fraud" vs. "Not Fraud").
* **Regression:** Predicting a continuous numeric value (e.g., tomorrow's stock price = $\$243.50$).
* **In Our Project:** We solve a **time-series regression problem**.

### 3.2 Time-Series vs. Standard Machine Learning
* In normal machine learning (like predicting house prices), each house is an independent data point. Shuffling the rows doesn't hurt.
* In **Time-Series**, **order and sequence matter!** Today's price depends on yesterday's price, which depends on last week's price.

### 3.3 The Lookahead Bias Trap (Data Leakage)
* **Crucial Interview Point:** If you use `train_test_split(shuffle=True)` on stock prices, the model trains on Friday's price to predict Wednesday's price! That is cheating (lookahead bias).
* **Our Solution:** We enforce a **strict chronological split**. We train only on the first 80% of historical dates (the past) and evaluate on the remaining 20% (the unseen future).

### 3.4 Feature Engineering & "Lag Features"
* A machine learning model cannot just take a raw timestamp and predict a price. We have to create smart numerical clues:
  * **Lag 1 ($t-1$):** Yesterday's close price.
  * **Lag 2 ($t-2$):** Price from 2 days ago.
  * **Lag 5 ($t-5$):** Price from 5 days ago (1 trading week).
  * **Rolling Mean 5:** Average price over the past 5 days.
  * **Rolling Volatility 10:** Standard deviation over the past 10 days.

### 3.5 Multicollinearity & Why Ridge Regression?
* **The Problem:** Yesterday's price ($t-1$) and the day before yesterday's price ($t-2$) are almost identical (correlation $> 0.98$). When inputs are copies of each other, standard linear regression gets confused and produces erratic, unstable mathematical weights.
* **The Solution (Ridge L2 Regularization):** Ridge adds a penalty ($\lambda \sum w_i^2$) to the loss function. This shrinks the weights and prevents the model from overreacting to noise or collinearity.

### 3.6 Evaluation Metrics (Explained in Plain Dollars)
* **MAE (Mean Absolute Error):** On average, how many dollars was the prediction off?
  * *Example:* If Apple is $\$230$ and MAE is $\$3.20$, the model is typically within $\pm \$3.20$.
* **RMSE (Root Mean Squared Error):** Similar to MAE, but squares errors before averaging. It heavily penalizes big, disastrous misses.
* **MAPE (Mean Absolute Percentage Error):** The percentage error ($3.2\%$). This allows comparing Bitcoin (priced at $\$90,000$) to Apple (priced at $\$230$) fairly.
* **$R^2$ Score (Coefficient of Determination):** A score from 0 to 1 indicating how much of the price variance is explained by the model.

### 3.7 Autoregressive Multi-Step Forecasting
* How do we forecast 14 days ahead if we only know today's price?
  * **Step 1:** Predict day $t+1$ using known past lags.
  * **Step 2:** Feed that $t+1$ prediction back in as the new "yesterday" to predict day $t+2$.
  * **Step 3:** Repeat for the desired horizon (recursive autoregression).
* **Confidence Interval Cone:** Because errors compound over time, uncertainty expands by $\sqrt{\text{days}}$, forming a shaded 95% confidence cone on the chart.

---

# 4. Domain 3: Python & Data Engineering Stack

### 4.1 Virtual Environment (`.venv`)
* **Plain English:** A private, isolated folder containing specific versions of Python packages for this project.
* **Why it matters:** It prevents library version conflicts with other projects on your computer.

### 4.2 `requirements.txt`
* **Plain English:** A recipe list of all required external Python packages and versions. When deploying to the cloud, the hosting platform reads this file and installs everything automatically.

### 4.3 Pandas (`pd.DataFrame`)
* **Plain English:** Python's version of supercharged Excel spreadsheets in code. Allows instant filtering, math columns, date parsing, and rolling window calculations.

### 4.4 NumPy (`np.ndarray`)
* **Plain English:** Python's fundamental library for high-speed scientific and array mathematics (matrix multiplication, square roots, standard deviations).

### 4.5 `yfinance`
* **Plain English:** An open-source Python library that queries Yahoo Finance's publicly available endpoints to retrieve historical stock splits, dividends, prices, and corporate profiles for free without requiring API keys or credit cards.

### 4.6 In-Memory Caching (`@st.cache_data` & TTL)
* **Plain English:** If a user moves an indicator slider, re-downloading 5 years of stock quotes over the internet takes 3 seconds and might trigger Yahoo's rate limits.
* **TTL (Time-To-Live):** `@st.cache_data(ttl=900)` stores the downloaded data in RAM for 15 minutes. The user gets instant, sub-second responses when interacting with the dashboard.

---

# 5. Domain 4: Interactive Web & Visualization Stack

### 5.1 Streamlit
* **Plain English:** A Python web framework that turns data scripts into interactive web applications without needing HTML, CSS, JavaScript, or React.
* **Reactive Model:** Whenever a user clicks a button or adjusts a slider, Streamlit re-executes the Python script from top to bottom, updating the screen instantly.

### 5.2 Plotly vs. Matplotlib
* **Matplotlib:** Generates static picture files (PNGs). You cannot zoom, hover, or click on them.
* **Plotly:** Generates interactive JavaScript/WebGL charts. You can hover to read exact prices, click to toggle indicators, pan across time periods, and zoom in on specific candlesticks.

---

# 6. Domain 5: Git, Version Control & Cloud Deployment

### 6.1 Git (Local Version Control)
* **Repository (Repo):** A project folder tracked by Git with a history of every change made.
* **Commit:** A permanent snapshot of your project at a specific point in time (like a video game save point).
* **Branch (`main`):** The primary production timeline of your code.
* **Remote (`origin`):** The cloud destination where the code is backed up (GitHub).

### 6.2 Administrator Elevation & UAC
* **What Happened:** When installing Git via `winget`, Windows displayed a User Account Control (UAC) prompt asking for Administrator permission. Running `cmd` as Administrator allowed Windows to install Git into `C:\Program Files\Git`.

### 6.3 Streamlit Community Cloud (Hosting / PaaS)
* **Platform-as-a-Service (PaaS):** A free hosting platform provided by Snowflake. It links directly to your GitHub repository:
  1. Detects changes pushed to GitHub.
  2. Spawns a cloud virtual machine.
  3. Installs packages from `requirements.txt`.
  4. Runs `streamlit run app.py` and provides a permanent public URL accessible worldwide!

---

# 7. Domain 6: Flashcard Glossary (Quick Recall)

| Term | In 5 Words or Less |
| :--- | :--- |
| **Ticker** | Asset's stock market abbreviation. |
| **OHLCV** | Open, High, Low, Close, Volume. |
| **SMA** | Simple rolling arithmetic mean. |
| **EMA** | Exponential weighted moving average. |
| **Bollinger Bands** | Volatility envelope around 20-day SMA. |
| **RSI** | 0-100 speed & momentum oscillator. |
| **MACD** | Difference between fast/slow EMAs. |
| **Overbought** | Price pushed up too fast ($\ge 70$). |
| **Oversold** | Price dumped down too fast ($\le 30$). |
| **Lookahead Bias** | Cheating by using future data. |
| **Lag Feature** | Historical prices used as inputs. |
| **Ridge Regression** | L2-regularized stable linear model. |
| **MAE** | Average dollar prediction error. |
| **RMSE** | Error metric heavily penalizing outliers. |
| **MAPE** | Normalized percentage prediction error. |
| **Streamlit** | Pure Python interactive web framework. |
| **Plotly** | Hardware-accelerated interactive charting. |
| **TTL Cache** | Storing API data in RAM. |

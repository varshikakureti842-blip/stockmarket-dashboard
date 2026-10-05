"""Time-Series Machine Learning Forecasting Engine
Implements lag feature engineering, chronological train/test splitting,
model training (Ridge / Random Forest), performance evaluation, and multi-step future forecasting.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_time_series_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """Create lag features, rolling statistics, and calendar attributes without lookahead bias.

    Args:
        df: DataFrame containing at least ['Date', 'Close'].
        lags: List of day offsets to compute lag features for.

    Returns:
        DataFrame with engineered features and NaN rows dropped.
    """
    feat_df = df.copy()
    feat_df.sort_values("Date", ascending=True, inplace=True)
    feat_df.reset_index(drop=True, inplace=True)

    # 1. Autoregressive Lag Features
    for lag in lags:
        feat_df[f"lag_{lag}"] = feat_df["Close"].shift(lag)

    # 2. Rolling window features (closed='left' to prevent current day leakage)
    feat_df["rolling_mean_5"] = feat_df["Close"].shift(1).rolling(5).mean()
    feat_df["rolling_mean_10"] = feat_df["Close"].shift(1).rolling(10).mean()
    feat_df["rolling_std_10"] = feat_df["Close"].shift(1).rolling(10).std()

    # 3. Calendar temporal features
    feat_df["day_of_week"] = feat_df["Date"].dt.dayofweek
    feat_df["month"] = feat_df["Date"].dt.month

    # Drop early rows containing NaNs created by lag and rolling operations
    feat_df.dropna(inplace=True)
    feat_df.reset_index(drop=True, inplace=True)
    return feat_df


def train_and_evaluate_forecaster(
    df: pd.DataFrame,
    model_type: str = "Ridge",
    train_ratio: float = 0.80,
    forecast_horizon: int = 14
) -> Tuple[Optional[Dict], Optional[str]]:
    """Train time-series model with chronological split and project future horizon days.

    Args:
        df: Historical DataFrame with ['Date', 'Close'].
        model_type: 'Ridge' or 'Random Forest'.
        train_ratio: Fraction of historical data used for training (chronological).
        forecast_horizon: Number of days to forecast into the future.

    Returns:
        Tuple of (results_dict, error_message).
    """
    if len(df) < 40:
        return None, "Insufficient data points for machine learning (need at least 40 records)."

    feat_df = create_time_series_features(df)
    if len(feat_df) < 20:
        return None, "Not enough data points after lag feature creation."

    feature_cols = [c for c in feat_df.columns if c.startswith("lag_") or c.startswith("rolling_") or c in ["day_of_week", "month"]]
    target_col = "Close"

    X = feat_df[feature_cols].values
    y = feat_df[target_col].values
    dates = feat_df["Date"].values

    # STRICT CHRONOLOGICAL SPLIT (Avoids data leakage / lookahead bias)
    split_idx = int(len(feat_df) * train_ratio)
    if split_idx < 10 or (len(feat_df) - split_idx) < 5:
        return None, "Train/test split resulted in too few samples in train or test partition."

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]

    # Select and fit estimator
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    else:
        # Default: Ridge Regression with L2 Regularization
        model = Ridge(alpha=1.0)

    model.fit(X_train, y_train)

    # In-sample & Out-of-sample predictions
    test_preds = model.predict(X_test)

    # Calculate Evaluation Metrics
    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    r2 = r2_score(y_test, test_preds)
    mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100.0

    # Test residual standard deviation for uncertainty estimation
    residuals = y_test - test_preds
    residual_std = np.std(residuals)

    # 4. Multi-step Autoregressive Forecasting for the Future Horizon
    last_known_row = feat_df.iloc[-1].copy()
    current_close = float(last_known_row["Close"])
    recent_prices = list(feat_df["Close"].iloc[-15:].values)

    future_dates = []
    future_preds = []
    last_date = pd.to_datetime(feat_df["Date"].iloc[-1])

    # Determine date increment (check if crypto / 7 days vs stock / 5 days)
    is_crypto = (df["Date"].diff().dt.days == 1).mean() > 0.85

    curr_date = last_date
    for step in range(1, forecast_horizon + 1):
        if is_crypto:
            curr_date += pd.Timedelta(days=1)
        else:
            # Skip weekends for traditional stock markets
            curr_date += pd.Timedelta(days=1)
            while curr_date.weekday() >= 5:
                curr_date += pd.Timedelta(days=1)

        future_dates.append(curr_date)

        # Assemble step feature vector
        # lags: [1, 2, 3, 5, 10]
        lag_1 = recent_prices[-1]
        lag_2 = recent_prices[-2] if len(recent_prices) >= 2 else lag_1
        lag_3 = recent_prices[-3] if len(recent_prices) >= 3 else lag_2
        lag_5 = recent_prices[-5] if len(recent_prices) >= 5 else lag_3
        lag_10 = recent_prices[-10] if len(recent_prices) >= 10 else lag_5

        roll_5 = np.mean(recent_prices[-5:])
        roll_10 = np.mean(recent_prices[-10:])
        roll_std = np.std(recent_prices[-10:])
        dow = curr_date.dayofweek
        mo = curr_date.month

        step_features = np.array([[lag_1, lag_2, lag_3, lag_5, lag_10, roll_5, roll_10, roll_std, dow, mo]])
        pred_val = float(model.predict(step_features)[0])

        future_preds.append(pred_val)
        recent_prices.append(pred_val)

    # Uncertainty bounds expand with sqrt of time step (Random walk variance propagation)
    step_scales = np.sqrt(np.arange(1, forecast_horizon + 1))
    upper_bound = np.array(future_preds) + (1.96 * residual_std * step_scales)
    lower_bound = np.array(future_preds) - (1.96 * residual_std * step_scales)

    results = {
        "model_name": model_type,
        "feature_names": feature_cols,
        "metrics": {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 4),
            "MAPE": round(float(mape), 2),
        },
        "test_dates": pd.to_datetime(dates_test),
        "y_test_actual": y_test,
        "y_test_pred": test_preds,
        "future_dates": future_dates,
        "future_preds": np.array(future_preds),
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
    }

    return results, None

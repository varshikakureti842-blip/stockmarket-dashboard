"""Unit tests for Technical Indicators and Feature Engineering Modules."""

import unittest
import numpy as np
import pandas as pd
from src.indicators import (
    compute_sma,
    compute_ema,
    compute_bollinger_bands,
    compute_rsi,
    compute_macd,
    enrich_with_all_indicators,
)
from src.model import create_time_series_features, train_and_evaluate_forecaster


class TestIndicators(unittest.TestCase):
    def setUp(self):
        # Create a synthetic dataset with known properties
        dates = pd.date_range(start="2025-01-01", periods=60, freq="D")
        prices = 100.0 + np.sin(np.linspace(0, 3.14 * 4, 60)) * 10.0 + np.linspace(0, 20, 60)
        self.df = pd.DataFrame({
            "Date": dates,
            "Open": prices - 0.5,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": np.random.randint(100000, 500000, size=60),
        })

    def test_sma_calculation(self):
        enriched = compute_sma(self.df, windows=[5, 20])
        self.assertIn("SMA_5", enriched.columns)
        self.assertIn("SMA_20", enriched.columns)
        # Check that 5-period SMA matches manual rolling mean
        expected_5 = self.df["Close"].rolling(5, min_periods=1).mean()
        np.testing.assert_allclose(enriched["SMA_5"].values, expected_5.values)

    def test_bollinger_bands(self):
        enriched = compute_bollinger_bands(self.df, window=20, num_std=2.0)
        self.assertIn("BB_Upper_20", enriched.columns)
        self.assertIn("BB_Lower_20", enriched.columns)
        # Upper band must be >= Lower band
        valid_rows = enriched.dropna(subset=["BB_Upper_20", "BB_Lower_20"])
        self.assertTrue((valid_rows["BB_Upper_20"] >= valid_rows["BB_Lower_20"]).all())

    def test_rsi_bounds(self):
        enriched = compute_rsi(self.df, window=14)
        self.assertIn("RSI_14", enriched.columns)
        rsi = enriched["RSI_14"]
        # RSI must be strictly bounded between 0 and 100
        self.assertTrue((rsi >= 0.0).all() and (rsi <= 100.0).all())

    def test_macd_columns(self):
        enriched = compute_macd(self.df, fast=12, slow=26, signal=9)
        self.assertIn("MACD_Line", enriched.columns)
        self.assertIn("MACD_Signal", enriched.columns)
        self.assertIn("MACD_Hist", enriched.columns)
        # Histogram must equal Line - Signal
        diff = enriched["MACD_Line"] - enriched["MACD_Signal"]
        np.testing.assert_allclose(enriched["MACD_Hist"].values, diff.values)

    def test_feature_engineering_no_leakage(self):
        feat_df = create_time_series_features(self.df, lags=[1, 2, 5])
        self.assertIn("lag_1", feat_df.columns)
        self.assertIn("lag_5", feat_df.columns)
        self.assertIn("rolling_mean_5", feat_df.columns)
        # Verify lag 1 is shifted by exactly 1 position
        first_row_date = feat_df["Date"].iloc[0]
        orig_row_idx = self.df[self.df["Date"] == first_row_date].index[0]
        self.assertAlmostEqual(feat_df["lag_1"].iloc[0], self.df["Close"].iloc[orig_row_idx - 1])

    def test_model_training_and_forecast(self):
        results, err = train_and_evaluate_forecaster(self.df, model_type="Ridge", forecast_horizon=7)
        self.assertIsNone(err)
        self.assertIsNotNone(results)
        self.assertEqual(len(results["future_preds"]), 7)
        self.assertIn("MAE", results["metrics"])
        self.assertGreater(results["metrics"]["MAE"], 0)


if __name__ == "__main__":
    unittest.main()

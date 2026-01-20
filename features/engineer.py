"""
Feature engineering module.
Computes technical indicators for ML model training and prediction.

IMPORTANT: Feature logic MUST be identical for training and prediction.
This module ensures consistency by using the same functions for both cases.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging

import sys
sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])
from config import (
    SMA_PERIODS, 
    RSI_PERIOD, 
    VOLATILITY_WINDOW,
    HORIZONS,
    TARGET_THRESHOLDS,
    ATR_MULTIPLIERS,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Computes technical features for stock price prediction.
    
    Features (strictly as specified):
    - N-day returns (dynamic by horizon)
    - SMA 10 / 20 / 50
    - RSI (14)
    - Volatility (20-day)
    - Volume % change
    - Price above/below SMA 50
    """
    
    # Feature names that will be used by the model
    FEATURE_COLUMNS = [
        "return_1d",
        "return_3d", 
        "return_5d",
        "return_10d",
        "sma_10",
        "sma_20",
        "sma_50",
        "price_to_sma_10",
        "price_to_sma_20",
        "price_to_sma_50",
        "above_sma_50",
        "rsi_14",
        "volatility_20d",
        "volume_change_pct",
        "volume_sma_ratio",
        "atr_14",
    ]
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.sma_periods = SMA_PERIODS
        self.rsi_period = RSI_PERIOD
        self.volatility_window = VOLATILITY_WINDOW
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for the given OHLCV data.
        
        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume]
            
        Returns:
            DataFrame with original data plus computed features
        """
        df = df.copy()
        
        logger.info(f"Computing features for {len(df)} rows")
        
        # 1. N-day returns
        df = self._compute_returns(df)
        
        # 2. Simple Moving Averages
        df = self._compute_sma(df)
        
        # 3. Price relative to SMAs
        df = self._compute_price_sma_ratios(df)
        
        # 4. RSI
        df = self._compute_rsi(df)
        
        # 5. Volatility
        df = self._compute_volatility(df)
        
        # 6. Volume features
        df = self._compute_volume_features(df)

        # 7. ATR (Average True Range)
        df = self._compute_atr(df)
        
        logger.info(f"Features computed. Columns: {list(df.columns)}")
        
        return df
    
    def _compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute N-day returns."""
        close = df["Close"]
        
        # 1-day return (percentage)
        df["return_1d"] = close.pct_change() * 100
        
        # 3-day return
        df["return_3d"] = close.pct_change(periods=3) * 100
        
        # 5-day return
        df["return_5d"] = close.pct_change(periods=5) * 100
        
        # 10-day return
        df["return_10d"] = close.pct_change(periods=10) * 100
        
        return df
    
    def _compute_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Simple Moving Averages."""
        close = df["Close"]
        
        for period in self.sma_periods:
            df[f"sma_{period}"] = close.rolling(window=period).mean()
        
        return df
    
    def _compute_price_sma_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price relative to SMAs (as percentage deviation)."""
        close = df["Close"]
        
        for period in self.sma_periods:
            sma_col = f"sma_{period}"
            # Percentage deviation from SMA
            df[f"price_to_sma_{period}"] = ((close - df[sma_col]) / df[sma_col]) * 100
        
        # Binary flag: above SMA 50
        df["above_sma_50"] = (close > df["sma_50"]).astype(int)
        
        return df
    
    def _compute_rsi(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        Compute Relative Strength Index (RSI).
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if period is None:
            period = self.rsi_period
            
        close = df["Close"]
        delta = close.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        
        # Calculate average gains and losses using EMA
        avg_gain = gains.ewm(span=period, adjust=False).mean()
        avg_loss = losses.ewm(span=period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle division by zero
        rsi = rsi.replace([np.inf, -np.inf], np.nan)
        rsi = rsi.fillna(50)  # Neutral RSI when no movement
        
        df[f"rsi_{period}"] = rsi
        
        return df
    
    def _compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling volatility (standard deviation of returns)."""
        returns = df["Close"].pct_change()
        
        # 20-day rolling volatility (annualized)
        df["volatility_20d"] = returns.rolling(window=self.volatility_window).std() * np.sqrt(252) * 100
        
        return df
    
    def _compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features."""
        volume = df["Volume"]
        
        # Volume percentage change
        df["volume_change_pct"] = volume.pct_change() * 100
        
        # Volume relative to 20-day average
        volume_sma = volume.rolling(window=20).mean()
        df["volume_sma_ratio"] = volume / volume_sma
        
        # Handle infinities
        df["volume_change_pct"] = df["volume_change_pct"].replace([np.inf, -np.inf], 0)
        df["volume_sma_ratio"] = df["volume_sma_ratio"].replace([np.inf, -np.inf], 1)
        
        return df
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute Average True Range (ATR)."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR (Wilder's Smoothing)
        # First value is simple mean
        # Subsequent values: ((Prior ATR * (period-1)) + Current TR) / period
        # Pandas ewm(alpha=1/period) is equivalent to Wilder's smoothing
        df[f"atr_{period}"] = tr.ewm(alpha=1/period, adjust=False).mean()
        
        # Also compute ATR percent (ATR / Close) for relative comparison
        df[f"atr_pct_{period}"] = (df[f"atr_{period}"] / close) * 100
        
        return df
    
    def create_labels(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Create target labels for training.
        
        Label = 1 if future return >= threshold for the horizon
        Label = 0 otherwise
        
        Args:
            df: DataFrame with Close prices
            horizon: Number of days to look ahead
            
        Returns:
            DataFrame with 'target' column added
        """
        df = df.copy()
        
        # Use Dynamic Threshold based on ATR
        # Multiplier depends on horizon
        multiplier = ATR_MULTIPLIERS.get(horizon, 0.5 * (horizon ** 0.5))
        
        # We need ATR calculated. It should be there from compute_features
        # Ensure 'atr_14' exists. If not, compute it temporarily.
        if "atr_14" not in df.columns:
            logger.warning("ATR not found in dataframe. Computing it now for labels.")
            df = self._compute_atr(df)
            
        # Dynamic Threshold = Multiplier * ATR%
        # Note: creates a Series of thresholds, one for each day
        # We use ATR% (atr_pct_14) because future_return is in %
        threshold_series = df["atr_pct_14"] * multiplier
        
        # Fallback to fixed threshold if nominal ATR is too low (e.g. extremely low volatility)
        # Minimum threshold of 0.5% to avoid noise
        threshold_series = threshold_series.clip(lower=0.5)
        
        # Calculate future return (looking ahead)
        future_close = df["Close"].shift(-horizon)
        future_return = ((future_close - df["Close"]) / df["Close"]) * 100
        
        # Create binary label
        # Compare return vs dynamic threshold
        df["target"] = (future_return >= threshold_series).astype(int)
        
        # Store metadata for analysis
        df["future_return"] = future_return
        df["dynamic_threshold"] = threshold_series
        
        positive_ratio = df['target'].mean()
        logger.info(f"Labels created for {horizon}-day horizon using Dynamic ATR (Multiplier: {multiplier})")
        logger.info(f"Average Threshold: {threshold_series.mean():.2f}%")
        logger.info(f"Positive samples: {df['target'].sum()} / {len(df)} ({positive_ratio:.2%})")
        
        return df
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame, 
        horizon: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for ML training.
        
        Args:
            df: Raw OHLCV DataFrame
            horizon: Prediction horizon in days
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Compute features
        df = self.compute_features(df)
        
        # Create labels
        df = self.create_labels(df, horizon)
        
        # Remove rows with NaN (from rolling calculations and future look-ahead)
        df = df.dropna(subset=self.FEATURE_COLUMNS + ["target"])
        
        # Remove last 'horizon' rows (no valid labels due to look-ahead)
        df = df.iloc[:-horizon] if horizon > 0 else df
        
        X = df[self.FEATURE_COLUMNS]
        y = df["target"]
        
        logger.info(f"Training data prepared: {len(X)} samples, {len(self.FEATURE_COLUMNS)} features")
        
        return X, y
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for prediction (latest row only).
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Tuple of (features for prediction, latest close price)
        """
        # Compute features
        df = self.compute_features(df)
        
        # Get the latest row with valid features
        df_valid = df.dropna(subset=self.FEATURE_COLUMNS)
        
        if df_valid.empty:
            raise ValueError("No valid data after feature computation")
        
        # Get features for prediction (latest row)
        X = df_valid[self.FEATURE_COLUMNS].iloc[[-1]]  # Keep as DataFrame
        latest_close = df_valid["Close"].iloc[-1]
        
        logger.info(f"Prediction data prepared for date: {df_valid.index[-1].date()}")
        
        return X, latest_close
    
    def get_feature_names(self) -> List[str]:
        """Return the list of feature column names."""
        return self.FEATURE_COLUMNS.copy()


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Import dependencies
    sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])
    from data.fetcher import StockDataFetcher
    from data.validator import DataValidator
    
    fetcher = StockDataFetcher()
    validator = DataValidator()
    engineer = FeatureEngineer()
    
    # Fetch and validate data
    df, error = fetcher.fetch("TCS")
    if error:
        print(f"Error: {error}")
    else:
        # Validate
        result = validator.validate(df, "TCS")
        print(result)
        
        # Clean
        df = validator.clean_data(df)
        
        # Compute features
        df_features = engineer.compute_features(df)
        print(f"\nFeatures computed. Shape: {df_features.shape}")
        print(f"\nFeature columns: {engineer.FEATURE_COLUMNS}")
        print(f"\nLatest features:\n{df_features[engineer.FEATURE_COLUMNS].tail(1).T}")
        
        # Prepare training data for 7-day horizon
        X, y = engineer.prepare_training_data(df, horizon=7)
        print(f"\nTraining data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Positive class ratio: {y.mean():.2%}")
        
        # Prepare prediction data
        X_pred, latest_price = engineer.prepare_prediction_data(df)
        print(f"\nPrediction features shape: {X_pred.shape}")
        print(f"Latest close price: ₹{latest_price:.2f}")

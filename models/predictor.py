"""
ML Predictor module.
Handles model loading and prediction for stock price movement.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import joblib
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])
from config import HORIZONS, get_model_path
from features.engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class StockPredictor:
    """
    Stock movement predictor using trained XGBoost models.
    
    Provides predictions in the following format:
    {
        "probability": 0.65,
        "expected_move": 4.0,
        "trend": "bullish"
    }
    
    NOTE: This class does NOT generate text explanations.
    """
    
    def __init__(self):
        """Initialize the predictor and load available models."""
        self.models: Dict[int, object] = {}
        self.feature_engineer = FeatureEngineer()
        self._load_models()
    
    def _load_models(self):
        """Load all available trained models."""
        for horizon in HORIZONS:
            model_path = get_model_path(horizon)
            if model_path.exists():
                try:
                    self.models[horizon] = joblib.load(model_path)
                    logger.info(f"Loaded model for {horizon}-day horizon")
                except Exception as e:
                    logger.warning(f"Failed to load model for {horizon}-day: {e}")
            else:
                logger.warning(f"Model not found for {horizon}-day horizon: {model_path}")
        
        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def is_model_available(self, horizon: int) -> bool:
        """Check if a model is available for the given horizon."""
        return horizon in self.models
    
    def get_available_horizons(self) -> list:
        """Return list of horizons with available models."""
        return list(self.models.keys())
    
    def predict(
        self, 
        df: pd.DataFrame, 
        horizon: int
    ) -> Dict:
        """
        Make a prediction for the given data and horizon.
        
        Args:
            df: OHLCV DataFrame (cleaned and validated)
            horizon: Prediction horizon in days
            
        Returns:
            Dictionary with prediction results:
            {
                "probability": float,  # Probability of positive movement
                "expected_move": float,  # Expected percentage move
                "trend": str,  # "bullish", "bearish", or "neutral"
                "features": dict,  # Key feature values for transparency
            }
        """
        if horizon not in self.models:
            raise ValueError(f"No model available for {horizon}-day horizon. "
                           f"Available: {self.get_available_horizons()}")
        
        # Prepare features
        X, latest_close = self.feature_engineer.prepare_prediction_data(df)
        
        # Get model
        model = self.models[horizon]
        
        # Predict probability
        proba = model.predict_proba(X)[0]
        prob_positive = proba[1]  # Probability of class 1 (positive movement)
        
        # Estimate expected move based on probability and historical data
        expected_move = self._estimate_expected_move(df, horizon, prob_positive)
        
        # Determine trend
        trend = self._classify_trend(prob_positive, expected_move)
        
        # Extract key features for transparency
        key_features = self._extract_key_features(X)
        
        result = {
            "probability": round(prob_positive, 4),
            "expected_move": round(expected_move, 2),
            "trend": trend,
            "latest_close": round(latest_close, 2),
            "features": key_features,
        }
        
        logger.info(f"Prediction for {horizon}-day horizon: {result}")
        
        return result
    
    def _estimate_expected_move(
        self, 
        df: pd.DataFrame, 
        horizon: int, 
        probability: float
    ) -> float:
        """
        Estimate expected percentage move based on historical data and probability.
        
        This is a simple estimation, not a precise prediction.
        """
        # Calculate historical returns for the horizon
        returns = df["Close"].pct_change(periods=horizon) * 100
        returns = returns.dropna()
        
        # Get historical statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Weighted estimate based on probability
        # Higher probability → closer to positive mean + std
        # Lower probability → closer to negative mean - std
        positive_estimate = mean_return + 0.5 * std_return
        negative_estimate = mean_return - 0.5 * std_return
        
        expected_move = probability * positive_estimate + (1 - probability) * negative_estimate
        
        return expected_move
    
    def _classify_trend(self, probability: float, expected_move: float) -> str:
        """Classify the trend based on probability and expected move."""
        if probability >= 0.55 and expected_move > 0:
            return "bullish"
        elif probability <= 0.45 or expected_move < -1:
            return "bearish"
        else:
            return "neutral"
    
    def _extract_key_features(self, X: pd.DataFrame) -> dict:
        """Extract key feature values for transparency."""
        row = X.iloc[0]
        
        return {
            "rsi": round(row["rsi_14"], 2),
            "above_sma_50": bool(row["above_sma_50"]),
            "price_to_sma_50_pct": round(row["price_to_sma_50"], 2),
            "volatility": round(row["volatility_20d"], 2),
            "return_5d": round(row["return_5d"], 2),
            "volume_ratio": round(row["volume_sma_ratio"], 2),
        }
    
    def get_model_info(self, horizon: int) -> Optional[dict]:
        """Get information about a trained model."""
        if horizon not in self.models:
            return None
        
        model = self.models[horizon]
        
        return {
            "horizon": horizon,
            "model_type": type(model).__name__,
            "n_estimators": getattr(model, "n_estimators", None),
            "max_depth": getattr(model, "max_depth", None),
            "feature_names": self.feature_engineer.get_feature_names(),
        }


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    predictor = StockPredictor()
    
    print(f"Available horizons: {predictor.get_available_horizons()}")
    
    if predictor.get_available_horizons():
        # Test with sample data
        sys.path.insert(0, str(__file__).rsplit("\\", 2)[0])
        from data.fetcher import StockDataFetcher
        from data.validator import DataValidator
        
        fetcher = StockDataFetcher()
        validator = DataValidator()
        
        df, error = fetcher.fetch("RELIANCE")
        if not error:
            df = validator.clean_data(df)
            
            for horizon in predictor.get_available_horizons():
                result = predictor.predict(df, horizon)
                print(f"\n{horizon}-day prediction: {result}")
    else:
        print("No models available. Run training first: python -m models.train")

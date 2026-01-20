"""
Model training script.
Trains XGBoost models for each prediction horizon.

Usage:
    python -m models.train              # Train all horizons
    python -m models.train --horizon 7  # Train specific horizon
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging
import argparse
from typing import List, Tuple, Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    HORIZONS, 
    XGBOOST_PARAMS, 
    TARGET_THRESHOLDS,
    get_model_path,
    MODELS_DIR
)
from data.fetcher import StockDataFetcher
from data.validator import DataValidator
from features.engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Training stocks - diversified across sectors
TRAINING_STOCKS = [
    # IT
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM",
    # Banking
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    # Industrial
    "RELIANCE", "LT", "BAJFINANCE", "BAJAJFINSV",
    # Consumer
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "ASIANPAINT",
    # Auto
    "MARUTI", "TATAMOTORS", "M&M", "HEROMOTOCO", "EICHERMOT",
    # Pharma
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB",
    # Metals & Energy
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "ONGC", "NTPC", "POWERGRID",
    # Telecom & Others
    "BHARTIARTL", "TITAN", "ULTRACEMCO", "GRASIM",
]


def fetch_training_data(stocks: List[str]) -> pd.DataFrame:
    """
    Fetch and combine data from multiple stocks for training.
    
    Args:
        stocks: List of stock symbols
        
    Returns:
        Combined DataFrame with data from all valid stocks
    """
    fetcher = StockDataFetcher()
    validator = DataValidator()
    
    all_data = []
    successful = 0
    
    for symbol in stocks:
        logger.info(f"Fetching {symbol}...")
        
        df, error = fetcher.fetch(symbol)
        
        if error:
            logger.warning(f"Failed to fetch {symbol}: {error}")
            continue
        
        # Validate
        result = validator.validate(df, symbol)
        
        if not result.is_valid:
            logger.warning(f"Validation failed for {symbol}: {result.errors}")
            continue
        
        # Clean
        df = validator.clean_data(df)
        
        # Add symbol column for tracking
        df["symbol"] = symbol
        
        all_data.append(df)
        successful += 1
        logger.info(f"Added {symbol}: {len(df)} rows")
    
    if not all_data:
        raise ValueError("No valid data collected from any stock")
    
    combined = pd.concat(all_data, axis=0)
    logger.info(f"Combined data: {len(combined)} rows from {successful} stocks")
    
    return combined


def prepare_dataset(
    df: pd.DataFrame, 
    horizon: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and labels for a specific horizon.
    
    Args:
        df: Combined OHLCV data
        horizon: Prediction horizon
        
    Returns:
        Tuple of (features, labels)
    """
    engineer = FeatureEngineer()
    
    all_X = []
    all_y = []
    
    # Process each stock separately to avoid cross-stock contamination
    for symbol in df["symbol"].unique():
        stock_df = df[df["symbol"] == symbol].drop(columns=["symbol"])
        stock_df = stock_df.sort_index()
        
        try:
            X, y = engineer.prepare_training_data(stock_df, horizon)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            logger.warning(f"Failed to prepare data for {symbol}: {e}")
    
    X = pd.concat(all_X, axis=0)
    y = pd.concat(all_y, axis=0)
    
    logger.info(f"Dataset prepared: {len(X)} samples")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    logger.info(f"Positive ratio: {y.mean():.2%}")
    
    return X, y


def train_model(
    X: pd.DataFrame, 
    y: pd.Series, 
    horizon: int,
    params: dict = None
) -> Tuple[XGBClassifier, Dict]:
    """
    Train XGBoost model with evaluation.
    
    Args:
        X: Feature matrix
        y: Target labels
        horizon: Prediction horizon (for logging)
        params: XGBoost parameters
        
    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    if params is None:
        params = XGBOOST_PARAMS.copy()
    
    # Sort by date index to ensure time order
    X = X.sort_index()
    y = y.sort_index()
    
    # STRICT TIME-AWARE SPLIT
    # We use a fixed date cut-off or percentage based cut-off that respects time
    # Here we use the 80% mark as the cut-off
    
    # Ensure index is datetime
    if not isinstance(X.index, pd.DatetimeIndex):
        X.index = pd.to_datetime(X.index)
        y.index = pd.to_datetime(y.index)
        
    dates = X.index
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    logger.info(f"Time-Series Split Date: {split_date}")
    logger.info(f"Train Range: {X_train.index.min().date()} to {X_train.index.max().date()}")
    logger.info(f"Test Range:  {X_test.index.min().date()} to {X_test.index.max().date()}")
    logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Calculate class weights for imbalanced data
    class_weights = {0: 1.0, 1: len(y_train) / (2 * y_train.sum())}
    
    # Initialize and train model
    model = XGBClassifier(
        **params,
        scale_pos_weight=class_weights[1],
        eval_metric="logloss",
        early_stopping_rounds=20,
    )
    
    # Train with early stopping (using strictly future data for validation)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "horizon": horizon,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "positive_ratio": y.mean(),
        "train_start": str(X_train.index.min().date()),
        "train_end": str(X_train.index.max().date()),
        "test_start": str(X_test.index.min().date()),
        "test_end": str(X_test.index.max().date())
    }
    
    # TimeSeriesSplit for cross-validation instead of random KFold
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = []
    for train_index, val_index in tscv.split(X):
        # Create fold
        X_fold_train, X_fold_val = X.iloc[train_index], X.iloc[val_index]
        y_fold_train, y_fold_val = y.iloc[train_index], y.iloc[val_index]
        
        # Train fold model
        fold_model = XGBClassifier(**params, scale_pos_weight=class_weights[1])
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Score
        score = roc_auc_score(y_fold_val, fold_model.predict_proba(X_fold_val)[:, 1])
        cv_scores.append(score)
        
    metrics["cv_roc_auc_mean"] = np.mean(cv_scores)
    metrics["cv_roc_auc_std"] = np.std(cv_scores)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Model Performance for {horizon}-day horizon:")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} (+/- {metrics['cv_roc_auc_std']:.4f})")
    logger.info(f"{'='*50}\n")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    logger.info("Top 5 Features:")
    for _, row in feature_importance.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, metrics


def save_model(model: XGBClassifier, horizon: int) -> Path:
    """Save trained model to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = get_model_path(horizon)
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved to: {model_path}")
    
    return model_path


def train_all_horizons(stocks: List[str] = None) -> Dict:
    """
    Train models for all horizons.
    
    Args:
        stocks: List of stocks to use for training
        
    Returns:
        Dictionary with training results for each horizon
    """
    if stocks is None:
        stocks = TRAINING_STOCKS
    
    logger.info(f"Starting training for horizons: {HORIZONS}")
    logger.info(f"Using {len(stocks)} stocks for training")
    
    # Fetch all training data once
    logger.info("Fetching training data...")
    combined_df = fetch_training_data(stocks)
    
    results = {}
    
    for horizon in HORIZONS:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Training {horizon}-day horizon model")
        logger.info(f"Target threshold: {TARGET_THRESHOLDS[horizon]}%")
        logger.info(f"{'#'*60}\n")
        
        try:
            # Prepare dataset
            X, y = prepare_dataset(combined_df, horizon)
            
            # Train model
            model, metrics = train_model(X, y, horizon)
            
            # Save model
            model_path = save_model(model, horizon)
            
            results[horizon] = {
                "status": "success",
                "metrics": metrics,
                "model_path": str(model_path),
            }
            
        except Exception as e:
            logger.error(f"Failed to train {horizon}-day model: {e}")
            results[horizon] = {
                "status": "failed",
                "error": str(e),
            }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    for horizon, result in results.items():
        if result["status"] == "success":
            m = result["metrics"]
            logger.info(f"{horizon}-day: ROC-AUC={m['roc_auc']:.4f}, F1={m['f1']:.4f}")
        else:
            logger.info(f"{horizon}-day: FAILED - {result.get('error', 'Unknown error')}")
    
    logger.info("="*60)
    
    return results


def train_single_horizon(horizon: int, stocks: List[str] = None) -> Dict:
    """Train model for a single horizon."""
    if horizon not in HORIZONS:
        raise ValueError(f"Invalid horizon: {horizon}. Must be one of {HORIZONS}")
    
    if stocks is None:
        stocks = TRAINING_STOCKS
    
    logger.info(f"Training {horizon}-day horizon model")
    
    # Fetch data
    combined_df = fetch_training_data(stocks)
    
    # Prepare dataset
    X, y = prepare_dataset(combined_df, horizon)
    
    # Train
    model, metrics = train_model(X, y, horizon)
    
    # Save
    model_path = save_model(model, horizon)
    
    return {
        "status": "success",
        "horizon": horizon,
        "metrics": metrics,
        "model_path": str(model_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock prediction models")
    parser.add_argument(
        "--horizon", 
        type=int, 
        choices=HORIZONS,
        help=f"Train specific horizon only. Options: {HORIZONS}"
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        help="List of stocks to train on (default: predefined list)"
    )
    
    args = parser.parse_args()
    
    if args.horizon:
        result = train_single_horizon(args.horizon, args.stocks)
        print(f"\nResult: {result}")
    else:
        results = train_all_horizons(args.stocks)
        print(f"\nResults: {results}")

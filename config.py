"""
Central configuration for the Stock Decision Assistant.
All constants and settings are defined here for consistency across modules.
"""

from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Data Configuration
MARKET = "NSE"  # National Stock Exchange of India
DATA_SOURCE = "yfinance"
MIN_HISTORY_YEARS = 5
MIN_TRADING_DAYS = 1000  # Approximately 5 years of trading days

# Feature Engineering Constants
SMA_PERIODS = [10, 20, 50]
RSI_PERIOD = 14
VOLATILITY_WINDOW = 20

# ML Configuration
HORIZONS = [3, 5, 7, 10, 15]  # Holding periods in days
DEFAULT_HORIZON = 7

# Target thresholds for labeling (horizon-scaled)
# DEPRECATED in favor of Dynamic ATR-based thresholds
# These are kept for reference or fallback
TARGET_THRESHOLDS = {
    3: 1.5,
    5: 2.5,
    7: 3.5,
    10: 5.0,
    15: 7.0,
}

# Dynamic Labeling Configuration
# Target = ATR(14) * Multiplier
ATR_MULTIPLIERS = {
    3: 1.0,   # ~1.73 sigma random walk -> 1.0 is achievable trend
    5: 1.5,
    7: 2.0,   # ~2.64 sigma -> 2.0 requires trend
    10: 2.5,
    15: 3.0,
}

# XGBoost Default Parameters
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

# Risk Management Rules
RISK_CONFIG = {
    "min_probability": 0.55,      # Below this → AVOID
    "max_risk_per_trade": 0.015,  # 1.5% max risk
    "default_stop_loss_pct": 0.02,  # 2% stop loss
    "target_multipliers": {        # Target as multiple of stop-loss
        3: 1.5,
        5: 2.0,
        7: 2.5,
        10: 3.0,
        15: 3.5,
    }
}

# Data Validation Thresholds
VALIDATION_CONFIG = {
    "min_avg_volume": 100000,     # Minimum average daily volume
    "max_gap_pct": 20,            # Maximum allowed single-day gap
    "min_price": 10,              # Minimum stock price
    "max_missing_pct": 5,         # Maximum percentage of missing data
}

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Model file naming convention
def get_model_filename(horizon: int) -> str:
    """Returns the model filename for a given horizon."""
    return f"xgb_{horizon}d.pkl"

def get_model_path(horizon: int) -> Path:
    """Returns the full path to the model file for a given horizon."""
    return MODELS_DIR / get_model_filename(horizon)

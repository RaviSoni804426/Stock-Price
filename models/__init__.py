"""
ML models module for training and prediction.
"""

# Lazy imports to avoid import errors when dependencies aren't installed
def __getattr__(name):
    if name == "StockPredictor":
        from .predictor import StockPredictor
        return StockPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["StockPredictor"]

"""
Data module for fetching and validating stock data.
"""

# Lazy imports to avoid import errors when dependencies aren't installed
def __getattr__(name):
    if name == "StockDataFetcher":
        from .fetcher import StockDataFetcher
        return StockDataFetcher
    if name == "DataValidator":
        from .validator import DataValidator
        return DataValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["StockDataFetcher", "DataValidator"]

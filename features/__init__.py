"""
Feature engineering module for computing technical indicators.
"""

# Lazy imports to avoid import errors when dependencies aren't installed
def __getattr__(name):
    if name == "FeatureEngineer":
        from .engineer import FeatureEngineer
        return FeatureEngineer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["FeatureEngineer"]

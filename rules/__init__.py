"""
Risk and decision rules engine.
"""

# Lazy imports to avoid import errors when dependencies aren't installed
def __getattr__(name):
    if name == "RulesEngine":
        from .engine import RulesEngine
        return RulesEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["RulesEngine"]

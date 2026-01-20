"""
LLM explanation layer for generating human-readable investment plans.
"""

# Lazy imports to avoid import errors when dependencies aren't installed
def __getattr__(name):
    if name == "PlanExplainer":
        from .explainer import PlanExplainer
        return PlanExplainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["PlanExplainer"]

"""Compatibility alias; implementation lives in src.evaluation.paper_metrics."""
import importlib
import sys

_impl = importlib.import_module("src.evaluation.paper_metrics")
# Also support loaders that keep the original module object after exec_module.
globals().update({key: value for key, value in vars(_impl).items() if not key.startswith("__")})
sys.modules[__name__] = _impl

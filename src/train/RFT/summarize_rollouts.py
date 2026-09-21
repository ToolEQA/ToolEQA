"""Compatibility alias; implementation lives in src.evaluation.summarize_rollouts."""
import importlib
import sys

_impl = importlib.import_module("src.evaluation.summarize_rollouts")
# Also support loaders that keep the original module object after exec_module.
globals().update({key: value for key, value in vars(_impl).items() if not key.startswith("__")})
if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl

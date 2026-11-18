import importlib

from .path import Path
from .metrics import Metrics
from .counter import Counter
from .checkpoint import Checkpointer


def __getattr__(name):
    if name in ("Logger", "logger"):
        logger_module = importlib.import_module(f"{__name__}.logger")
        globals()["logger"] = logger_module
        globals()["Logger"] = logger_module.Logger
        if name == "Logger":
            return logger_module.Logger
        return logger_module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""TREMOR: event-to-exposure risk engine."""
import os

if os.environ.get("TREMOR_LIGHT") != "1":
    try:  # torch must load before pandas/scikit-learn on some Windows builds (DLL initialisation order)
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - torch is optional; the engine falls back to light models
        pass

__version__ = "1.0.0"

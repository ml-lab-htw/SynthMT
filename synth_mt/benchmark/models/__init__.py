# Only expose lightweight, always-available model classes at the top level.
# Advanced models (e.g., AnyStar, StarDist, etc.) are NOT imported here to avoid optional dependency errors.
# Import them directly from their submodules if needed.

from .base import BaseModel
from .anchor_point_model import AnchorPointModel
from .factory import ModelFactory, setup_model_factory

__all__ = [
    "BaseModel",
    "AnchorPointModel",
    "ModelFactory",
    "setup_model_factory"
]

# Example for advanced usage:
# from synth_mt.benchmark.models.anystar import AnyStar

# Expose main config API
from .base import BaseConfig
from .album_config import AlbumentationsConfig
from .spots import SpotConfig, SpotTuningConfig
from .synthetic_data import SyntheticDataConfig
from .tuning import TuningConfig

__all__ = [
    "BaseConfig",
    "AlbumentationsConfig",
    "SpotConfig",
    "SpotTuningConfig",
    "SyntheticDataConfig",
    "TuningConfig"
]

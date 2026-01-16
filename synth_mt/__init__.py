"""
SynthMT: Top-level API

This module exposes the most relevant classes and functions for data generation, benchmarking, configuration, and plotting.

Note: To avoid import-time errors from optional heavy dependencies (e.g., TensorFlow for StarDist/AnyStar), only lightweight and always-available components are imported here. For advanced models or features, import directly from their submodules when needed.
"""

# Data generation
from .data_generation.microtubule import Microtubule, MicrotubuleState
from .data_generation.spots import SpotGenerator
from .data_generation.video import draw_mt, render_frame, generate_frames, generate_video

# Benchmarking (core only, no models)
from .benchmark.dataset import BenchmarkDataset
from .benchmark.metrics import (
    fit_parametric_curve, eval_parametric_curve, as_instance_stack, anchor_points_to_instance_masks
)

# Configs
from .config.base import BaseConfig
from .config.album_config import AlbumentationsConfig
from .config.spots import SpotConfig, SpotTuningConfig
from .config.synthetic_data import SyntheticDataConfig
from .config.tuning import TuningConfig

# File IO
from .file_io.utils import fiji_auto_contrast, fiji_auto_contrast_brightness, process_tiff_video, extract_frames, process_avi_video
from .file_io.writers import OutputManager, sort_instance_mask, merge_instance_mask, export_full_tiff_video_maks

# Plotting (core only)
from .plotting.plotting import show_frame

# Utilities
from .utils.postprocessing import get_area_length_ranges, get_instance_properties, filter_instance_masks, filter_anchor_points
from .utils.preprocessing import process_image
from .utils.rle import mask2rle, rle2mask
from .utils.logger import setup_logging
from .utils.matlab import MatlabEngine, matlab_engine

__all__ = [
    # Data generation
    "Microtubule", "MicrotubuleState", "SpotGenerator", "draw_mt", "render_frame", "generate_frames", "generate_video",
    # Benchmarking
    "BenchmarkDataset", "fit_parametric_curve", "eval_parametric_curve", "as_instance_stack", "anchor_points_to_instance_masks",
    # Configs
    "BaseConfig", "AlbumentationsConfig", "SpotConfig", "SpotTuningConfig", "SyntheticDataConfig", "TuningConfig",
    # File IO
    "fiji_auto_contrast", "fiji_auto_contrast_brightness", "process_tiff_video", "extract_frames", "process_avi_video",
    "OutputManager", "sort_instance_mask", "merge_instance_mask", "export_full_tiff_video_maks",
    # Plotting
    "show_frame",
    # Utilities
    "get_area_length_ranges", "get_instance_properties", "filter_instance_masks", "filter_anchor_points", "process_image", "mask2rle", "rle2mask", "setup_logging", "MatlabEngine", "matlab_engine"
]

# Advanced models and features (e.g., AnyStar, StarDist, etc.) are NOT imported here to avoid optional dependency errors.
# Import them directly from their submodules if needed, e.g.:
# from synth_mt.benchmark.models.anystar import AnyStar

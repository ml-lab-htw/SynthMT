from abc import ABC, abstractmethod

import numpy as np
import os
import torch
import logging

from synth_mt.config.base import BaseConfig

logger = logging.getLogger(__name__)


class BaseModel(BaseConfig, ABC):
    """Abstract base class for a segmentation model."""

    def __init__(
        self,
        model_name: str,
        save_dir: str = ".models",  # save_dir: str | None = None,?
        grayscale: bool = False,
        sharpen_radius: float = 0.0,
        smooth_radius: float = 0.0,
        percentile_min: float = 0.0,
        percentile_max: float = 100.0,
        clip_to_percentiles: bool = False,
        rescale_using_percentiles: bool = False,
        invert: bool = False,
        histogram_normalization: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self._save_dir = self.ensure_save_dir(save_dir)
        self.grayscale = grayscale
        self.sharpen_radius = sharpen_radius
        self.smooth_radius = smooth_radius
        self.percentile_min = percentile_min
        self.percentile_max = percentile_max
        self.clip_to_percentiles = clip_to_percentiles
        self.rescale_using_percentiles = rescale_using_percentiles
        self.invert = invert
        self.histogram_normalization = histogram_normalization
        self._device = self.get_device()
        # Inform about device usage for this model
        logger.debug(f"Using device '{self._device}' for model '{self.model_name}'")

    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def ensure_save_dir(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predicts instance masks for a single image.

        Args:
            image: A single input image as a NumPy array (H, W, C).

        Returns:
            A NumPy array of instance masks (num_instances, H, W), where each
            slice along the first axis is a binary mask for one instance.
        """
        pass

    def predict_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Predicts instance masks for a batch of images.
        This is a default implementation, subclasses should override this if
        they can process batches more efficiently.
        """
        return [self.predict(image) for image in images]

    def __str__(self) -> str:
        return self.model_name

    def validate(self) -> None:
        """Validate model parameters."""
        pass

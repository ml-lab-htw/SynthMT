import logging
from typing import Tuple

import numpy as np
import imageio.v2 as imageio

from .base import BaseModel
from ..dataset import BenchmarkDataset

logger = logging.getLogger()


class HumanAnnotator(BaseModel):
    def __init__(
        self,
        model_name: str = "human-annotator",
        dataset: BenchmarkDataset = None,
        mask_mapping: Tuple[str, str] = ("single_frame", "single_frame_masks/mk"),
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self._dataset = dataset
        self._mask_mapping = mask_mapping

    def load_model(self):
        pass

    def predict(self, mask_path: str) -> np.ndarray:
        return imageio.imread(mask_path)


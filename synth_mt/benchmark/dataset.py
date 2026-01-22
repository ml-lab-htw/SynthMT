import json
import logging
import os
import random
from glob import glob
from typing import List, Tuple, Dict, Any

import imageio.v2 as imageio
import numpy as np

logger = logging.getLogger(__name__)


class BenchmarkDataset:
    """Loads the synthetic dataset for benchmarking."""

    def get_image_path(self, idx: int) -> str:
        """
        Returns the file path of the image at the given index.
        """
        if idx < 0 or idx >= len(self.image_files):
            raise IndexError("Index out of bounds for dataset.")
        return self.image_files[idx]

    def __init__(
        self,
        image_path: str,
        mask_mapping: Tuple[str, str] = ("images", "masks"),
        num_samples: int = -1,
        seed: int = 42,
    ):
        self.mask_mapping = mask_mapping
        self.image_path = image_path
        self.image_files = sorted(
            glob(os.path.join(self.image_path, "*.png"))
            + glob(os.path.join(self.image_path, "*.jpg"))
            + glob(os.path.join(self.image_path, "*.tif"))
        )

        if 0 < num_samples < len(self.image_files):
            random.seed(seed)
            self.image_files = random.sample(self.image_files, num_samples)

        if not self.image_files:
            raise FileNotFoundError(f"Dataset not found or incomplete in {self.image_path}")

        # check mask mapping
        test_mask_path = self.image_files[0].replace(self.mask_mapping[0], self.mask_mapping[1])
        test_mask_path = test_mask_path.replace("png", "tif")
        if not os.path.exists(test_mask_path):
            logger.warning(f"Dataset does not contain masks: {self.image_path}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Returns:
            - image: (H, W, C)
            - gt_mask: (H, W) with instance labels
            - gt_data: dicts for the frame
        """

        # Load image
        image_path = self.image_files[idx]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = imageio.imread(image_path)

        # Load mask
        mask_path = image_path.replace(self.mask_mapping[0], self.mask_mapping[1])
        mask_path = mask_path.replace("png", "tif")

        if not os.path.exists(mask_path):
            return image, None, []

        mask = imageio.imread(mask_path)

        gt_path = image_path.replace("images", "gt")
        try:
            frame_idx_str = os.path.basename(image_path).split("_")[-1].split(".")[0]
            frame_idx = int(frame_idx_str)
        except ValueError:
            raise ValueError(f"Could not extract frame index from image filename: {image_path}")

        gt_path = gt_path.replace("png", "json")
        gt_path = gt_path.replace("frame", "ground")
        gt_path = gt_path.replace(frame_idx_str, "truth")

        if os.path.isfile(gt_path):
            with open(gt_path, "r") as f:
                gt_data = json.load(f)
                frame_gt = gt_data[frame_idx]
        else:
            frame_gt = {}

        return image, mask, frame_gt

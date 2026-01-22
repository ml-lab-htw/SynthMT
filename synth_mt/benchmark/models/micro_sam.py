from __future__ import annotations

import logging
import os

import numpy as np

from .base import BaseModel

logger = logging.getLogger()

try:
    from micro_sam import util
    from micro_sam.automatic_segmentation import (
        get_predictor_and_segmenter,
        InstanceSegmentationWithDecoder,
    )
except ImportError as e:
    logger.warning(f"Failed to import micro_sam: {e}")


class MicroSAM(BaseModel):
    """
    MicroSAM (μSAM) automatic instance segmentation for microscopy images.

    Notes:
      - Uses the μSAM library for automatic instance segmentation (AIS).
      - Specialized for microscopy data with decoder-based segmentation.
      - Since our images are only 512x512, we are not using any tiling.
    """

    def __init__(
        self,
        model_name: str = "microSAM",
        center_distance_threshold: float = 0.5,
        boundary_distance_threshold: float = 0.5,
        foreground_threshold: float = 0.5,
        foreground_smoothing: float = 1.0,
        distance_smoothing: float = 1.6,
        **kwargs,
    ):
        """
        Initialize MicroSAM (µSAM) model with tunable segmentation parameters.

        Parameters
        ----------
        center_distance_threshold : float
            Center distance predictions below this value will be used to find seeds
            (intersected with thresholded boundary distance predictions).
            Default: 0.5
        boundary_distance_threshold : float
            Boundary distance predictions below this value will be used to find seeds
            (intersected with thresholded center distance predictions).
            Default: 0.5
        foreground_threshold : float
            Foreground predictions above this value will be used as foreground mask.
            Default: 0.5
        foreground_smoothing : float
            Sigma value for smoothing the foreground predictions, to avoid
            checkerboard artifacts in the prediction.
            Default: 1.0
        distance_smoothing : float
            Sigma value for smoothing the distance predictions.
            Default: 1.6
        """
        super().__init__(model_name=model_name, **kwargs)
        self._model_type = "vit_l_lm"
        """
        model_type : str
            μSAM model type. Available options include "vit_l_lm" (default), "vit_b_lm", etc.
        """
        self._checkpoint = None  # str(checkpoint) if checkpoint is not None else None
        self._batch_size = 1
        """
        batch_size : int
            Batch size for processing. Default: 1
        """

        # Segmentation parameters
        self.boundary_distance_threshold = boundary_distance_threshold
        self.center_distance_threshold = center_distance_threshold
        self.distance_smoothing = distance_smoothing
        self.foreground_smoothing = foreground_smoothing
        self.foreground_threshold = foreground_threshold
        self._min_size = 0
        """
        min_size : int
            Minimal object size in the segmentation result.
        """

        # Use non-tiled AIS for 512x512 images (InstanceSegmentationWithDecoder)
        # Note: For 512x512 images with tile_shape=(512,512), no actual tiling occurs,
        # so we use is_tiled=False to get the simpler InstanceSegmentationWithDecoder
        self._is_tiled = False

        # Set μSAM cache dir for weights
        self._save_dir = os.path.join(self._save_dir, "MicroSAM")
        os.environ.setdefault("MICROSAM_CACHEDIR", str(self._save_dir))
        logger.debug(f"Set MICROSAM_CACHEDIR to {self._save_dir}")

        self._predictor = None
        self._segmenter = None

    def load_model(self):
        if self._predictor is not None and self._segmenter is not None:
            return

        self._predictor, self._segmenter = get_predictor_and_segmenter(
            model_type=self._model_type,
            checkpoint=self._checkpoint,
            device=self._device,
            amg=False,  # AIS mode only
            is_tiled=self._is_tiled,
        )
        if not isinstance(self._segmenter, InstanceSegmentationWithDecoder):
            raise RuntimeError("Loaded segmenter is not AIS (InstanceSegmentationWithDecoder)")

    def predict(self, image: np.ndarray) -> np.ndarray:
        if image.ndim not in (2, 3):
            raise ValueError(f"MicroSAM expects (H,W) or (H,W,C); got {image.shape}")

        if self._predictor is None or self._segmenter is None:
            self.load_model()
        assert self._predictor is not None and self._segmenter is not None

        img = np.asarray(image)
        ndim = 2  # RGB/multi-channel treated as 2D image

        # 1) Compute embeddings
        embeddings = util.precompute_image_embeddings(
            predictor=self._predictor,
            input_=img,
            save_path=None,
            ndim=ndim,
            verbose=False,
            batch_size=self._batch_size,
        )

        # 2) Initialize segmenter
        self._segmenter.initialize(
            image=img,
            image_embeddings=embeddings,
            verbose=False,
        )

        # 3) Generate prediction
        masks = self._segmenter.generate(
            center_distance_threshold=self.center_distance_threshold,
            boundary_distance_threshold=self.boundary_distance_threshold,
            foreground_threshold=self.foreground_threshold,
            foreground_smoothing=self.foreground_smoothing,
            distance_smoothing=self.distance_smoothing,
            min_size=self._min_size,
            output_mode=None,
            n_threads=1,
        )

        # 4) Convert to label image if needed
        return masks


if __name__ == "__main__":
    # Simple test
    import matplotlib.pyplot as plt
    from skimage import data

    model = MicroSAM()
    image = data.cells3d()[30, 1]  # Example microscopy image
    seg = model.predict(image)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Input Image")
    ax[1].imshow(seg, cmap="nipy_spectral")
    ax[1].set_title("MicroSAM Segmentation")
    plt.show()

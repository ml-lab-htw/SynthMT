import logging

import numpy as np


from .base import BaseModel

logger = logging.getLogger()

try:
    from cellpose import models
except ImportError as e:
    logger.warning(f"Failed to import cellpose: {e}")

class CellposeSAM(BaseModel):
    """
    Cellpose-SAM (CPSAM) instance segmentation.

    Notes:
      - Uses CellposeModel with pretrained_model='cpsam'.
      - Channels are NOT required/used for CPSAM; it uses the first 3 channels,
        truncating the rest. For grayscale, (H, W) is fine.
      - Diameter is largely invariant for CPSAM (it shouldn’t change which objects are found).
      - The `self._save_dir` has no affect here. Instead the environment variable
        `CELLPOSE_LOCAL_MODELS_PATH` must be set beforehand. Doing it here as in
        `micro_sam.py` sadly does not work because of how CellposeSAM sets up its code in
        `cellpose/models.py`.
    """

    def __init__(
        self,
        model_name: str = "Cellpose-SAM",
        cellprob_threshold: float = 0.0,
        diameter: float | None = None,
        flow_threshold: float = 0.4,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.cellprob_threshold = cellprob_threshold
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self._model = None

    def load_model(self):
        if self._model is not None:
            return

        self._model = models.CellposeModel(gpu=True, pretrained_model="cpsam", use_bfloat16=False)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        image : np.ndarray
            2D (H, W) or 3D (H, W, C). CPSAM will use up to the first 3 channels.

        Returns
        -------
        np.ndarray
            (N, H, W) uint16 stack of instance masks (N=0 if none found).
        """
        if self._model is None:
            self.load_model()
        assert self._model is not None

        masks, _, _ = self._model.eval(
            x=image,
            anisotropy=None,
            augment=False,
            batch_size=1,
            bsize=256,
            cellprob_threshold=self.cellprob_threshold,
            channel_axis=None,
            compute_masks=True,
            diameter=self.diameter,
            do_3D=False,
            flow_threshold=self.flow_threshold,
            invert=False,  # Part of our preprocessing step
            max_size_fraction=1,  # Part of our postprocessing step
            min_size=0,  # Part of our postprocessing step
            niter=None,
            normalize=False,  # Part of our preprocessing step
            # normalize={
            #     "lowhigh": None,
            #     "percentile": [0, 100.0],
            #     "normalize": False,
            #     "sharpen_radius": 0,
            #     "smooth_radius": 0,
            #     "invert": False,
            # },
            resample=True,
            rescale=None,
            stitch_threshold=0.0,
            tile_overlap=0.1,
            z_axis=None,
        )

        if masks is None:
            h, w = image.shape[:2]
            return np.empty((0, h, w), dtype=np.uint16)

        return masks.astype(np.uint16)

    def predict_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Run Cellpose-SAM prediction on a batch of images.
        """
        if self._model is None:
            self.load_model()
        assert self._model is not None

        # The `eval` method of CellposeModel can take a list of images directly.
        masks_batch, _, _ = self._model.eval(
            x=images,
            anisotropy=None,
            augment=False,
            batch_size=len(images),  # Process all images in one batch
            bsize=256,
            cellprob_threshold=self.cellprob_threshold,
            channel_axis=None,
            compute_masks=True,
            diameter=self.diameter,
            do_3D=False,
            flow_threshold=self.flow_threshold,
            invert=False,
            max_size_fraction=1,
            min_size=0,
            niter=None,
            normalize=False,
            resample=True,
            rescale=None,
            stitch_threshold=0.0,
            tile_overlap=0.1,
            z_axis=None,
        )

        # The output `masks_batch` is a list of mask arrays, one for each image.
        processed_masks = []
        for i, masks in enumerate(masks_batch):
            if masks is None:
                h, w = images[i].shape[:2]
                processed_masks.append(np.empty((0, h, w), dtype=np.uint16))
            else:
                processed_masks.append(masks.astype(np.uint16))

        return processed_masks


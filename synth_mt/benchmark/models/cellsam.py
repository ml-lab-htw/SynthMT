import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .base import BaseModel

logger = logging.getLogger()
try:
    from cellSAM import segment_cellular_image, get_model  # , cellsam_pipeline
    from cellSAM.cellsam_pipeline import normalize_image
except ImportError as e:
    logger.warning(f"Failed to import cellSAM: {e}")

class CellSAM(BaseModel):
    """
    CellSAM model. Note: Token (to access and download the model) has to be set in .env file:
    `DEEPCELL_ACCESS_TOKEN=<your-token>`

    Note: Sadly, there is no way to make CellSAM nicely use the `self._save_dir` parameter, as it
    offers no way to dynamically change its models cache dir (but will always use ~/.deepcell).
    When using it on a cluster, we circumvent this by using a symbolik link (see `run_benchmark.yaml`).
    """

    def __init__(
        self,
        model_name: str = "CellSAM",
        bbox_threshold: float = 0.4,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.bbox_threshold = float(bbox_threshold)

        # Ensure DEEPCELL_ACCESS_TOKEN is available
        self._ensure_token()
        self._model = None

    def _ensure_token(self):
        # Load .env first
        env_path = Path.cwd() / ".env"
        load_dotenv(dotenv_path=env_path)

        if "DEEPCELL_ACCESS_TOKEN" not in os.environ:
            raise RuntimeError(
                "DEEPCELL_ACCESS_TOKEN not found. "
                "Provide it via:\n"
                "  • .env file with DEEPCELL_ACCESS_TOKEN=your_token\n"
                "  • env var export DEEPCELL_ACCESS_TOKEN=your_token\n"
                "  • CellSAM(access_token='your_token')"
            )

    def load_model(self):
        """Load CellSAM model on the correct device."""
        return get_model(model="cellsam_general", version="1.2").to(self._device)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run CellSAM model and return (N, H, W) uint16 instance masks.
        """
        if self._model is None:
            self._model = self.load_model()
        assert self._model is not None

        image = np.asarray(image).astype(np.float32)

        # mask = cellsam_pipeline(
        #     image,
        #     chunks=256,
        #     model_path=None,
        #     bbox_threshold=self.bbox_threshold,
        #     low_contrast_enhancement=False,
        #     swap_channels=False,
        #     use_wsi=False,
        #     gauge_cell_size=False,
        #     block_size=0,
        #     overlap=0,
        #     iou_depth=0,
        #     iou_threshold=0,
        # )


        try:

            image = normalize_image(image)  # normalize to 0-1 min max - channelwise
            # NOTE: We have this here because CellSAM's pipeline always does that.
            #  This, in our eyes, has nothing to do with "image preprocessing", but rather
            #  about the image being in the right condition for the model; it belongs more
            #  closely to the architecture than other more advanced actual preprocessing.
            mask, _, _ = segment_cellular_image(
                image,
                model=self._model,
                normalize=False,
                # normalize=True,
                postprocess=False,
                remove_boundaries=False,
                bounding_boxes=None,
                bbox_threshold=self.bbox_threshold,
                fast=False,
                device=self._device,
            )

            # plt.imshow(mask)
            # plt.show()

        except Exception as e:
            mask = np.zeros(image.shape[:2], dtype=np.uint16)
            print(f"CellSAM prediction failed with error: {e}. Returning empty mask.")

        return mask.astype(np.uint64)

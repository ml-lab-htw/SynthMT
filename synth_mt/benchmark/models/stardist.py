from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger()

try:
    from stardist.models import StarDist2D, StarDist3D
except ImportError as e:
    logger.warning(f"Failed to import stardist: {e}")

from .base import BaseModel


class StarDist(BaseModel):
    """
    Generic StarDist wrapper for 2D and 3D models.
    Returns (N,H,W) uint16 masks.
    For 3D models, it can handle 2D images by treating them as 3D images with a single z-slice.

    Subclasses can override `_prepare_model_dir()` to dynamically provide a
    local folder containing a StarDist model (with config/weights/thresholds).
    If `_prepare_model_dir()` returns None, this class will fall back to
    `StarDist2D.from_pretrained(pretrained)` or `StarDist3D.from_pretrained(pretrained)`.

    Notes:
        - The `self._save_dir` has no affect here, since the environment variable `KERAS_HOME` must
          be set beforehand (see, e.g., `scripts/run_benchmark.py`). Doing it here as in `micro_sam.py`
          sadly does not work because of how transformers works internally (.env variable must be set
          first, before importing it :'().

    Parameters
    ----------
    prob_thresh : Optional[float]
        Probability threshold for predict_instances (None -> use model default).
    nms_thresh : Optional[float]
        Non-max suppression threshold for predict_instances (None -> use model default).
        Lower value suppresses more.
    """

    def __init__(
        self,
        model_name: str = "StarDist",
        pretrained: str = "2D_versatile_fluo",
        prob_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        **kwargs,
    ):
        """
        pretrained : Optional[str]
            Built-in StarDist model name (e.g. '2D_versatile_fluo', '3D_versatile_fluo').

        2D_versatile_fluo
        2D_versatile_he
        2D_paper_dsb2018

        Default values for prob_thresh and nms_thresh are loaded automatically when None is given here.
        """
        super().__init__(model_name=f"{model_name}", **kwargs)
        self.grayscale = True
        self.pretrained = pretrained
        self._model_dir = None
        """
        model_dir : Optional[Union[str, Path]]
            Path to a folder holding a trained StarDist model (contains config/weights/thresholds).
        """
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self._is_3d = None

        self._model: Optional[Union[StarDist2D, StarDist3D]] = None  # lazy-loaded

    # -------------------------
    # Hooks for subclasses
    # -------------------------
    def _prepare_model_dir(self) -> Optional[Path]:
        """
        Return a directory that contains a StarDist model (with config/weights/thresholds),
        or None to use `pretrained` instead. Subclasses can override to implement
        on-demand downloads or custom discovery.

        Base implementation: return `self._model_dir` if provided; else None.
        """
        return self._model_dir

    # -------------------------
    # Internals
    # -------------------------
    def load_model(self) -> None:
        if self._model is not None:
            return

        model_dir = self._prepare_model_dir()
        if model_dir is not None:
            model_dir = Path(model_dir).resolve()
            if not model_dir.exists():
                raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

            config_path = model_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"config.json not found in model_dir: {model_dir}")

            with open(config_path) as f:
                config = json.load(f)

            n_dim = config.get("n_dim")
            if n_dim not in [2, 3]:
                raise ValueError(f"Unsupported n_dim in config.json: {n_dim}")
            self._is_3d = n_dim == 3

            basedir = str(model_dir.parent)
            name = model_dir.name
            ModelClass = StarDist3D if self._is_3d else StarDist2D
            self._model = ModelClass(None, name=name, basedir=basedir)
        else:
            if self.pretrained is None:
                raise ValueError("Either provide 'pretrained' or a 'model_dir'.")

            self._is_3d = "3D_" in self.pretrained or self.pretrained == "AnyStar"
            ModelClass = StarDist3D if self._is_3d else StarDist2D
            try:
                self._model = ModelClass.from_pretrained(self.pretrained)
            except Exception as e:
                # Provide a more informative error if the pretrained model is not found
                available_models = [
                    m["name"] for m in ModelClass.get_pretained_models_list()
                ]  # TODO: this does not work
                raise ValueError(
                    f"Pretrained model '{self.pretrained}' not found for {ModelClass.__name__}. "
                    f"Available models: {available_models}"
                ) from e

        # adopt stored thresholds if user didn't override
        thr = getattr(self._model, "thresholds", None)
        if self.prob_thresh is None and getattr(thr, "prob", None) is not None:
            self.prob_thresh = float(thr.prob)
        if self.nms_thresh is None and getattr(thr, "nms", None) is not None:
            self.nms_thresh = float(thr.nms)

    # -------------------------
    # Public API
    # -------------------------
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict instance masks for a single image."""
        if self._model is None:
            self.load_model()
        assert self._model is not None

        img = image.copy().astype(np.float32)

        # --- Input image handling ---
        axes = None
        if not self._is_3d:
            # Handle different pretrained 2D models
            if self.pretrained == "2D_versatile_he":
                # This model expects a color image (Y,X,C).
                if img.ndim == 2:
                    # If grayscale, convert to RGB for this model.
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                axes = "YXC"
            else:
                # Other 2D models expect grayscale (Y,X).
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                axes = "YX"
        else:  # 3D model (including AnyStar)
            if img.ndim == 3 and img.shape[-1] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # If we have a 2D image for a 3D model, add a Z axis
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            axes = "ZYX"

        # --- Prediction ---
        labels, _ = self._model.predict_instances(
            img,
            axes=axes,
            n_tiles=None,
            prob_thresh=self.prob_thresh,
            nms_thresh=self.nms_thresh,
            verbose=False,
        )
        return labels.astype(np.uint16)

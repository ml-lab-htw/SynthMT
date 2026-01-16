import logging
import os

import cv2
import numpy as np

from synth_mt.benchmark.models.anchor_point_model import AnchorPointModel
from synth_mt.utils.matlab import MatlabEngine

logger = logging.getLogger(__name__)


try:
    import matlab.engine
except ImportError:
    logger.warning("Matlab engine not available")
    matlab = None


class FIESTA(AnchorPointModel):
    def __init__(
        self,
        background_filter: bool = False,
        binary_image_processing: str = "average",
        dynamicfil: bool = True,
        focus_correction: bool = True,
        fwhm_estimate: float = 3.0,
        height_threshold: float = 0.1,
        min_cod: float = 0.05,
        model_name="FIESTA",
        reduce_fit_box: float = 1.0,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)

        self.grayscale = True
        self.background_filter = background_filter
        self.binary_image_processing = binary_image_processing
        self.dynamicfil = dynamicfil
        self.focus_correction = focus_correction
        self.fwhm_estimate = fwhm_estimate
        self.height_threshold = height_threshold
        self.min_cod = min_cod
        self.reduce_fit_box = reduce_fit_box

        self._areafilt_max = 1e6
        self._areafilt_min = 0
        self._find_beads = False
        self._find_molecules = True
        self._include_data = True
        self._quadratic = "_quadratic"
        self._scale = 1.0

    def load_model(self):
        """No model to load for FIESTA."""
        fiesta_path = os.path.join(".", "fiesta")
        fiesta_bin_path = os.path.join(fiesta_path, "bin")
        if not os.path.exists(fiesta_bin_path):
            raise FileNotFoundError(f"FIESTA path does not exist: {fiesta_bin_path}")

        if matlab:
            eng = MatlabEngine.get_engine()
            eng.addpath(eng.genpath(fiesta_path), nargout=0)
            eng.addpath(eng.genpath(fiesta_bin_path), nargout=0)
        else:
            raise ImportError("MATLAB engine for Python is not installed.")

    def _build_params(self):
        """Build MATLAB-compatible params struct (dict auto-converted)."""
        return {
            "areafilt": matlab.double([float(self._areafilt_min), float(self._areafilt_max)]),
            "background_filter": bool(self.background_filter),
            "binary_image_processing": self.binary_image_processing,
            "creation_time_vector": 1.0,
            "dynamicfil": bool(self.dynamicfil),
            "find_beads": bool(self._find_beads),
            "find_molecules": bool(self._find_molecules),
            "focus_correction": bool(self.focus_correction),
            "fwhm_estimate": float(self.fwhm_estimate),
            "height_threshold": float(self.height_threshold),
            "include_data": bool(self._include_data),
            "min_cod": float(self.min_cod),
            "options": {},  # becomes MATLAB struct
            "quadratic": self._quadratic,
            "reduce_fit_box": float(self.reduce_fit_box),
            "scale": float(self._scale),
        }

    def _run_backend(self, image: np.ndarray) -> str | None:
        """Run MATLAB backend and return the path of the CSV file."""
        eng = MatlabEngine.get_engine()

        if image.ndim == 3 and image.shape[2] > 1:
            image = image.astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image_matlab = matlab.double(image.tolist(), size=image.shape)
        params_matlab = self._build_params()

        predictions_dir = os.path.join(self._work_dir, "Predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        csv_filename = os.path.join(
            predictions_dir, f"fiesta_pred_{len(os.listdir(predictions_dir))}.csv"
        )

        # plt.imshow(image, cmap="gray")
        # plt.title("FIESTA Input Image")
        # plt.axis("off")
        # plt.show()

        eng.run_scan_image(image_matlab, params_matlab, csv_filename, nargout=0)
        if not os.path.isfile(csv_filename):
            logger.warning(
                f"FileNotFoundError: Expected CSV not found at: {csv_filename}. "
                "Probably because no instances found; returning None."
            )
            csv_filename = None

        # plt.imshow(predMask, cmap="gray")
        # plt.title("FIESTA Prediction Mask")
        # plt.axis("off")
        # plt.show()

        return csv_filename

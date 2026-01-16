import logging
from dataclasses import dataclass

from .base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class AlbumentationsConfig(BaseConfig):
    """Configuration for post-generation image augmentations using Albumentations."""

    # --- Geometric Transforms ---
    p: float = 0.0  # The "master" probability that any augmentation is applied to a frame.

    rotate_limit: int = 0  # Max rotation in degrees. Set to 0 to disable.
    affine_p: float = 0.0  # Probability of applying affine transformations.
    shift_scale_rotate_p: float = 0.0

    horizontal_flip_p: float = 0.0
    vertical_flip_p: float = 0.0

    # Simulates tissue stretching. Crucial for microscopy.
    elastic_p: float = 0.0
    elastic_alpha: int = 1
    elastic_sigma: int = 20
    elastic_alpha_affine: int = 20

    # Simulates lens distortion.
    grid_distortion_p: float = 0.0

    # --- Pixel-level & Noise Transforms ---
    brightness_contrast_p: float = 0.0
    brightness_limit: float = 0.1
    contrast_limit: float = 0.1

    gauss_noise_p: float = 0.0
    gauss_noise_mean_range: tuple[float, float] = (-0.1, 0.1)
    gauss_noise_std_range: tuple[float, float] = (0.1, 0.5)

    def __post_init__(self):
        """
        Called after the dataclass is initialized.
        Logs the creation and key parameters of the configuration.
        """
        logger.info("Albumentations configuration loaded.")
        logger.debug(f"Master augmentation probability (p) set to: {self.p}")
        # Log other important high-level settings if desired
        if self.rotate_limit == 0:
            logger.debug("Rotation is disabled (rotate_limit=0).")
        else:
            logger.debug(f"Rotation limit set to: {self.rotate_limit} degrees.")

    def validate(self):
        """
        Validates the configuration parameters.
        Raises ValueError on failure.
        """
        logger.debug("Validating AlbumentationsConfig...")

        if not (0.0 <= self.p <= 1.0):
            msg = f"Master probability 'p' must be between 0 and 1, but got {self.p}"
            logger.error(msg)
            raise ValueError(msg)

        # You could add more validation checks here for other parameters
        # For example:
        # if not (0.0 <= self.horizontal_flip_p <= 1.0):
        #     msg = f"'horizontal_flip_p' must be between 0 and 1, but got {self.horizontal_flip_p}"
        #     logger.error(msg)
        #     raise ValueError(msg)

        logger.debug("AlbumentationsConfig validation successful.")

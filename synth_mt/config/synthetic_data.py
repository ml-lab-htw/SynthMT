import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional

from .base import BaseConfig
from .spots import SpotConfig

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class SyntheticDataConfig(BaseConfig):
    """
    Configuration for synthetic microtubule video generation using a stateful,
    event-driven model for dynamic instability.
    """

    # ─── core video info ────────────────────────────────────
    id: int | str = 309
    img_size: Tuple[int, int] = (512, 512)
    num_frames: int = 50
    fps: int = 5
    num_microtubule: int = 10
    microtubule_seed_min_dist: int = 10
    margin: int = 5

    # --- Microtubule Dynamics ---
    growth_speed: float = 2.5  # Average length change per frame when growing
    shrink_speed: float = 5.0  # Average length change per frame when shrinking
    catastrophe_prob: float = 0.01  # Probability of switching from growth to shrink per frame
    rescue_prob: float = 0.01  # Probability of switching from shrink to growth per frame
    max_pause_at_min_frames: int = 5  # Max frames to pause at min length before forced rescue

    growth_speed_std: float = 0.0
    shrink_speed_std: float = 0.0
    catastrophe_prob_std: float = 0.0
    rescue_prob_std: float = 0.0

    # --- Microtubule Geometry ---
    base_wagon_length_min: float = 5.0
    base_wagon_length_max: float = 50.0
    microtubule_length_min: int = 100
    microtubule_length_max: int = 200

    # Bending is applied to the dynamic "tail" part of the microtubule.
    tail_wagon_length: float = (
        10.0  # Visual segments for drawing the tail. Does not affect growth speed.
    )
    # The bending angles for the microtubule's visual segments are drawn from a Gamma distribution,
    # which is defined by a shape (k) and a scale (θ) parameter. The mean bend angle is shape * scale.
    bending_angle_gamma_shape: float = (
        1.0  # Shape (k) of the Gamma distribution for bend angles. Controls the *variability* of curvature.
        # - Small values (e.g., < 1) lead to mostly straight segments with occasional sharp, random kinks.
        # - Large values (e.g., > 2) lead to smoother, more uniform curves, as bend angles are more consistent.
    )
    bending_angle_gamma_scale: float = (
        0.005  # Scale (θ) of the Gamma distribution for bend angles. Controls the *magnitude* of curvature.
        # - Small values result in smaller average bend angles, making microtubules straighter.
        # - Large values result in larger average bend angles, making microtubules more curved or "wiggly".
    )
    max_angle_sign_changes: int = 1  # 0 for C-shape, 1 for S-shape, etc.
    prob_to_flip_bend: float = (
        0.1  # Probability to use an available sign change when adding new visual wagons
    )

    # --- Minus-end (opposite direction) dynamics ---
    minus_end_target_length_mean: float = (
        0.0  # Mean of the target length distribution for the minus-end.
    )
    minus_end_target_length_std: float = 20.0  # Std dev of the target length distribution.
    minus_end_velocity: float = 1.0  # Speed of growth/shrink for the minus-end.

    # --- Photophysics & Camera Realism ---
    psf_sigma_h: float = 0.3  # Horizontal PSF sigma (line width)
    psf_sigma_v: float = 0.75  # Vertical PSF sigma (line length)
    tubule_width_variation: float = 0.1  # % variation in line width
    background_level: float = 0.7  # Base background level [0, 1]
    tubulus_contrast: float = -0.2  # Base contrast of microtubule body. Negative for dark-on-light.
    seed_red_channel_boost: float = 0.2  # How much to boost red channel for the seed segment.
    tip_brightness_factor: float = 1.2  # Multiplier for the growing tip's brightness.
    red_channel_noise_std: float = 0.01  # Std dev of noise added only to the red channel.
    quantum_efficiency: float = 80.0  # Factor for Poisson noise simulation.
    gaussian_noise: float = 0.05  # Std dev of global Gaussian noise.
    vignetting_strength: float = 0.1  # Strength of the radial vignetting effect.
    global_blur_sigma: float = 0.5  # Sigma for a final global Gaussian blur.
    jitter_px: float = 0.5  # Random jitter applied to all microtubules each frame.

    # --- Ancillary Objects (Spots) ---
    fixed_spots: SpotConfig = field(default_factory=lambda: SpotConfig(count=50))
    moving_spots: SpotConfig = field(default_factory=lambda: SpotConfig(count=0))
    random_spots: SpotConfig = field(default_factory=lambda: SpotConfig(count=20))

    # --- Rendering & Output ---
    color_mode: bool = True
    global_contrast: float = 0.0
    global_brightness: float = 0.0
    annotation_color_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    um_per_pixel: float = 0.1
    scale_bar_um: float = 5.0
    show_time: bool = False
    show_scale: bool = False
    generate_mt_mask: bool = True
    generate_seed_mask: bool = False
    albumentations: Optional[dict] = None

    def __post_init__(self):
        super().__post_init__()
        logger.debug(f"SyntheticDataConfig '{self.id}' initialized. Running initial validation...")
        try:
            self.validate()
        except ValueError as e:
            logger.critical(
                f"Initial validation of SyntheticDataConfig failed: {e}", exc_info=False
            )
            raise

    def validate(self):
        """Validates configuration parameters."""
        logger.debug(f"Starting validation for SyntheticDataConfig '{self.id}'...")
        errors = []
        if self.base_wagon_length_min > self.base_wagon_length_max:
            errors.append("base_wagon_length_min cannot be greater than base_wagon_length_max.")
        if self.microtubule_length_min > self.microtubule_length_max:
            errors.append("microtubule_length_min cannot be greater than microtubule_length_max.")
        if self.psf_sigma_h <= 0 or self.psf_sigma_v <= 0:
            errors.append("PSF sigmas must be positive.")
        if self.fps <= 0:
            errors.append("FPS must be positive.")

        if errors:
            raise ValueError(
                f"SyntheticDataConfig validation failed for '{self.id}':\n" + "\n".join(errors)
            )
        logger.debug(f"SyntheticDataConfig '{self.id}' validation successful.")

    @classmethod
    def from_trial(cls, trial):
        """Creates a SyntheticDataConfig from an Optuna trial."""

        obj = cls()
        properties = vars(obj).keys()

        # general fields
        for field_name in properties:
            if field_name in trial.params:
                value = trial.params[field_name]
                setattr(obj, field_name, value)
                # print(f"YES: {field_name}={value}")
            # else:
            #     print(f"NO:  '{field_name}'")

        obj.fixed_spots = SpotConfig.from_trial(trial, "fixed_spots")
        obj.moving_spots = SpotConfig.from_trial(trial, "moving_spots")
        obj.random_spots = SpotConfig.from_trial(trial, "random_spots")

        return obj

import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional

from optuna import Trial

from .base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class SpotTuningConfig(BaseConfig):
    """Configuration for tuning the parameters of a single type of spot."""

    count_range: Tuple[int, int] = (0, 50)
    intensity_min_range: Tuple[float, float] = (0.0, 0.2)
    intensity_max_range: Tuple[float, float] = (0.0, 0.3)
    radius_min_range: Tuple[int, int] = (1, 5)
    radius_max_range: Tuple[int, int] = (5, 10)
    kernel_size_min_range: Tuple[int, int] = (0, 5)
    kernel_size_max_range: Tuple[int, int] = (5, 10)
    sigma_range: Tuple[float, float] = (0.1, 5.0)

    polygon_p_range: Optional[Tuple[float, float]] = (0.0, 1.0)
    polygon_vertex_count_min_range: Optional[Tuple[int, int]] = (
        3,
        10,
    )  # Min vertices for a polygon (e.g., triangle)
    polygon_vertex_count_max_range: Optional[Tuple[int, int]] = (
        3,
        10,
    )  # Max vertices for a polygon (e.g., heptagon)

    # Specific to moving spots
    max_step_range: Optional[Tuple[int, int]] = None

    def validate(self):
        pass

    def from_trial(self, trial: Trial, name: str) -> "SpotConfig":
        """Creates a SpotConfig instance by suggesting parameters from an Optuna trial."""
        logger.debug(
            f"Suggesting SpotConfig parameters for '{name}' using Optuna trial {trial.number}."
        )

        # --- Standard Parameters ---
        count = trial.suggest_int(f"{name}_count", *self.count_range)
        intensity_min = trial.suggest_float(f"{name}_intensity_min", *self.intensity_min_range)
        intensity_max = trial.suggest_float(
            f"{name}_intensity_max",
            max(intensity_min, self.intensity_max_range[0]),
            self.intensity_max_range[1],
        )
        radius_min = trial.suggest_int(f"{name}_radius_min", *self.radius_min_range)
        radius_max = trial.suggest_int(
            f"{name}_radius_max",
            max(radius_min, self.radius_max_range[0]),
            self.radius_max_range[1],
        )
        kernel_size_min = trial.suggest_int(f"{name}_kernel_size_min", *self.kernel_size_min_range)
        kernel_size_max = trial.suggest_int(
            f"{name}_kernel_size_max",
            max(kernel_size_min, self.kernel_size_max_range[0]),
            self.kernel_size_max_range[1],
        )
        sigma = trial.suggest_float(f"{name}_sigma", *self.sigma_range)

        # --- Moving Spot Specific ---
        max_step = None
        if self.max_step_range is not None:
            max_step = trial.suggest_int(f"{name}_max_step", *self.max_step_range)

        # --- Polygon and Color Parameters ---
        polygon_p = 0.0
        if self.polygon_p_range:
            polygon_p = trial.suggest_float(f"{name}_polygon_p", *self.polygon_p_range)

        polygon_vertex_count_min = 3
        if self.polygon_vertex_count_min_range:
            polygon_vertex_count_min = trial.suggest_int(
                f"{name}_polygon_vertex_count_min", *self.polygon_vertex_count_min_range
            )

        polygon_vertex_count_max = 7
        if self.polygon_vertex_count_max_range:
            polygon_vertex_count_max = trial.suggest_int(
                f"{name}_polygon_vertex_count_max",
                max(polygon_vertex_count_min, self.polygon_vertex_count_max_range[0]),
                self.polygon_vertex_count_max_range[1],
            )

        config = SpotConfig(
            count=count,
            intensity_min=intensity_min,
            intensity_max=intensity_max,
            radius_min=radius_min,
            radius_max=radius_max,
            kernel_size_min=kernel_size_min,
            kernel_size_max=kernel_size_max,
            sigma=sigma,
            max_step=max_step,
            polygon_p=polygon_p,
            polygon_vertex_count_min=polygon_vertex_count_min,
            polygon_vertex_count_max=polygon_vertex_count_max,
        )
        logger.debug(
            f"Successfully created SpotConfig for '{name}' via Optuna trial. Final config: {config.asdict()}"
        )
        return config


@dataclass(eq=False)
class SpotConfig(BaseConfig):
    """Configuration for a single type of spot in a synthetic video."""

    count: int = 20
    intensity_min: float = 0.005
    intensity_max: float = 0.08
    radius_min: int = 1
    radius_max: int = 3
    kernel_size_min: int = 0
    kernel_size_max: int = 2
    sigma: float = 0.5

    polygon_p: float = 0.0  # Probability of a spot being a polygon instead of a circle
    polygon_vertex_count_min: int = 3  # Min vertices for a polygon (e.g., triangle)
    polygon_vertex_count_max: int = 7  # Max vertices for a polygon (e.g., heptagon)

    # Specific to moving spots
    max_step: Optional[int] = 0

    def __post_init__(self):
        super().__post_init__()
        self.polygon_vertex_count_max = max(
            self.polygon_vertex_count_min, self.polygon_vertex_count_max
        )
        self.kernel_size_max = max(self.kernel_size_min, self.kernel_size_max)
        logger.debug(f"SpotConfig initialized with count={self.count}.")

    def validate(self):
        logger.debug("Validating SpotConfig parameters...")
        errors = []

        if not (self.count >= 0):
            errors.append(f"Count must be non-negative, but got {self.count}.")

        if not (-1.0 <= self.intensity_min <= 1.0):
            errors.append(
                f"Intensity min must be between 0.0 and 1.0, but got {self.intensity_min}."
            )
        if not (-1.0 <= self.intensity_max <= 1.0):
            errors.append(
                f"Intensity max must be between 0.0 and 1.0, but got {self.intensity_max}."
            )
        if not (self.intensity_min <= self.intensity_max):
            errors.append(
                f"Intensity min ({self.intensity_min}) must be less than or equal to intensity max ({self.intensity_max})."
            )

        if not (self.radius_min >= 0):
            errors.append(f"Radius min must be non-negative, but got {self.radius_min}.")
        if not (self.radius_max >= 0):
            errors.append(f"Radius max must be non-negative, but got {self.radius_max}.")
        if not (self.radius_min <= self.radius_max):
            errors.append(
                f"Radius min ({self.radius_min}) must be less than or equal to radius max ({self.radius_max})."
            )

        if not (self.kernel_size_min >= 0):
            errors.append(f"Kernel size min must be non-negative, but got {self.kernel_size_min}.")
        if not (self.kernel_size_max >= 0):
            errors.append(f"Kernel size max must be non-negative, but got {self.kernel_size_max}.")
        if not (self.kernel_size_min <= self.kernel_size_max):
            errors.append(
                f"Kernel size min ({self.kernel_size_min}) must be less than or equal to kernel size max ({self.kernel_size_max})."
            )

        if not (self.sigma > 0.0):
            errors.append(f"Sigma must be positive, but got {self.sigma}.")

        if not (0.0 <= self.polygon_p <= 1.0):
            errors.append(f"polygon_p must be between 0.0 and 1.0, but got {self.polygon_p}.")
        if not (3 <= self.polygon_vertex_count_min <= self.polygon_vertex_count_max):
            errors.append(
                f"polygon_vertex_count_min ({self.polygon_vertex_count_min}) must be >= 3 and <= polygon_vertex_count_max ({self.polygon_vertex_count_max})."
            )

        if errors:
            full_msg = f"SpotConfig validation failed with {len(errors)} error(s):\n" + "\n".join(
                errors
            )
            logger.error(full_msg)
            raise ValueError(full_msg)

        logger.debug("SpotConfig validation successful.")

    @classmethod
    def from_trial(cls, trial, type):

        obj = cls()
        spot_props = vars(obj).keys()
        for field_name in spot_props:
            trial_name = type + "_" + field_name
            if trial_name in trial.params:
                value = trial.params[trial_name]
                setattr(obj, field_name, value)
            #     print(f"YES: {field_name}={value}")
            # else:
            #     print(f"NO:  {field_name}")

        return obj

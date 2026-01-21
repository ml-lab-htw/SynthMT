import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

from optuna import Trial

from .base import BaseConfig
from .spots import SpotTuningConfig, SpotConfig
from .synthetic_data import SyntheticDataConfig

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class TuningConfig(BaseConfig):
    """
    Configuration for hyperparameter tuning of synthetic microtubule data generation.

    This version is optimized for SINGLE-FRAME comparison by fixing inter-frame
    dynamic parameters to default values.
    """

    # ─── General tuning settings ────────────────────────────────────
    model_name: str = "openai/clip-vit-base-patch32"
    embedding_layer: int = 3
    hf_cache_dir: Optional[str] = None
    reference_video_path: str = ""
    reference_images_dir: str = ""
    num_compare_frames: int = 1
    temp_dir: str = ".temp"
    output_config_folder: str = "/scratch/koddenbrock/mt/hpo/optimal_configs"
    output_config_num_best: int = 10  # Number of best configs to save
    output_config_id: int | str = "best_synthetic_config"
    output_config_num_frames: int = 50
    output_config_num_png: int = 10  # Number of sample PNG frames to save
    direction: str = "maximize"
    similarity_metric: str = "fid"
    num_trials: int = 100
    pca_components: Optional[int] = 0
    load_if_exists: bool = True

    # ─── Static Video Properties (not tuned) ────────────────────────
    img_size: Tuple[int, int] = (512, 512)
    fps: int = 5

    # =========================================================================
    #         TUNABLE PARAMETER RANGES (STATIC/VISUAL PROPERTIES)
    # =========================================================================

    # ─── Microtubule geometry & bending ranges ────────────────────────
    base_segment_length_min_range: Tuple[float, float] = (2.0, 20.0)
    base_segment_length_max_range: Tuple[float, float] = (30.0, 80.0)
    microtubule_length_min_range: Tuple[int, int] = (50, 150)
    microtubule_length_max_range: Tuple[int, int] = (100, 250)
    tail_segment_length_range: Tuple[float, float] = (5.0, 20.0)
    bending_angle_gamma_shape_range: Tuple[float, float] = (0.5, 2.0)
    bending_angle_gamma_scale_range: Tuple[float, float] = (0.01, 0.05)
    max_angle_sign_changes_range: Tuple[int, int] = (0, 3)
    minus_end_target_length_std_range: Tuple[float, float] = (1.0, 8.0)  # Tunable std dev
    minus_end_target_length_mean_range: Tuple[float, float] = (-2.0, 2.0)
    prob_to_flip_bend_range: Tuple[float, float] = (0.0, 0.1)

    # ─── Seeding & rendering ranges ───────────────────────
    num_microtubule_range: Tuple[int, int] = (5, 50)
    microtubule_seed_min_dist: int = 10
    margin: int = 5
    psf_sigma_h_range: Tuple[float, float] = (0.2, 0.4)
    psf_sigma_v_range: Tuple[float, float] = (0.6, 0.9)
    tubule_width_variation_range: Tuple[float, float] = (0.0, 0.2)

    # ─── Photophysics / camera realism ranges ─────────────────────
    background_level_range: Tuple[float, float] = (0.5, 0.9)
    tubulus_contrast_range: Tuple[float, float] = (-0.6, 0.6)
    seed_red_channel_boost_range: Tuple[float, float] = (0.2, 0.6)
    tip_brightness_factor_range: Tuple[float, float] = (1.0, 1.5)
    red_channel_noise_std_range: Tuple[float, float] = (0.0, 0.05)
    quantum_efficiency_range: Tuple[float, float] = (40.0, 120.0)
    gaussian_noise_range: Tuple[float, float] = (0.01, 0.15)
    vignetting_strength_range: Tuple[float, float] = (0.0, 0.2)
    global_blur_sigma_range: Tuple[float, float] = (0.3, 1.5)
    global_contrast_range: Tuple[float, float] = (0.0, 0.2)
    global_brightness_range: Tuple[float, float] = (0.0, 0.2)

    # ─── Spot Tuning Ranges (Static Spots Only) ───────────────────────
    fixed_spots_tuning: SpotTuningConfig = field(
        default_factory=lambda: SpotTuningConfig(
            count_range=(0, 100),
            intensity_min_range=(0.0001, 0.1),
            intensity_max_range=(0.1, 0.3),
            polygon_p_range=(0.0, 0.7),
            polygon_vertex_count_min_range=(3, 5),
            polygon_vertex_count_max_range=(5, 10),
        )
    )
    random_spots_tuning: SpotTuningConfig = field(
        default_factory=lambda: SpotTuningConfig(
            count_range=(0, 50), intensity_min_range=(0.0001, 0.1), intensity_max_range=(0.0, 0.5)
        )
    )

    # =========================================================================
    #          FIXED PARAMETERS (NOT TUNED)
    # =========================================================================

    # --- Inter-frame dynamics are fixed as they don't affect single-frame comparison ---
    growth_speed: float = 2.5
    shrink_speed: float = 5.0
    catastrophe_prob: float = 0.01
    rescue_prob: float = 0.01
    max_pause_at_min_frames: int = 5
    jitter_px: float = 0.5
    minus_end_velocity: float = 1.0  # Fixed dynamics
    moving_spots: SpotConfig = field(default_factory=lambda: SpotConfig(count=0))  # Disabled

    # --- Other fixed parameters ---
    color_mode: bool = True
    annotation_color_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    um_per_pixel: float = 0.1
    scale_bar_um: float = 5.0
    show_time: bool = False
    show_scale: bool = False
    generate_mt_mask: bool = True
    generate_seed_mask: bool = False

    def validate(self):
        """
        Validates the tuning configuration.
        """
        logger.debug("Validating TuningConfig...")

        # 1. Check reference data source
        if not self.reference_video_path and not self.reference_images_dir:
            raise ValueError(
                "Either 'reference_video_path' or 'reference_images_dir' must be specified."
            )
        if self.reference_video_path and self.reference_images_dir:
            raise ValueError(
                "Specify either 'reference_video_path' or 'reference_images_dir', not both."
            )
        if self.reference_video_path and not os.path.isfile(self.reference_video_path):
            raise FileNotFoundError(f"Reference video not found: {self.reference_video_path}")
        if self.reference_images_dir and not os.path.isdir(self.reference_images_dir):
            raise FileNotFoundError(
                f"Reference image directory not found: {self.reference_images_dir}"
            )

        # 2. Check optimization direction
        if self.direction not in ["maximize", "minimize"]:
            raise ValueError(
                f"Invalid direction '{self.direction}'. Must be 'maximize' or 'minimize'."
            )

        # 3. Check trial and output numbers
        if self.num_trials <= 0:
            raise ValueError(f"'num_trials' must be positive, but got {self.num_trials}.")
        if self.output_config_num_best <= 0:
            raise ValueError(
                f"'output_config_num_best' must be positive, but got {self.output_config_num_best}."
            )

        logger.debug("TuningConfig validation successful.")

    def suggest_synthetic_config_from_trial(self, trial: Trial) -> SyntheticDataConfig:
        """
        Uses the ranges defined in this tuning config to suggest parameters for an Optuna trial.
        """
        logger.debug(f"Generating SyntheticDataConfig for Optuna trial {trial.number}.")
        suggested_params = {}

        # --- Microtubule Geometry (with dependent ranges) ---
        suggested_params["base_segment_length_min"] = trial.suggest_float(
            "base_segment_length_min", *self.base_segment_length_min_range
        )
        suggested_params["base_segment_length_max"] = trial.suggest_float(
            "base_segment_length_max",
            max(suggested_params["base_segment_length_min"], self.base_segment_length_max_range[0]),
            self.base_segment_length_max_range[1],
        )

        suggested_params["microtubule_length_min"] = trial.suggest_int(
            "microtubule_length_min",
            max(
                int(suggested_params["base_segment_length_max"]),
                self.microtubule_length_min_range[0],
            ),
            self.microtubule_length_min_range[1],
        )
        suggested_params["microtubule_length_max"] = trial.suggest_int(
            "microtubule_length_max",
            max(suggested_params["microtubule_length_min"], self.microtubule_length_max_range[0]),
            self.microtubule_length_max_range[1],
        )

        # --- Bending & Minus-End ---
        suggested_params["tail_segment_length"] = trial.suggest_float(
            "tail_segment_length", *self.tail_segment_length_range
        )
        suggested_params["bending_angle_gamma_shape"] = trial.suggest_float(
            "bending_angle_gamma_shape", *self.bending_angle_gamma_shape_range
        )
        suggested_params["bending_angle_gamma_scale"] = trial.suggest_float(
            "bending_angle_gamma_scale", *self.bending_angle_gamma_scale_range
        )
        suggested_params["max_angle_sign_changes"] = trial.suggest_int(
            "max_angle_sign_changes", *self.max_angle_sign_changes_range
        )
        suggested_params["minus_end_target_length_std"] = trial.suggest_float(
            "minus_end_target_length_std", *self.minus_end_target_length_std_range
        )
        suggested_params["minus_end_target_length_mean"] = trial.suggest_float(
            "minus_end_target_length_mean", *self.minus_end_target_length_mean_range
        )
        suggested_params["prob_to_flip_bend"] = trial.suggest_float(
            "prob_to_flip_bend", *self.prob_to_flip_bend_range
        )

        # --- Seeding & Rendering ---
        suggested_params["num_microtubule"] = trial.suggest_int(
            "num_microtubule", *self.num_microtubule_range
        )
        suggested_params["psf_sigma_h"] = trial.suggest_float(
            "psf_sigma_h", *self.psf_sigma_h_range
        )
        suggested_params["psf_sigma_v"] = trial.suggest_float(
            "psf_sigma_v", *self.psf_sigma_v_range
        )
        suggested_params["tubule_width_variation"] = trial.suggest_float(
            "tubule_width_variation", *self.tubule_width_variation_range
        )

        # --- Photophysics & Camera Realism ---
        suggested_params["background_level"] = trial.suggest_float(
            "background_level", *self.background_level_range
        )
        suggested_params["tubulus_contrast"] = trial.suggest_float(
            "tubulus_contrast", *self.tubulus_contrast_range
        )
        suggested_params["seed_red_channel_boost"] = trial.suggest_float(
            "seed_red_channel_boost", *self.seed_red_channel_boost_range
        )
        suggested_params["tip_brightness_factor"] = trial.suggest_float(
            "tip_brightness_factor", *self.tip_brightness_factor_range
        )
        suggested_params["red_channel_noise_std"] = trial.suggest_float(
            "red_channel_noise_std", *self.red_channel_noise_std_range
        )
        suggested_params["quantum_efficiency"] = trial.suggest_float(
            "quantum_efficiency", *self.quantum_efficiency_range
        )
        suggested_params["gaussian_noise"] = trial.suggest_float(
            "gaussian_noise", *self.gaussian_noise_range
        )
        suggested_params["vignetting_strength"] = trial.suggest_float(
            "vignetting_strength", *self.vignetting_strength_range
        )
        suggested_params["global_blur_sigma"] = trial.suggest_float(
            "global_blur_sigma", *self.global_blur_sigma_range
        )
        suggested_params["global_contrast"] = trial.suggest_float(
            "global_contrast", *self.global_contrast_range
        )
        suggested_params["global_brightness"] = trial.suggest_float(
            "global_brightness", *self.global_brightness_range
        )

        # --- Spots (Static only) ---
        fixed_spots_cfg = self.fixed_spots_tuning.from_trial(
            trial,
            "fixed_spots",
        )
        random_spots_cfg = self.random_spots_tuning.from_trial(
            trial,
            "random_spots",
        )

        # --- Build the final config object ---
        synth_cfg = SyntheticDataConfig(
            id=f"trial_{trial.number}",
            img_size=self.img_size,
            fps=self.fps,
            num_frames=self.num_compare_frames,
            # Tuned static spots
            fixed_spots=fixed_spots_cfg,
            random_spots=random_spots_cfg,
            # Pass all fixed (non-tuned) parameters explicitly
            moving_spots=self.moving_spots,
            growth_speed=self.growth_speed,
            shrink_speed=self.shrink_speed,
            catastrophe_prob=self.catastrophe_prob,
            rescue_prob=self.rescue_prob,
            max_pause_at_min_frames=self.max_pause_at_min_frames,
            jitter_px=self.jitter_px,
            minus_end_velocity=self.minus_end_velocity,
            color_mode=self.color_mode,
            annotation_color_rgb=self.annotation_color_rgb,
            um_per_pixel=self.um_per_pixel,
            scale_bar_um=self.scale_bar_um,
            show_time=self.show_time,
            show_scale=self.show_scale,
            generate_mt_mask=self.generate_mt_mask,
            generate_seed_mask=self.generate_seed_mask,
            albumentations=None,
            margin=self.margin,
            microtubule_seed_min_dist=self.microtubule_seed_min_dist,
            **suggested_params,
        )

        try:
            synth_cfg.validate()
        except ValueError as e:
            logger.warning(
                f"SyntheticDataConfig for trial {trial.number} failed validation: {e}. Pruning trial."
            )
            raise

        return synth_cfg

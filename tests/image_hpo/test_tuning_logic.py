import pytest

from synth_mt.config.spots import SpotConfig
from synth_mt.config.synthetic_data import SyntheticDataConfig
from synth_mt.config.tuning import TuningConfig


# A simple mock object that simulates optuna.Trial's behavior
class MockOptunaTrial:
    def __init__(self, params):
        self.params = params
        self.number = params.get("number", 0)

    def suggest_float(self, name, low, high, *, log=False, step=None):
        return self.params.get(name)

    def suggest_int(self, name, low, high, step=1):
        return self.params.get(name)

    def suggest_categorical(self, name, choices):
        return self.params.get(name)


def test_create_config_from_trial():
    """
    Tests if TuningConfig can correctly create a SyntheticDataConfig
    using the new flexible rendering and bending parameters from a mocked Optuna trial.
    """
    tuning_cfg = TuningConfig()

    # Define a complete set of parameters matching the NEW suggest_synthetic_config_from_trial method.
    trial_params = {
        "number": 42,  # Example trial number

        # Kinematics
        "growth_speed": 2.0,
        "shrink_speed": 4.0,
        "catastrophe_prob": 0.01,
        "rescue_prob": 0.005,
        "pause_on_max_length": 5,
        "pause_on_min_length": 8,
        "min_length_min": 40, "min_length_max": 90,
        "microtubule_length_min": 100, "microtubule_length_max": 180,
        "bending_prob": 0.1,
        "um_per_pixel": 20,
        "scale_bar_um": 20,

        # Geometry & Shape
        "base_wagon_length_min": 15.0, "base_wagon_length_max": 45.0,
        "max_num_wagons": 10,
        "max_angle": 0.2,
        "max_angle_sign_changes": 1,
        "prob_to_flip_bend": 0.05,
        "min_wagon_length_min": 5, "min_wagon_length_max": 15,
        "max_wagon_length_min": 15, "max_wagon_length_max": 30,
        "num_microtubule": 25, "microtubule_seed_min_dist": 30, "margin": 10,

        # Rendering & Realism
        "psf_sigma_h": 0.4, "psf_sigma_v": 0.9,
        "tubule_width_variation": 0.1,
        "background_level": 0.7,
        "tubulus_contrast": -0.3,
        "seed_red_channel_boost": 0.6,
        "tip_brightness_factor": 1.2,
        "quantum_efficiency": 60.0,
        "gaussian_noise": 0.05,
        "jitter_px": 0.4,
        "vignetting_strength": 0.1,
        "global_blur_sigma": 0.8,

        # Spots
        "fixed_spots_count": 10, "fixed_spots_intensity_min": 0.05, "fixed_spots_intensity_max": 0.1,
        "fixed_spots_radius_min": 1, "fixed_spots_radius_max": 2, "fixed_spots_kernel_size_min": 1,
        "fixed_spots_kernel_size_max": 2, "fixed_spots_sigma": 0.5,
        "moving_spots_count": 5, "moving_spots_intensity_min": 0.05, "moving_spots_intensity_max": 0.1,
        "moving_spots_radius_min": 1, "moving_spots_radius_max": 2, "moving_spots_kernel_size_min": 1,
        "moving_spots_kernel_size_max": 2, "moving_spots_sigma": 0.5, "moving_spots_max_step": 4,
        "random_spots_count": 5, "random_spots_intensity_min": 0.05, "random_spots_intensity_max": 0.1,
        "random_spots_radius_min": 1, "random_spots_radius_max": 2, "random_spots_kernel_size_min": 1,
        "random_spots_kernel_size_max": 2, "random_spots_sigma": 0.5,

        "fixed_spots_polygon_p": 0.3, "fixed_spots_polygon_vertex_count_min": 3, "fixed_spots_polygon_vertex_count_max": 7,
        "moving_spots_polygon_p": 0.3, "moving_spots_polygon_vertex_count_min": 3, "moving_spots_polygon_vertex_count_max": 7,
        "random_spots_polygon_p": 0.3, "random_spots_polygon_vertex_count_min": 3, "random_spots_polygon_vertex_count_max": 7,

        "max_pause_at_min_frames": 5,
        "tail_wagon_length": 10.0,
        "red_channel_noise_std": 0.01,
    }
    mock_trial = MockOptunaTrial(trial_params)

    # Run the method we want to test
    synth_cfg = tuning_cfg.suggest_synthetic_config_from_trial(mock_trial)

    # Assert that the structure is correct and new values are populated
    assert isinstance(synth_cfg, SyntheticDataConfig)
    assert synth_cfg.tubulus_contrast == -0.3
    assert synth_cfg.seed_red_channel_boost == 0.6
    assert synth_cfg.max_angle_sign_changes == 1
    assert synth_cfg.prob_to_flip_bend == 0.05

    # Assert that nested objects are still created correctly
    assert isinstance(synth_cfg.fixed_spots, SpotConfig)
    assert synth_cfg.fixed_spots.count == 10
    assert synth_cfg.moving_spots.max_step == 0  # Default value for moving spots
import pytest

from synth_mt.config.spots import SpotTuningConfig
from synth_mt.config.synthetic_data import SyntheticDataConfig
from synth_mt.config.tuning import TuningConfig

def test_json_io(shared_tmp_path):
    config = SyntheticDataConfig()
    config_path = shared_tmp_path / "test.json"
    config.to_json(config_path)

    loaded_config = SyntheticDataConfig.from_json(config_path)
    assert config == loaded_config

def test_update_error():
    """Tests that updating with an invalid key raises a KeyError."""
    config = SyntheticDataConfig()
    with pytest.raises(KeyError):
        config.update({"invalid_key": 123})



def test_nested_config_io(shared_tmp_path):
    """
    Tests that a nested config (TuningConfig) can be saved and loaded correctly.
    This implicitly tests the recursive logic in the refactored BaseConfig.
    """
    # Create a config with nested SpotTuningConfig objects
    config = TuningConfig()
    config.fixed_spots_tuning.count_range = (10, 20)  # Make a change to verify
    config_path = shared_tmp_path / "nested_tuning_config.json"

    config.to_json(config_path)
    loaded_config = TuningConfig.from_json(config_path)

    # Assert that the top-level and nested objects are equal
    assert config == loaded_config
    assert isinstance(loaded_config.fixed_spots_tuning, SpotTuningConfig)
    assert tuple(loaded_config.fixed_spots_tuning.count_range) == (10, 20)


def test_nested_update(shared_tmp_path):
    """Tests that the .update() method works correctly with nested dictionaries."""
    config = TuningConfig()

    # Define an override dictionary with a nested structure
    overrides = {
        "num_trials": 999,
        "fixed_spots_tuning": {
            "count_range": [50, 60],  # Use lists, as they come from JSON/YAML
            "sigma_range": [0.1, 0.2]
        }
    }

    config.update(overrides)

    # Assert that both top-level and nested values were updated
    assert config.num_trials == 999
    assert tuple(config.fixed_spots_tuning.count_range) == (50, 60)
    assert tuple(config.fixed_spots_tuning.sigma_range) == (0.1, 0.2)
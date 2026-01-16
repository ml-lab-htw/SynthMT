import numpy as np
import pytest

from synth_mt.config.spots import SpotConfig
from synth_mt.data_generation.spots import SpotGenerator


def test_spot_generator_initialization():
    """Tests that the generator initializes the correct number of spots."""
    spot_cfg = SpotConfig(count=10)
    img_shape = (100, 100)
    generator = SpotGenerator(spot_cfg, img_shape)

    assert generator.n_spots == 10
    assert len(generator.coords) == 10
    assert len(generator.intensities) == 10
    assert all(0 <= y < 100 and 0 <= x < 100 for y, x in generator.coords)


def test_moving_spots_update():
    """Tests that the update method moves the spots."""
    spot_cfg = SpotConfig(count=5, max_step=10)
    img_shape = (100, 100)
    generator = SpotGenerator(spot_cfg, img_shape)

    initial_coords = list(generator.coords)
    generator.update()
    updated_coords = list(generator.coords)

    assert initial_coords != updated_coords


def test_fixed_spots_no_update():
    """Tests that update does nothing if max_step is None."""
    spot_cfg = SpotConfig(count=5, max_step=None)
    img_shape = (100, 100)
    generator = SpotGenerator(spot_cfg, img_shape)

    initial_coords = list(generator.coords)
    generator.update()
    updated_coords = list(generator.coords)

    assert initial_coords == updated_coords

# TODO
# def test_generator_apply_calls_draw_on_rgb(mocker):
#     """
#     Tests that apply calls the drawing function with the correct state on an RGB image.
#     """
#     # Mock the external draw_spots function
#     mock_draw = mocker.patch("synth_mt.data_generation.spots.draw_spots")
#
#     spot_cfg = SpotConfig(count=5, sigma=1.23)
#     img_shape = (10, 10)
#     generator = SpotGenerator(spot_cfg, img_shape)
#
#     img_in = np.zeros((10, 10, 3))
#
#     generator.apply(img_in)
#
#     # Assert that our mocked function was called once
#     mock_draw.assert_called_once()
#
#     # Assert it was called with the correct arguments from the generator's state
#     args, kwargs = mock_draw.call_args
#
#     assert np.array_equal(args[0], img_in)
#     assert np.array_equal(args[1], generator.coords)
#     assert np.array_equal(args[2], generator.intensities)
#     assert kwargs["sigma"] == generator.sigma
#     assert kwargs["polygon_p"] == generator.polygon_p
#     assert kwargs["polygon_vertex_count_min"] == generator.polygon_vertex_count_min
#     assert kwargs["polygon_vertex_count_max"] == generator.polygon_vertex_count_max
#     assert kwargs["kernel_size_min"] == generator.kernel_size_min
#     assert kwargs["kernel_size_max"] == generator.kernel_size_max

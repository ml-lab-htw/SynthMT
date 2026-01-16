import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from synth_mt.config.synthetic_data import SyntheticDataConfig
from synth_mt.file_io import OutputManager


@pytest.fixture
def mock_cfg():
    """Fixture for a mock SyntheticDataConfig."""
    cfg = MagicMock(spec=SyntheticDataConfig)
    cfg.id = "test_series"
    cfg.img_size = (64, 64)
    cfg.num_frames = 2
    cfg.fps = 1
    cfg.generate_mt_mask = True
    cfg.generate_seed_mask = True
    cfg.save = MagicMock()
    return cfg


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture for a temporary output directory."""
    return str(tmp_path / "output")


@patch("synth_mt.file_io.writers.tifffile")
@patch("synth_mt.file_io.writers.cv2")
@patch("synth_mt.file_io.writers.imageio")
def test_output_manager_initialization_and_paths(
    mock_imageio, mock_cv2, mock_tifffile, mock_cfg, temp_output_dir
):
    """Test that OutputManager initializes and creates all expected directories."""
    manager = OutputManager(cfg=mock_cfg, base_output_dir=temp_output_dir)

    # Check that all directories are created
    for path in manager.paths.values():
        if path.endswith("_dir"):
            assert os.path.isdir(path)
        else:
            assert os.path.isdir(os.path.dirname(path))

    # Check that config is saved
    mock_cfg.save.assert_called_once()


@patch("synth_mt.file_io.writers.imageio.get_writer")
@patch("synth_mt.file_io.writers.cv2.VideoWriter")
@patch("synth_mt.file_io.writers.tifffile.TiffWriter")
@pytest.mark.parametrize(
    "kwargs, tiff_calls, cv2_calls, imageio_calls",
    [
        (
            {
                "write_video_tiff": True,
                "write_video_masks_tiff": False,
                "write_video_mp4": False,
                "write_video_masks_mp4": False,
                "write_video_gif": False,
                "write_video_masks_gif": False,
            },
            1,
            0,
            0,
        ),
        (
            {
                "write_video_tiff": False,
                "write_video_masks_tiff": True,
                "write_video_mp4": False,
                "write_video_masks_mp4": False,
                "write_video_gif": False,
                "write_video_masks_gif": False,
            },
            2,
            0,
            0,
        ),
        (
            {
                "write_video_tiff": False,
                "write_video_masks_tiff": False,
                "write_video_mp4": True,
                "write_video_masks_mp4": False,
                "write_video_gif": False,
                "write_video_masks_gif": False,
            },
            0,
            1,
            0,
        ),
        (
            {
                "write_video_tiff": False,
                "write_video_masks_tiff": False,
                "write_video_mp4": False,
                "write_video_masks_mp4": True,
                "write_video_gif": False,
                "write_video_masks_gif": False,
            },
            0,
            2,
            0,
        ),
        (
            {
                "write_video_tiff": False,
                "write_video_masks_tiff": False,
                "write_video_mp4": False,
                "write_video_masks_mp4": False,
                "write_video_gif": True,
                "write_video_masks_gif": False,
            },
            0,
            0,
            1,
        ),
        (
            {
                "write_video_tiff": False,
                "write_video_masks_tiff": False,
                "write_video_mp4": False,
                "write_video_masks_mp4": False,
                "write_video_gif": False,
                "write_video_masks_gif": True,
            },
            0,
            0,
            2,
        ),
    ],
)
def test_output_manager_selective_writers(
    mock_tiff_writer,
    mock_cv2_writer,
    mock_imageio_writer,
    kwargs,
    tiff_calls,
    cv2_calls,
    imageio_calls,
    mock_cfg,
    temp_output_dir,
):
    """Test that writers are only initialized if their flags are True."""
    OutputManager(cfg=mock_cfg, base_output_dir=temp_output_dir, **kwargs)

    assert mock_tiff_writer.call_count == tiff_calls
    assert mock_cv2_writer.call_count == cv2_calls
    assert mock_imageio_writer.call_count == imageio_calls


@patch("synth_mt.file_io.writers.tifffile")
@patch("synth_mt.file_io.writers.cv2")
@patch("synth_mt.file_io.writers.imageio")
def test_append_calls_tiff_writers(mock_imageio, mock_cv2, mock_tifffile, mock_cfg, temp_output_dir):
    """Test that the append method calls video tiff writer and buffers masks."""
    mock_tifffile.TiffWriter.side_effect = [MagicMock(), MagicMock(), MagicMock()]
    manager = OutputManager(
        cfg=mock_cfg,
        base_output_dir=temp_output_dir,
        write_video_mp4=False,
        write_video_masks_mp4=False,
        write_video_gif=False,
        write_video_masks_gif=False,
        write_image_pngs=False,
        write_image_tiff=False,
        write_image_masks_tiff=False,
        write_image_masks_png=False,
    )
    frame_img = np.zeros((*mock_cfg.img_size, 3), dtype=np.uint8)
    mt_mask = np.zeros((5, *mock_cfg.img_size), dtype=np.uint16)
    seed_mask = np.zeros((5, *mock_cfg.img_size), dtype=np.uint16)

    manager.append(0, frame_img, mt_mask, seed_mask, [], export_current_frame=False)

    # Video is written directly
    manager.writers["video_tiff"].write.assert_called_once_with(frame_img)

    # Masks are buffered
    assert len(manager.all_mt_masks) == 1
    assert np.array_equal(manager.all_mt_masks[0], mt_mask)
    assert len(manager.all_seed_masks) == 1
    assert np.array_equal(manager.all_seed_masks[0], seed_mask)


@patch("synth_mt.file_io.writers.tifffile")
@patch("synth_mt.file_io.writers.cv2")
@patch("synth_mt.file_io.writers.imageio")
def test_append_calls_mp4_writers(mock_imageio, mock_cv2, mock_tifffile, mock_cfg, temp_output_dir):
    """Test that the append method calls mp4 writers."""
    mock_cv2.VideoWriter.side_effect = [MagicMock(), MagicMock(), MagicMock()]
    manager = OutputManager(
        cfg=mock_cfg,
        base_output_dir=temp_output_dir,
        write_video_tiff=False,
        write_video_masks_tiff=False,
        write_video_gif=False,
        write_video_masks_gif=False,
        write_image_pngs=False,
        write_image_tiff=False,
        write_image_masks_tiff=False,
        write_image_masks_png=False,
    )
    frame_img = np.zeros((*mock_cfg.img_size, 3), dtype=np.uint8)
    mt_mask = np.zeros((5, *mock_cfg.img_size), dtype=np.uint16)
    seed_mask = np.zeros((5, *mock_cfg.img_size), dtype=np.uint16)

    manager.append(0, frame_img, mt_mask, seed_mask, [], export_current_frame=False)

    manager.writers["video_mp4"].write.assert_called_once()
    manager.writers["mt_mask_mp4"].write.assert_called_once()
    manager.writers["seed_mask_mp4"].write.assert_called_once()


@patch("synth_mt.file_io.writers.tifffile")
@patch("synth_mt.file_io.writers.cv2")
@patch("synth_mt.file_io.writers.imageio")
def test_append_calls_gif_writers(mock_imageio, mock_cv2, mock_tifffile, mock_cfg, temp_output_dir):
    """Test that the append method calls gif writers."""
    mock_imageio.get_writer.side_effect = [MagicMock(), MagicMock(), MagicMock()]
    manager = OutputManager(
        cfg=mock_cfg,
        base_output_dir=temp_output_dir,
        write_video_tiff=False,
        write_video_masks_tiff=False,
        write_video_mp4=False,
        write_video_masks_mp4=False,
        write_image_pngs=False,
        write_image_tiff=False,
        write_image_masks_tiff=False,
        write_image_masks_png=False,
    )
    frame_img = np.zeros((*mock_cfg.img_size, 3), dtype=np.uint8)
    mt_mask = np.zeros((5, *mock_cfg.img_size), dtype=np.uint16)
    seed_mask = np.zeros((5, *mock_cfg.img_size), dtype=np.uint16)

    manager.append(0, frame_img, mt_mask, seed_mask, [], export_current_frame=False)

    manager.writers["video_gif"].append_data.assert_called_once_with(frame_img)
    manager.writers["mt_mask_gif"].append_data.assert_called_once()
    manager.writers["seed_mask_gif"].append_data.assert_called_once()


@patch("synth_mt.file_io.writers.tifffile")
@patch("synth_mt.file_io.writers.cv2")
@patch("synth_mt.file_io.writers.imageio")
def test_append_calls_single_image_writers(
    mock_imageio, mock_cv2, mock_tifffile, mock_cfg, temp_output_dir
):
    """Test that the append method calls single image writers."""
    manager = OutputManager(
        cfg=mock_cfg,
        base_output_dir=temp_output_dir,
        write_video_tiff=False,
        write_video_masks_tiff=False,
        write_video_mp4=False,
        write_video_masks_mp4=False,
        write_video_gif=False,
        write_video_masks_gif=False,
    )
    frame_img = np.zeros((*mock_cfg.img_size, 3), dtype=np.uint8)
    mt_mask = np.zeros((5, *mock_cfg.img_size), dtype=np.uint16)
    seed_mask = np.zeros((5, *mock_cfg.img_size), dtype=np.uint16)

    manager.append(0, frame_img, mt_mask, seed_mask, [], export_current_frame=True)

    assert mock_imageio.imwrite.call_count > 0

    # Check tifffile calls
    tiff_calls = mock_tifffile.imwrite.call_args_list
    expected_tiff_paths = [
        os.path.join(
            temp_output_dir, "images_tiff", "series_test_series_frame_0000.tif"
        ),
        os.path.join(
            temp_output_dir, "image_masks", "series_test_series_frame_0000.tif"
        ),
        os.path.join(
            temp_output_dir,
            "image_masks_seed",
            "series_test_series_frame_0000.tif",
        ),
    ]

    called_tiff_paths = [c.args[0] for c in tiff_calls]
    assert all(p in called_tiff_paths for p in expected_tiff_paths)

    # Check that the correct data is being written
    assert any(
        c.args[0] == expected_tiff_paths[0] and np.array_equal(c.args[1], frame_img)
        for c in tiff_calls
    )
    assert any(
        c.args[0] == expected_tiff_paths[1]
        and np.array_equal(c.args[1], mt_mask.astype(np.uint16))
        for c in tiff_calls
    )
    assert any(
        c.args[0] == expected_tiff_paths[2]
        and np.array_equal(c.args[1], seed_mask.astype(np.uint16))
        for c in tiff_calls
    )


@patch("synth_mt.file_io.writers.tifffile")
@patch("synth_mt.file_io.writers.cv2")
@patch("synth_mt.file_io.writers.imageio")
def test_append_gt_data(mock_imageio, mock_cv2, mock_tifffile, mock_cfg, temp_output_dir):
    """Test that ground truth data is appended correctly."""
    manager = OutputManager(cfg=mock_cfg, base_output_dir=temp_output_dir)
    gt_data = [{"frame_index": 0}]

    manager.append(0, None, None, None, gt_data, export_current_frame=False)

    assert manager.all_gt[0] == gt_data


@patch("synth_mt.file_io.writers.tifffile")
@patch("synth_mt.file_io.writers.cv2.VideoWriter")
@patch("synth_mt.file_io.writers.imageio.get_writer")
def test_close_method(
    mock_imageio_writer, mock_cv2_writer, mock_tifffile, mock_cfg, temp_output_dir
):
    """Test that the close method calls close on all writers and saves GT."""
    # Ensure each call to a writer class returns a new mock
    mock_tiff_writer_instance = MagicMock()
    mock_cv2_writer_instance = MagicMock()
    mock_imageio_writer_instance = MagicMock()

    # We only have one of each non-mask writer in this setup
    mock_tifffile.TiffWriter.return_value = mock_tiff_writer_instance
    mock_cv2_writer.return_value = mock_cv2_writer_instance
    mock_imageio_writer.return_value = mock_imageio_writer_instance

    manager = OutputManager(cfg=mock_cfg, base_output_dir=temp_output_dir)
    manager.all_gt = [{"test": "data"}]
    manager.all_mt_masks = [np.zeros((5, *mock_cfg.img_size), dtype=np.uint16)]
    manager.all_seed_masks = [np.zeros((5, *mock_cfg.img_size), dtype=np.uint16)]

    with patch("builtins.open", new_callable=MagicMock) as mock_open, patch(
        "json.dump"
    ) as mock_json_dump:
        manager.close()

        # Check that buffered masks are written
        # We can't use assert_any_call directly with numpy arrays due to ambiguity.
        # Instead, we check the calls manually.
        mt_mask_written = False
        seed_mask_written = False
        expected_mt_mask = np.stack(manager.all_mt_masks, axis=0)
        expected_seed_mask = np.stack(manager.all_seed_masks, axis=0)

        for c in mock_tifffile.imwrite.call_args_list:
            path, data = c.args
            if path == manager.paths["mt_masks_tiff"]:
                if np.array_equal(data, expected_mt_mask):
                    mt_mask_written = True
            elif path == manager.paths["seed_masks_tiff"]:
                if np.array_equal(data, expected_seed_mask):
                    seed_mask_written = True

        assert mt_mask_written, "MT mask TIFF was not written correctly."
        assert seed_mask_written, "Seed mask TIFF was not written correctly."

        # Check that other writers are closed
        mock_tiff_writer_instance.close.assert_called_once()
        mock_cv2_writer_instance.release.assert_called()
        mock_imageio_writer_instance.close.assert_called()

        # Check that GT file is saved
        mock_open.assert_called_once_with(manager.paths["ground_truth_file"], "w")
        mock_json_dump.assert_called_once()

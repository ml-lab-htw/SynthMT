import os
import pytest

# Skip all tests in this module if HUGGING_FACE_HUB_TOKEN is not set
if not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    pytest.skip("HUGGING_FACE_HUB_TOKEN not set", allow_module_level=True)

import numpy as np

from synth_mt.config.tuning import TuningConfig
from synth_mt.data_generation.optimization.embeddings import ImageEmbeddingExtractor


@pytest.fixture
def tiny_model_tuning_config(shared_tmp_path):
    """A fixture that provides a TuningConfig pointing to a tiny, fast model."""
    # Create a dummy video file that extract_frames can process
    dummy_video_path = shared_tmp_path / "ref.mp4"
    dummy_frame = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)

    import cv2

    # Create a dummy mp4 video with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dummy_video_path), fourcc, 1.0, (512, 512))
    out.write(dummy_frame)
    out.release()

    # Use a fast, small model for testing
    return TuningConfig(
        model_name="openai/clip-vit-base-patch16",
        reference_video_path=str(dummy_video_path),
        num_compare_frames=1,
        pca_components=3,  # Enable PCA for one of the tests
    )


def test_extractor_initialization(tiny_model_tuning_config):
    """Tests that the extractor can be initialized without errors."""
    extractor = ImageEmbeddingExtractor(tiny_model_tuning_config)
    assert extractor.model is not None
    assert extractor.processor is not None


def test_no_pca(tiny_model_tuning_config, mocker):
    """Tests that if PCA is disabled, the dimension is the model's original one."""
    tiny_model_tuning_config.pca_components = None  # Disable PCA
    dummy_frame = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    mocker.patch("synth_mt.file_io.utils.extract_frames", return_value=([dummy_frame], None))

    extractor = ImageEmbeddingExtractor(tiny_model_tuning_config)
    ref_embeddings = extractor.extract_from_references()

    assert extractor.pca_model is None
    assert ref_embeddings.shape[1] > 3  # The original dimension is much larger


import logging

import cv2
import numpy as np
from cellpose.transforms import smooth_sharpen_img

logger = logging.getLogger(__name__)

# Optional import for histogram normalization
try:
    from cellSAM.utils import _histogram_normalization

    HAS_CELLSAM = True
except ImportError:
    HAS_CELLSAM = False
    logger.debug("cellSAM not available, histogram_normalization will be disabled")


def sample_to_arrays(sample):
    """Convert a dataset sample to numpy arrays."""
    image = np.array(sample["image"].convert("RGB"), dtype=np.uint8)
    masks = sample["mask"]
    gt_masks = np.stack([np.array(m.convert("L")) for m in masks], axis=0)
    return image, gt_masks


def get_preprocess_params(model):
    """Extract preprocessing parameters from the model configuration."""
    return {
        "grayscale": model.grayscale,
        "sharpen_radius": model.sharpen_radius,
        "smooth_radius": model.smooth_radius,
        "percentile_min": model.percentile_min,
        "percentile_max": model.percentile_max,
        "clip_to_percentiles": model.clip_to_percentiles,
        "rescale_using_percentiles": model.rescale_using_percentiles,
        "invert": model.invert,
        "histogram_normalization": model.histogram_normalization,
    }


def process_image(
    image,
    grayscale: bool = True,
    sharpen_radius=0.0,
    smooth_radius=0.0,
    percentile_min=0.0,
    percentile_max=100.0,
    clip_to_percentiles=False,
    rescale_using_percentiles=False,
    invert=False,
    histogram_normalization=False,
):
    """
    Preprocesses an image with intensity normalization, inversion, sharpening, and smoothing.
    The default values lead to no changes being made to the image.
    The function is heavily inspired by the `normalize_img` function from cellpose/transforms.py,
    see https://github.com/MouseLand/cellpose/blob/0a3de7b6420470c1d1ccf33ba6f4bc821fe532cc/cellpose/transforms.py#L644C5-L644C18

    Args:
        image (np.ndarray): The input image with dtype uint8.
        grayscale (bool, optional): Whether to convert the image to grayscale. Defaults to True.
        sharpen_radius (float, optional): The radius for sharpening the image. Defaults to 0.
        smooth_radius (float, optional): The radius for smoothing the image. Defaults to 0.
        percentile_min (float): The minimum percentile for normalization. Should be between 0 and 100.
        percentile_max (float): The maximum percentile for normalization. Should be between 0 and 100.
        clip_to_percentiles (bool): Whether to clip the image values to the percentile range. Defaults to False.
        rescale_using_percentiles (bool): Whether to rescale the image using the percentile range. Defaults to False.
        invert (bool): Whether to invert the image. Useful if cells are dark instead of bright. Defaults to False.
        histogram_normalization (bool): Whether to apply histogram normalization as in cellSAM. Defaults to False.
    Returns:
        np.ndarray: The preprocessed image with dtype float32.
    """
    if not image.dtype == np.uint8:
        raise ValueError("Input image must have dtype uint8")

    # Valicdate percentile
    if not (0 <= percentile_min < percentile_max <= 100):
        raise ValueError("Invalid percentile range, should be between 0 and 100")

    # Validate further arguments
    if (
        (sharpen_radius > 0 or smooth_radius > 0)
        and (percentile_min == 0 and percentile_max == 0)
        and invert
    ):
        raise ValueError(
            "If sharpening or smoothing together with inverting is requested, percentile_min and "
            "percentile_max must not both be zero beacuse inverting becomes ambiguous."
        )

    processed_image = image.copy().astype(np.float32)

    # Convert to grayscale if requested
    if grayscale and processed_image.ndim == 3 and processed_image.shape[-1] > 1:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    # Check if we will do anything at all
    if (
        sharpen_radius == 0.0
        and smooth_radius == 0.0
        and not clip_to_percentiles
        and not rescale_using_percentiles
        and not invert
        and not histogram_normalization
    ):
        return processed_image

    nchan = image.shape[-1]

    # Apply sharpening and smoothing as specified
    if sharpen_radius > 0 or smooth_radius > 0:
        processed_image = smooth_sharpen_img(
            processed_image, sharpen_radius=sharpen_radius, smooth_radius=smooth_radius
        )

    # Apply intensity normalization and inversion as specified
    if percentile_min > 0 or percentile_max < 100:
        for c in range(nchan):
            if np.ptp(processed_image[..., c]) > 0.0:
                processed_image_slice = processed_image[..., c]
                processed_image_lower = np.percentile(processed_image_slice, percentile_min)
                processed_image_upper = np.percentile(processed_image_slice, percentile_max)
                if clip_to_percentiles:
                    processed_image[..., c] = np.clip(
                        processed_image[..., c], processed_image_lower, processed_image_upper
                    )
                if rescale_using_percentiles:
                    rescaler = processed_image_upper - processed_image_lower
                    if rescaler > 1e-3:
                        processed_image[..., c] -= processed_image_lower
                        processed_image[..., c] /= rescaler
                    else:
                        processed_image[..., c] = 0
        if invert:
            processed_image = 1 - processed_image
    elif invert:
        processed_image = 255 - processed_image

    if histogram_normalization:
        if not HAS_CELLSAM:
            logger.warning(
                "histogram_normalization requested but cellSAM is not installed. Skipping."
            )
        else:
            processed_image = _histogram_normalization(processed_image)

    return processed_image

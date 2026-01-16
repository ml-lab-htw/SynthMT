import numpy as np


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

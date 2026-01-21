import cv2
import numpy as np
from scipy.interpolate import splprep, splev


def sample_to_arrays(sample):
    """Convert a dataset sample to numpy arrays."""
    image = np.array(sample["image"].convert("RGB"), dtype=np.uint8)
    masks = sample["mask"]
    gt_masks = np.stack([np.array(m.convert("L")) for m in masks], axis=0)
    return image, gt_masks


def anchor_points_to_instance_mask_stack(ordered_coords_list, shape, thickness=2):
    """
    Convert a list of ordered anchor points to a stack of binary masks.
    Each mask is created by drawing a spline through the anchor points.
    """

    mask_stack = []
    for ordered_coords in ordered_coords_list:
        if len(ordered_coords) == 0:
            # No anchor points, skip this instance
            continue
        mask = np.zeros(shape, dtype=np.uint8)
        if len(ordered_coords) == 1:
            # Draw a single point if only one anchor
            pt = np.round(ordered_coords[0]).astype(int)
            mask[int(pt[1]), int(pt[0])] = 1
            mask_stack.append(mask)
            continue
        try:
            tck, u = splprep(ordered_coords.T, s=0)
            u_fine = np.linspace(0, 1, 200)
            spline_points = np.array(splev(u_fine, tck)).T
        except Exception:
            spline_points = ordered_coords

        # Draw the anchor points onto the mask
        for pt in np.round(ordered_coords).astype(int):
            x = np.clip(pt[0], 0, shape[1] - 1)
            y = np.clip(pt[1], 0, shape[0] - 1)
            mask[y, x] = 1

        # Draw the spline points as a polyline
        pts = np.round(spline_points).astype(int)
        for i in range(len(pts) - 1):
            p1 = tuple(pts[i])
            p2 = tuple(pts[i + 1])
            cv2.line(mask, p1, p2, 1, thickness)
        mask_stack.append(mask)
    return np.array(mask_stack)

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

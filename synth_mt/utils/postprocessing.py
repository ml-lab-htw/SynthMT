import logging

import numpy as np

from synth_mt.benchmark import metrics
from synth_mt.benchmark.dataset import BenchmarkDataset
from synth_mt.benchmark.metrics import as_instance_stack

logger = logging.getLogger(__name__)


def get_area_length_ranges(dataset: BenchmarkDataset):
    all_areas = []
    all_lengths = []
    image_resolution = None
    border_margin = 1  # set to -1 to include border instances

    for im, gt_mask, _ in dataset:
        if image_resolution is None and gt_mask is not None:
            image_resolution = min(gt_mask.shape[0], gt_mask.shape[1])
        if gt_mask is not None and gt_mask.max() > 0:
            instance_masks = as_instance_stack(gt_mask)
            for i in range(instance_masks.shape[0]):
                instance_mask = instance_masks[i]
                props = get_instance_properties(instance_mask)

                # check if the instance is at the border
                indices = np.argwhere(instance_mask)
                min_row, min_col = indices[:, 0].min(), indices[:, 1].min()
                max_row, max_col = indices[:, 0].max(), indices[:, 1].max()
                height, width = instance_mask.shape
                if (
                    min_row > border_margin
                    and min_col > border_margin
                    and max_row < height - border_margin
                    and max_col < width - border_margin
                ):
                    all_areas.append(props["area"])
                    all_lengths.append(props["length"])

    if image_resolution is None:
        return 0, 1.0, 0, 1.0  # Default if no masks found

    min_area = min(all_areas) if all_areas else 0
    max_area = max(all_areas) if all_areas else image_resolution
    min_length = min(all_lengths) if all_lengths else 0
    max_length = max(all_lengths) if all_lengths else image_resolution

    logger.debug(f"Area range: {min_area:.0f}px - {max_area:.0f}px")
    logger.debug(f"Length range: {min_length:.0f}px - {max_length:.0f}px")

    return min_area, max_area, min_length, max_length


def get_instance_properties(instance_mask):
    """Calculate properties of a single instance mask."""
    area = np.sum(instance_mask)
    length = metrics._get_instance_mask_length(instance_mask)
    return {"area": area, "length": length}


def filter_instance_masks(
    masks,
    min_area=0,
    max_area=np.inf,
    min_length=0,
    max_length=np.inf,
    border_margin=-1,
    **kwargs,
):
    """Filter instance masks based on area, length, and border conditions."""

    instance_masks = as_instance_stack(masks)
    filtered_instance_masks = []

    for i in range(instance_masks.shape[0]):
        curr_mask = instance_masks[i]

        if not np.any(curr_mask):
            continue

        height, width = instance_masks.shape[1:3]

        # Area condition (summed pixels)
        props = get_instance_properties(curr_mask)
        area = props["area"]
        if min_area <= area <= max_area:

            # Border condition (no touching border)
            indices = np.argwhere(curr_mask)
            min_row, min_col = indices[:, 0].min(), indices[:, 1].min()
            max_row, max_col = indices[:, 0].max(), indices[:, 1].max()
            if (
                min_row > border_margin
                and min_col > border_margin
                and max_row < height - border_margin
                and max_col < width - border_margin
            ):

                # Length condition
                length = props["length"]
                if min_length <= length <= max_length:
                    filtered_instance_masks.append(curr_mask)

    if not filtered_instance_masks:
        return np.zeros((0, *instance_masks.shape[1:]), dtype=instance_masks.dtype)
    else:
        return np.stack(filtered_instance_masks, axis=0)


def filter_anchor_points(
    anchor_points,
    min_length=0,
    max_length=np.inf,
    image_resolution=None,
    border_margin=-1,
    **kwargs,
):
    """Filter anchor points based on length and border conditions."""

    filtered_anchor_points = []

    for points in anchor_points:
        if points.shape[0] == 0:
            continue

        # Length condition
        length = metrics._get_ordered_anchor_points_length(points)
        if min_length <= length <= max_length:

            if image_resolution is not None:
                x_min = points[:, 0].min()
                x_max = points[:, 0].max()
                y_min = points[:, 1].min()
                y_max = points[:, 1].max()
                size_x, size_y = image_resolution

                # Border condition (no touching border)
                if (
                    x_min <= border_margin
                    or y_min <= border_margin
                    or x_max >= size_x - border_margin
                    or y_max >= size_y - border_margin
                ):
                    continue

                filtered_anchor_points.append(points)

    return filtered_anchor_points

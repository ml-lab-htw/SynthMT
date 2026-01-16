import logging
from typing import Tuple, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def create_overlay(image, masks):

    # Image as numpy array (H, W, 3)
    if isinstance(image, np.ndarray):
        img_array = image
    else:
        img_array = np.array(image.convert("RGB"))

    if masks is None:
        return img_array
    elif isinstance(masks, list) and len(masks) == 0:
        return img_array
    elif isinstance(masks, np.ndarray) and (masks.size == 0 or np.sum(masks) == 0):
        return img_array

    # Masks as stacked numpy array (N, H, W) where N is the number of instances
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            mask_stack = [masks]
        else:
            mask_stack = [masks[i] for i in range(masks.shape[0])]
    else:
        mask_stack = np.stack([np.array(mask.convert("L")) for mask in masks], axis=0)

    # Create a copy of the image for overlay
    overlay = img_array.copy().astype(float)

    # Get the "cool" colormap
    cmap = plt.cm.cool

    # Generate colors from the colormap
    num_masks = len(mask_stack)
    colors = [cmap(i / max(num_masks - 1, 1))[:3] for i in range(num_masks)]  # [:3] to get RGB only
    colors = np.array(colors) * 255  # Convert from [0,1] to [0,255]

    # Apply each mask with its color
    alpha = 0.75
    for i, mask in enumerate(mask_stack):
        mask_bool = mask > 0
        overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * colors[i]

    return overlay.astype(np.uint8)


def show_frame(frame: np.ndarray, title: str = "", figsize=(6, 4)) -> None:
    """
    Display a single frame using matplotlib.

    Args:
        frame (np.ndarray): The image frame to display, expected to be in RGB format for matplotlib.
        title (str): Title for the plot.
        figsize (tuple[int, int]): Figure size in inches.
    """

    plt.figure(figsize=figsize)  # Create a new figure

    if frame.dtype == np.uint8 and frame.shape[2] == 3:
        logger.debug("Frame is 3-channel, assuming BGR and converting to RGB for display.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.imshow(frame)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=False)


def get_colormap(
    all_track_ids: set[int], cmap_name: str = "tab20"
) -> dict[Any, Any] | dict[int, tuple[tuple[int, ...], ...]]:
    """
    Generates a consistent color map for track IDs.

    Args:
        all_track_ids (set[int]): Set of all unique track IDs.
        cmap_name (str): Matplotlib colormap name.

    Returns:
        dict[int, Tuple[int, int, int]]: A dictionary mapping track IDs to RGB colors (0-255).
    """
    if not all_track_ids:
        logger.warning("No track IDs provided for colormap generation. Returning empty map.")
        return {}

    cmap = plt.get_cmap(cmap_name)
    num_colors = cmap.N  # Number of distinct colors in the colormap

    sorted_ids = sorted(
        list(all_track_ids)
    )  # Convert set to list and sort for consistent assignment
    color_map = {
        track_id: tuple((np.array(cmap(i % num_colors)[:3]) * 255).astype(int))
        for i, track_id in enumerate(sorted_ids)
    }
    logger.debug("Colormap generated successfully.")
    return color_map



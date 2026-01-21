import logging
from typing import Tuple, Any, Literal, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from examples.utils import anchor_points_to_instance_mask_stack

logger = logging.getLogger(__name__)

def create_overlay(image, masks):

    is_anchor_point = isinstance(masks, list) and len(masks) > 0 and masks[0].shape[1] == 2

    if is_anchor_point:
        plotting_aps = [[ap_coords[0], ap_coords[-1]] for ap_coords in masks]
        masks = anchor_points_to_instance_mask_stack(masks, image.shape[:2])

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
    if is_anchor_point:
        alpha = 0.65
    else:
        alpha = 0.75

    for i, mask in enumerate(mask_stack):
        mask_bool = mask > 0
        overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * colors[i]

    if is_anchor_point:
        for ap_coords, col in zip(plotting_aps, colors):
            draw_points(
                overlay,
                points=[ap_coords[0], ap_coords[-1]],
                radius=5,
                color=tuple(col),
            )

    return overlay.astype(np.uint8)


def draw_points(
    image: np.ndarray,
    points: Iterable[Tuple[float, float]],
    radius: int,
    color: Tuple[int, int, int],
    alpha: float = 0.9,
    marker: Literal["circle", "x"] = "circle",
    thickness: int = 3,
    inplace: bool = True,
):
    """
    Draw points as circles or X markers with optional alpha blending.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, 3), BGR.
    points : iterable of (x, y)
        Point coordinates (float or int).
    radius : int
        Circle radius or half-size of X.
    color: (B, G, R)
        Marker color.
    alpha : float, optional
        Transparency in [0, 1]. 1.0 = opaque.
    marker: {"circle", "x"}, optional
        Marker type to draw.
    thickness : int, optional
        Line thickness for X marker.
    inplace : bool, optionally
        Modify image in-place.

    Returns
    -------
    np.ndarray
        Image with drawn markers.
    """

    if not inplace:
        image = image.copy()

    h, w = image.shape[:2]

    def _draw(img):
        for x_f, y_f in points:
            x = int(np.clip(round(x_f), 0, w - 1))
            y = int(np.clip(round(y_f), 0, h - 1))

            if marker == "circle":
                cv2.circle(img, (x, y), radius, color, -1)

            elif marker == "x":
                r = radius
                cv2.line(img, (x - r, y - r), (x + r, y + r), color, thickness)
                cv2.line(img, (x - r, y + r), (x + r, y - r), color, thickness)

            else:
                raise ValueError(f"Unknown marker type: {marker}")

    if alpha >= 1.0:
        _draw(image)
        return image

    overlay = image.copy()
    _draw(overlay)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


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



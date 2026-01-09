import numpy as np
from matplotlib import pyplot as plt


## Function for visualizing the mask as overlay on top of the image
def create_overlay(image, masks):

    # Image as numpy array (H, W, 3)
    if isinstance(image, np.ndarray):
        img_array = image
    else:
        img_array = np.array(image.convert("RGB"))

    # Masks as stacked numpy array (N, H, W) where N is the number of instances
    if isinstance(masks, np.ndarray):
        mask_stack = masks
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
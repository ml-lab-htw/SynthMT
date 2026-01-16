import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, Optional, Union, Any


import os
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
import seaborn as sns

from scipy.interpolate import splprep, splev
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from scipy.stats import entropy
from skimage.draw import line, disk
from skimage.morphology import skeletonize
import logging

sns.set_theme(
    style="darkgrid",  # or "darkgrid", "white", "ticks" – pick what you like
    context="paper",  # "paper", "notebook", "talk", "poster"
    palette="deep",  # or "muted", "bright", "colorblind", etc.
)

logger = logging.getLogger(__name__)


# -------------------------
# Utilities
# -------------------------


def fit_parametric_curve(
    points: np.ndarray,
    s: Optional[float] = 1.0,
    method: str = "spline",
    poly_degree: int = 3,
    k: int = 3,
) -> Tuple:
    """
    Fit a parametric curve to 2D points using either spline or polynomial fitting.

    Args:
        points: Array of shape (2, N) or (N, 2) containing x, y coordinates
        s: Smoothing factor for spline fitting (used only when method='spline')
        method: Either 'spline' or 'polyfit'
        poly_degree: Degree of polynomial for polyfit method
        k: Degree of the spline (default 3 for cubic spline). Must be <= 5.

    Returns:
        For method='spline': (tck, u) - spline parameters and parameter values
        For method='polyfit': (poly_params, u) - dictionary with polynomial coefficients and parameter values
    """
    # Ensure points are in shape (2, N)
    if points.shape[0] != 2:
        points = points.T

    if method == "spline":
        # Validate that we have enough points for the spline degree
        n_points = points.shape[1]

        # Automatically reduce k if we don't have enough points
        actual_k = min(k, n_points - 1)
        if actual_k < 1:
            raise ValueError(f"Need at least 2 points for spline fitting, got {n_points}")

        # Use scipy's spline fitting
        # print(f"points shape: {points.shape}, using k={actual_k}")
        tck, u = splprep(points, s=s, k=actual_k)
        return tck, u

    elif method == "polyfit":
        # Use polynomial fitting - fit x(t) and y(t) separately
        n_points = points.shape[1]
        u = np.linspace(0, 1, n_points)

        # Fit polynomials for both x and y as functions of parameter u
        x_coeffs = np.polyfit(u, points[0], poly_degree)
        y_coeffs = np.polyfit(u, points[1], poly_degree)

        # Store in a format similar to tck for compatibility
        poly_params = {
            "x_coeffs": x_coeffs,
            "y_coeffs": y_coeffs,
            "method": "polyfit",
            "degree": poly_degree,
        }
        return poly_params, u

    else:
        raise ValueError(f"Unknown method: {method}. Use 'spline' or 'polyfit'")


def eval_parametric_curve(
    u: np.ndarray, curve_params: Union[Tuple, Dict], der: int = 0
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Evaluate a parametric curve at given parameter values.

    Args:
        u: Parameter values to evaluate at
        curve_params: Either tck tuple (for spline) or dict with polynomial coefficients (for polyfit)
        der: Derivative order (0=position, 1=first derivative, 2=second derivative)

    Returns:
        For der=0: Tuple of (x, y) arrays or single array depending on format
        For der>0: Derivatives
    """
    if isinstance(curve_params, dict) and curve_params.get("method") == "polyfit":
        # Polynomial evaluation
        x_coeffs = curve_params["x_coeffs"]
        y_coeffs = curve_params["y_coeffs"]

        if der == 0:
            # Evaluate position
            x_vals = np.polyval(x_coeffs, u)
            y_vals = np.polyval(y_coeffs, u)
            return x_vals, y_vals

        elif der == 1:
            # First derivative
            x_der = np.polyder(x_coeffs, 1)
            y_der = np.polyder(y_coeffs, 1)
            x_vals = np.polyval(x_der, u)
            y_vals = np.polyval(y_der, u)
            return x_vals, y_vals

        elif der == 2:
            # Second derivative
            x_der = np.polyder(x_coeffs, 2)
            y_der = np.polyder(y_coeffs, 2)
            x_vals = np.polyval(x_der, u)
            y_vals = np.polyval(y_der, u)
            return x_vals, y_vals

        else:
            raise ValueError(f"Derivative order {der} not supported")

    else:
        # Spline evaluation - use splev
        return splev(u, curve_params, der=der)


def as_instance_stack(mask_or_stack: np.ndarray) -> np.ndarray:
    """
    Accept either a labeled mask (H, W) with background=0, or a stack (N, H, W).
    Return a boolean stack (N, H, W) of instance masks.
    """
    arr = np.asarray(mask_or_stack)
    if arr.ndim == 3:
        return arr.astype(bool)
    if arr.ndim != 2:
        raise ValueError(f"Expected (H,W) labeled mask or (N,H,W) stack, got {arr.shape}")
    ids = np.unique(arr)
    ids = ids[ids != 0]
    if ids.size == 0:  # Only background pixels found
        return np.zeros((0, arr.shape[0], arr.shape[1]), dtype=bool)
    return np.stack([arr == i for i in ids], axis=0).astype(bool)


def _compute_iou_matrix(
    gt_instance_masks: np.ndarray,
    pred_instance_masks: np.ndarray,
) -> np.ndarray:
    """
    Compute IoU matrix between GT and predicted instance masks.

    Args:
        gt_instance_masks  : (M, H, W) bool
        pred_instance_masks: (N, H, W) bool
    Returns:
        (M, N) float IoU matrix
    """
    M, N = len(gt_instance_masks), len(pred_instance_masks)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=float)

    # Ensure boolean, then use int64 accumulation to avoid overflow
    gm = gt_instance_masks.astype(bool)

    pm = pred_instance_masks.astype(bool)

    # intersections: (M, N)
    inter = (gm[:, None, :, :] & pm[None, :, :, :]).sum(axis=(2, 3), dtype=np.int64)

    true_areas = gm.sum(axis=(1, 2), dtype=np.int64)[:, None]  # (M, 1)
    test_areas = pm.sum(axis=(1, 2), dtype=np.int64)[None, :]  # (1, N)

    union = true_areas + test_areas - inter

    # replace NaNs with zeros (can happen if union is zero)
    return np.nan_to_num(inter / union, nan=0.0)


def _compute_iot_matrix(
    gt_instance_masks: np.ndarray,
    pred_instance_anchor_points: List[np.ndarray],
) -> np.ndarray:
    """
    Compute IoU matrix between GT and predicted instance masks.

    Args:
        gt_instance_masks  : (M, H, W) bool
        pred_instance_anchor_points: list of N arrays with (num_points, 2) coordinates
    Returns:
        (M, N) float IoU matrix
    """
    M, N = len(gt_instance_masks), len(pred_instance_anchor_points)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=float)

    # Ensure boolean, then use int64 accumulation to avoid overflow
    gm = gt_instance_masks.astype(bool)

    # Here, pred_instance_anchor_points is a list of anchor point arrays
    inter = np.zeros((M, N), dtype=np.int64)
    test_areas = np.zeros((1, N), dtype=np.int64)

    for n, anchor_points in enumerate(pred_instance_anchor_points):
        num_points = len(anchor_points)
        test_areas[0, n] = num_points
        if num_points == 0:
            continue

        # Clip coordinates to be within mask bounds to avoid index errors
        points = np.asarray(anchor_points).astype(int)
        points[:, 0] = np.clip(points[:, 0], 0, gm.shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, gm.shape[2] - 1)

        # Count how many points fall within each GT mask
        # gm[:, points[:, 0], points[:, 1]] creates a (M, num_points) boolean array
        inter[:, n] = np.sum(gm[:, points[:, 1], points[:, 0]], axis=1)

    # replace NaNs with zeros (can happen if union is zero)
    return np.nan_to_num(inter / test_areas, nan=0.0)


def _compute_skiou_matrix(
    gt_instance_masks: np.ndarray,
    pred_instance_masks: Optional[np.ndarray] = None,
    pred_instance_anchor_points: List[np.ndarray] = [],
    spline_s: Optional[float] = 0,
) -> np.ndarray:
    """
    Compute SKIoU matrix between GT and predicted instances. Distinguishes between predicted
    instances given in masks and anchor points format.

    Args:
        gt_instance_masks  : (M, H, W) bool
        pred_instance_masks: (N, H, W) bool
        pred_instance_anchor_points: list of N arrays with (num_points, 2) coordinates
        spline_s:
            Smoothing factor for splprep. Default of sklearn's splprep function is None; TARDIS
            uses 1. Even though we found worse results with s=0 and s=1 on the simple circle
            examples below, we choose s=0 because it preserves the geometry of the original anchor
            points most faithfully.
    Returns:
        (M, N) float SKIoU matrix
    """
    M = len(gt_instance_masks)
    N = (
        len(pred_instance_masks)
        if pred_instance_masks is not None
        else len(pred_instance_anchor_points)
    )
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=float)

    # Get skeletonized masks
    gt_skeletons = [skeletonize(instance) for instance in gt_instance_masks]
    mask_shape = gt_skeletons[0].shape
    pred_masks = []
    pred_skeletons = []
    if pred_instance_masks is not None:
        pred_masks = pred_instance_masks
        pred_skeletons = [skeletonize(instance) for instance in pred_instance_masks]
    elif len(pred_instance_anchor_points) > 0:
        for idx, anchor_points in enumerate(pred_instance_anchor_points):
            try:
                tck, u = fit_parametric_curve(anchor_points.T, s=spline_s, method="spline")
                n_points = 512  # Should suffice...
                u_fine = np.linspace(0, 1, n_points)
                points = np.array(eval_parametric_curve(u_fine, tck)).T.astype(int)
                points[:, 0] = np.clip(points[:, 0], 0, mask_shape[1] - 1)
                points[:, 1] = np.clip(points[:, 1], 0, mask_shape[0] - 1)
                # Draw points onto mask
                mask = np.zeros(mask_shape, dtype=bool)
                for i in range(n_points):
                    x, y = points[i]
                    mask[y, x] = True
                pred_masks.append(mask)
                pred_skeletons.append(skeletonize(mask))
            except (ValueError, TypeError) as e:
                # Skip instances with too few anchor points or other fitting issues
                logging.warning(
                    f"Skipping prediction instance {idx} with {len(anchor_points)} anchor points: {e}"
                )
                # Add an empty mask to maintain indexing
                empty_mask = np.zeros(mask_shape, dtype=bool)
                pred_masks.append(empty_mask)
                pred_skeletons.append(empty_mask)

    # Ensure boolean, then use int64 accumulation to avoid overflow
    gm = np.array(gt_instance_masks).astype(bool)
    gs = np.array(gt_skeletons).astype(bool)
    pm = np.array(pred_masks).astype(bool)
    ps = np.array(pred_skeletons).astype(bool)

    # intersections: (M,N,*mask_shape)
    inter = gm[:, None, :, :] & pm[None, :, :, :]
    # skeletonization
    inter_flat = inter.reshape(-1, *mask_shape)
    inters_flat = np.stack([skeletonize(im) for im in inter_flat], axis=0)
    # skeletonized intersections: (M, N)
    inters = inters_flat.sum(axis=(1, 2), dtype=np.int64).reshape(M, N)

    true_s = gs.sum(axis=(1, 2), dtype=np.int64)[:, None]  # (M, 1)
    test_s = ps.sum(axis=(1, 2), dtype=np.int64)[None, :]  # (1, N)

    union = true_s + test_s

    # replace NaNs with zeros (can happen if union is zero)
    return np.nan_to_num(2 * inters / union, nan=0.0)


def anchor_points_to_instance_masks(
    anchor_points_instance_masks: list, mask_shape: tuple, width: int = 1
) -> np.ndarray:
    """
    Convert a list of ordered anchor point arrays to a stack of instance segmentation masks.
    If width=0, it's simply a binary mask with 1 at the anchor points.
    Width=1 will connect the points by lines, and width>1 will use disks.

    Args:
        anchor_points_instance_masks: list of np.ndarray, each (N, 2) with (row, col) coordinates
        mask_shape: tuple, (H, W) shape of the output masks
        width: int, width of the line to draw.

    Returns:
        np.ndarray: (N_instances, H, W) boolean instance masks
    """
    masks = []
    for coords in anchor_points_instance_masks:
        mask = np.zeros(mask_shape, dtype=bool)
        if len(coords) < 2:
            masks.append(mask)
            continue
        for i in range(len(coords) - 1):
            c0, r0 = coords[i]
            c1, r1 = coords[i + 1]
            if width == 0:
                mask = draw_mask_points(mask, int(r0), int(c0))
                mask = draw_mask_points(mask, int(r1), int(c1))
            else:
                rr, cc = line(int(r0), int(c0), int(r1), int(c1))
                if width == 1:
                    mask = draw_mask_points(mask, rr, cc)
                else:
                    for r, c in zip(rr, cc):
                        rr_disk, cc_disk = disk((r, c), radius=width // 2, shape=mask.shape)
                        mask = draw_mask_points(mask, rr_disk, cc_disk)
        masks.append(mask)
    return np.stack(masks, axis=0) if masks else np.zeros((0, *mask_shape), dtype=bool)


def draw_mask_points(mask, xx, yy):
    """
    Safely draw one or more points into a boolean mask.

    Parameters
    ----------
    mask : np.ndarray
        2D boolean or integer mask of shape (H, W). Will be modified in-place.
    xx, yy : int or array-like
        Column (x) and row (y) coordinates to set True. Can be scalars or 1D
        array-likes of equal length.

    Behavior
    --------
    - If any coordinate equals the image extent (e.g. 512 for size 512) it is
      clamped to the maximum valid index (511) and a warning is emitted.
    - If any coordinate is greater than the image extent (>512 in the example)
      an IndexError is raised.

    Returns
    -------
    mask : np.ndarray
        The input mask with the given points set to True.
    """
    if mask.ndim != 2:
        raise ValueError("draw_mask_points expects a 2D mask (H, W)")

    xx = np.asarray(xx, dtype=int)
    yy = np.asarray(yy, dtype=int)

    # Broadcast scalars to arrays
    if xx.shape == ():
        xx = np.array([int(xx)])
    if yy.shape == ():
        yy = np.array([int(yy)])

    h, w = mask.shape

    # Negative indices are invalid here
    if np.any(xx < 0) or np.any(yy < 0):
        raise IndexError("Negative coordinates are not allowed")

    # Check for coordinates strictly greater than image size -> error
    if np.any(xx > h):
        raise IndexError(f"Row coordinate larger than image height: max(row)={int(xx.max())} > {h}")
    if np.any(yy > w):
        raise IndexError(f"Col coordinate larger than image width: max(col)={int(yy.max())} > {w}")

    # Clamp coordinates equal to image extent (e.g., 512 -> 511) and warn
    if np.any(xx == h):
        logger.warning(f"Row coordinate equal to image height ({h}); clamping to {h-1}")
        xx = np.where(xx == h, h - 1, xx)
    if np.any(yy == w):
        logger.warning(f"Col coordinate equal to image width ({w}); clamping to {w-1}")
        yy = np.where(yy == w, w - 1, yy)

    # Now xx and yy are valid indices in [0, h-1] and [0, w-1]
    # If arrays, use pairwise indexing
    mask[xx, yy] = True

    return mask


# -------------------------
# IoU ML metrics processing
# -------------------------
def _get_matches(
    iou_matrix: np.ndarray, iou_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Hungarian assignment on IoU/T/SKIoU to get GT<->Pred matches above threshold.
    Returns:
        matches: list of (gt_idx, pred_idx)
        unmatched_preds: list of pred indices
        unmatched_gts  : list of gt indices
    """
    M, N = iou_matrix.shape
    if M == 0 or N == 0:
        return [], list(range(N)), list(range(M))

    gt_idx, pr_idx = linear_sum_assignment(-iou_matrix)
    matches = [(g, p) for g, p in zip(gt_idx, pr_idx) if iou_matrix[g, p] >= iou_threshold]

    matched_g = {g for g, _ in matches}
    matched_p = {p for _, p in matches}

    unmatched_g = [g for g in range(M) if g not in matched_g]
    unmatched_p = [p for p in range(N) if p not in matched_p]
    return matches, unmatched_p, unmatched_g


# -------------------------
# Biologically relevant geometry metrics processing
# -------------------------
def _get_instance_mask_length(instance_mask: np.ndarray) -> float:
    """Skeleton length for given instance mask (proxy for filament length through counting pixels of skeleton)."""
    return float(np.sum(skeletonize(instance_mask)))


def _get_ordered_anchor_points_length(ordered_coords: np.ndarray) -> float:
    """Length for given ordered anchor points from an AnchorpointModel (e.g., TARDIS, FIESTA)
    (proxy for filament length through summing distances between points)."""
    if len(ordered_coords) < 2:
        return 0.0
    # Sum Euclidean distances between consecutive points
    diffs = np.diff(ordered_coords, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return float(seg_lengths.sum())


def _get_length_distribution(instance_masks: np.ndarray) -> np.ndarray:
    """Skeleton length distribution."""
    if instance_masks.size == 0:
        return np.array([])
    return np.array([_get_instance_mask_length(m) for m in instance_masks], dtype=float)


def _get_length_distribution_anchor_points(anchor_points_instance_masks: list) -> np.ndarray:
    """Skeleton length per instance (proxy for filament length) for ordered anchor points (e.g., from FIESTA or TARDIS)."""
    if len(anchor_points_instance_masks) == 0:
        return np.array([])
    lengths = []
    for coords in anchor_points_instance_masks:
        lengths.append(_get_ordered_anchor_points_length(coords))
    return np.array(lengths, dtype=float)


def _compute_average_curvature_from_ordered_skeleton_coords(
    ordered_coords: np.ndarray, spline_s: Optional[float] = 0
):
    """
    The curvature is calculated as the norm of the cross product of the first and second derivatives
    divided by the norm of the first derivative raised to the power of three.
    Returns average curvatures, all curvature values, and spline coefficients.

    Inspired by TARDIS, see https://github.com/SMLC-NYSBC/TARDIS/blob/9e3e405db993acf37d491c53066418d8aece441c/tardis_em/analysis/geometry_metrics.py#L20

    Parameters:
        ordered_coords: np.ndarray
            Ordered coordinates of the skeleton points.
        spline_s: float or None
            Smoothing factor for splprep. Default of sklearn's splprep function is None; TARDIS
            uses 1. Even though we found worse results with s=0 and s=1 on the simple circle
            examples below, we choose s=0 because it preserves the geometry of the original anchor
            points most faithfully.
    """
    # Fit spline to ordered skeleton coords
    try:
        tck, u = fit_parametric_curve(ordered_coords.T, s=spline_s, method="spline")

        # Calculate the first and second derivatives
        der1 = eval_parametric_curve(u, tck, der=1)
        der2 = eval_parametric_curve(u, tck, der=2)
        r1 = np.vstack(der1).T
        r2 = np.vstack(der2).T

        # Calculate curvature
        cross = r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0]
        curvature_values = np.abs(cross) / (np.linalg.norm(r1, axis=1) ** 3)
        average_curvature = np.nanmean(curvature_values)
        return average_curvature, curvature_values, tck
    except Exception:  # TODO: debug better here?
        return np.nan, [], None


def _instance_mask_to_ordered_skeleton_anchor_coords(
    instance_mask: np.ndarray,
) -> np.ndarray:
    # Check if its an empty mask
    if np.sum(instance_mask) == 0:
        return np.empty((0, 2))
    skeleton = skeletonize(instance_mask)
    coords = np.argwhere(skeleton)
    # Order skeleton points (nearest-neighbor for non-branching filaments)
    ordered_coords = [coords[0]]
    remaining_coords = list(coords[1:])
    current_coord = coords[0]
    while remaining_coords:
        distances = cdist([current_coord], remaining_coords)
        nearest_idx = np.argmin(distances)
        current_coord = remaining_coords.pop(nearest_idx)
        ordered_coords.append(current_coord)
    return np.array(ordered_coords)


def _get_average_curvatures(
    instance_masks: Optional[np.ndarray],
    anchor_points_instance_masks: Optional[list] = None,
    debug_plot: bool = False,
    spline_s: Optional[float] = 0,
    mask_shape: Optional[tuple] = (512, 512),
) -> np.ndarray:
    """
    Computes the average curvature of splines fit to the skeleton of each instance mask (resp.
    to the skeleton given by an AnchorPointsModel directly).
    Returns an array of average curvatures, one for each instance mask.

    Parameters:
        instance_masks: np.ndarray
            Stack of instance masks.
        anchor_points_instance_masks: np.ndarray
            Stack of skeletonized instance masks (in anchor points format) from an AnchorPointModel
            (e.g., TARDIS or FIESTA).
        debug_plot: bool
            If True, show debug plots and print curvature values.
        spline_s: float or None
            Smoothing factor for splprep. Default of sklearn's splprep function is None; TARDIS
            uses 1. Even though we found worse results with s=0 and s=1 on the simple circle
            examples below, we choose s=0 because it preserves the geometry of the original anchor
            points most faithfully.
        mask_shape: tuple
            Shape of the masks, used if converting anchor points to instance masks.
    """
    assert (
        instance_masks is None or anchor_points_instance_masks is None
    ), "Provide either `instance_masks` or `anchor_points_instance_masks`, not both."
    average_curvatures = []

    if anchor_points_instance_masks is not None:
        instance_masks = anchor_points_to_instance_masks(
            anchor_points_instance_masks, mask_shape, width=1
        )

    if instance_masks is not None:
        anchor_points_instance_masks = [
            _instance_mask_to_ordered_skeleton_anchor_coords(m) for m in instance_masks
        ]

    if anchor_points_instance_masks is None or len(anchor_points_instance_masks) == 0:
        return np.array(average_curvatures)  # Return empty array

    for idx, ordered_coords in enumerate(anchor_points_instance_masks):
        if len(ordered_coords) < 2:  # Need enough points to calculate curvature
            average_curvatures.append(np.nan)
            continue

        average_curvature, curvature_values, tck = (
            _compute_average_curvature_from_ordered_skeleton_coords(ordered_coords, spline_s)
        )
        average_curvatures.append(average_curvature)

        if debug_plot and not np.isnan(average_curvature):
            std_curvature = np.nanstd(curvature_values)
            print(f"Instance {idx} mean curvature: {average_curvature}, std: {std_curvature}")
            u_fine = np.linspace(0, 1, 200)
            spline_points = np.array(eval_parametric_curve(u_fine, tck)).T
            if instance_masks is not None:
                plt.imshow(instance_masks[idx], cmap="gray", alpha=0.5)
            else:
                # Plot the anchor_points_instance_masks points
                # Also, scale the image to 512x512
                plt.scatter(ordered_coords[:, 1], ordered_coords[:, 0], c="red", s=spline_s)
                plt.xlim(0, 512)
                plt.ylim(512, 0)
            plt.plot(spline_points[:, 1], spline_points[:, 0], c="blue", label="Spline fit")
            plt.title(
                f"Instance {idx} - Avg curvature: {average_curvature:.4f}, Std: {std_curvature:.4f}"
            )
            plt.legend()
            plt.show()

    return np.array(average_curvatures, dtype=float)


# -------------------------
# Public metrics (segmentation and downstream)
# -------------------------
def calculate_segmentation_metrics(
    gt_masks: Union[np.ndarray, List[np.ndarray]],
    pred_masks: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]] = None,
    anchor_points_instance_masks: Optional[List[Optional[list]]] = None,
        thresholds=None,
    use_skeletonized_version=True,
    spline_s=0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute instance segmentation metrics for one or more images.

    Args:
        gt_masks: Ground truth masks. Can be a single mask array or a list of masks.
        pred_masks: Predicted masks. Can be a single mask, a list of masks, or None.
        anchor_points_instance_masks: List of anchor points for instance masks.
        thresholds: List of thresholds to use for metrics.
        use_skeletonized_version: Whether to use the skeletonized version of IoU and IoT.
        spline_s:
            Smoothing factor for splprep. Default of sklearn's splprep function is None; TARDIS
            uses 1. Even though we found worse results with s=0 and s=1 on the simple circle
            examples below, we choose s=0 because it preserves the geometry of the original anchor
            points most faithfully.

    Returns:
        A dictionary of mean metric values across all images.
    """

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9]

    if not isinstance(gt_masks, list):
        gt_masks = [gt_masks]
        if pred_masks is not None:
            pred_masks = [pred_masks]
        if anchor_points_instance_masks is not None:
            anchor_points_instance_masks = [anchor_points_instance_masks]

    all_metrics = []
    num_images = len(gt_masks)

    for i in range(num_images):
        gt_mask = gt_masks[i]
        pred_mask = pred_masks[i] if pred_masks is not None else None
        pred_anchor_points = (
            anchor_points_instance_masks[i] if anchor_points_instance_masks is not None else None
        )

        # import matplotlib.pyplot as plt
        # plt_mask = np.max(gt_mask, axis=0)
        # plt.imshow(plt_mask)
        # for aps in pred_anchor_points:
        #     for ap in aps:
        #         plt.scatter(ap[1], ap[0], c="red", s=1)
        # plt.title("Pred Anchor Points")
        # plt.show()

        gt_instance_masks = as_instance_stack(gt_mask)

        if pred_mask is not None:
            pred_instance_masks = as_instance_stack(pred_mask)
            num_current_instances = pred_instance_masks.shape[0]
            if use_skeletonized_version:
                similarity_matrix = _compute_skiou_matrix(
                    gt_instance_masks, pred_instance_masks=pred_instance_masks
                )
            else:
                similarity_matrix = _compute_iou_matrix(gt_instance_masks, pred_instance_masks)
        elif pred_anchor_points is not None:
            num_current_instances = len(pred_anchor_points)
            if use_skeletonized_version:
                similarity_matrix = _compute_skiou_matrix(
                    gt_instance_masks,
                    pred_instance_anchor_points=pred_anchor_points,
                    spline_s=spline_s,
                )
            else:
                similarity_matrix = _compute_iot_matrix(gt_instance_masks, pred_anchor_points)
        else:
            num_current_instances = 0
            shape = gt_instance_masks.shape[1:] if gt_instance_masks.ndim > 2 else gt_mask.shape
            pred_instance_masks = np.zeros((0, *shape), dtype=bool)
            similarity_matrix = np.zeros((gt_instance_masks.shape[0], 0), dtype=float)

        metrics: Dict[str, float] = {}

        # “AP50–95” (mean precision across thresholds)
        precisions: List[float] = []
        for thresh in thresholds:  # np.arange(0.5, 1.0, 0.05):
            matches, unmatched_preds, _ = _get_matches(similarity_matrix, thresh)
            tp = len(matches)
            fp = len(unmatched_preds)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(precision)
            if np.isclose(thresh, 0.5) or np.isclose(thresh, 0.75) or np.isclose(thresh, 0.9):
                metrics[f"AP@{thresh:.2f}"] = float(precision)

        metrics["AP50-95"] = float(np.mean(precisions) if precisions else 0.0)
        metrics["AP"] = metrics["AP50-95"]  # alias for compatibility

        # Thresholded metrics
        for thresh in thresholds:
            matches, unmatched_preds, unmatched_gts = _get_matches(similarity_matrix, float(thresh))
            tp, fp, fn = len(matches), len(unmatched_preds), len(unmatched_gts)

            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            metrics[f"F1@{thresh:.2f}"] = float(f1)

        # IoU/IoT/SKIoU
        if use_skeletonized_version:
            key = "SKIoU"
        else:
            key = "IoU/T"
        if gt_instance_masks.shape[0] > 0 and num_current_instances:
            # sim = similarity_matrix.max(axis=1)  # best IoU/IoT/SKIoU for each GT
            # metrics[f"{key}_mean"] = float(np.mean(sim))
            # metrics[f"{key}_median"] = float(np.median(sim))

            # enforce 1-to-1 matching (threshold = 0.0 means allow matching anything)
            matches, unmatched_preds, unmatched_gts = _get_matches(
                similarity_matrix, iou_threshold=0.0
            )

            ious = [similarity_matrix[g, p] for g, p in matches]

            # unmatched GTs carry IoU = 0
            ious.extend([0.0] * (len(unmatched_gts) + len(unmatched_preds)))

            metrics[f"{key}_mean"] = float(np.mean(ious)) if ious else 0.0
            metrics[f"{key}_median"] = float(np.median(ious)) if ious else 0.0
        else:
            metrics[f"{key}_mean"] = 0.0
            metrics[f"{key}_median"] = 0.0

        all_metrics.append(metrics)

    # Average metrics over all images
    if not all_metrics:
        return {}, {}

    all_metrics_df = pd.DataFrame(all_metrics)
    # Averaging out the metrics per image
    mean_metrics = all_metrics_df.mean().to_dict()

    return mean_metrics, all_metrics_df


def _compute_histogram_distributions(
    gt_data: np.ndarray, pred_data: np.ndarray, num_bins_suggestion: Optional[int] = None
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Computes normalized histogram distributions for two datasets with shared bins.
    """
    gt_data = gt_data[~np.isnan(gt_data)]
    pred_data = pred_data[~np.isnan(pred_data)]

    if gt_data.size == 0:
        logger.error("GT data not correctly given.")
        return None

    # If we have no predicitons but only gt data, return at least the gt data
    #  for further plotting
    if pred_data.size == 0:
        min_val = float(gt_data.min())
        max_val = float(gt_data.max())
    else:
        min_val = float(min(gt_data.min(), pred_data.min()))
        max_val = float(max(gt_data.max(), pred_data.max()))

    if max_val <= min_val:
        logger.error("Max value less than or equal to min value, cannot compute histogram.")
        return None

    if num_bins_suggestion is None:
        num_bins = max(10, int(np.sqrt(len(gt_data))))
        # num_bins = 100 # For figuring out the length spikes;
        #  binning is not the problem!
    else:
        num_bins = num_bins_suggestion

    # num_bins = min(num_bins, 100)  # Cap at 100 bins to avoid overfitting noise
    bins = np.linspace(min_val, max_val, num_bins)

    hist1_abs, _ = np.histogram(gt_data, bins=bins, density=False)
    hist2_abs, _ = np.histogram(pred_data, bins=bins, density=False)
    hist1, _ = np.histogram(gt_data, bins=bins, density=True)
    hist2, _ = np.histogram(pred_data, bins=bins, density=True)

    eps = 1e-10
    dist1 = hist1 + eps
    dist2 = hist2 + eps
    dist1 /= dist1.sum()
    dist2 /= dist2.sum()

    return dist1, dist2, bins, hist1_abs, hist2_abs


def calculate_downstream_metrics(
    gt_masks: Union[np.ndarray, List[np.ndarray]],
    pred_masks: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]] = None,
    anchor_points_instance_masks: Optional[List[Optional[list]]] = None,
    spline_s: Optional[float] = 0,
    save_histograms_path: Optional[str] = None,
    return_histogram_data: bool = False,
    pixel_per_micrometer: Union[float, List[float]] = 9.0,
):
    """
    Compute biological downstream metrics for one or more images.

    Args:
        gt_masks: Ground truth masks. Can be a single mask array or a list of masks.
        pred_masks: Predicted masks. Can be a single mask, a list of masks, or None.
        anchor_points_instance_masks: List of anchor points for instance masks.
        spline_s:
            Smoothing factor for splprep. Default of sklearn's splprep function is None; TARDIS
            uses 1. Even though we found worse results with s=0 and s=1 on the simple circle
            examples below, we choose s=0 because it preserves the geometry of the original anchor
            points most faithfully.
        save_histograms_path: If provided, saves histogram plots to this path.
        return_histogram_data: If True, returns histogram data along with metrics.
        pixel_per_micrometer: Pixels per micrometer scaling factor. Can be a single float or a list
            of floats matching the number of images.

    Returns:
        A dictionary of metric values calculated over all images.
        If `return_histogram_data=True` the function will return a tuple (metrics_dict,
        histograms_dict, raw_arr_data) where `histograms_dict` contains keys 'length' and
        'curvature' mapping to tuples (bins, gt_hist, pred_hist, x_axis_label, x_max, kl_val) that
        can be used to recreate plots. The third element of the tuple, `raw_arr_data`, contains the
        raw count, length, and curvature arrays.
    """
    # Normalize to lists
    if not isinstance(gt_masks, list):
        gt_masks = [gt_masks]
        if pred_masks is not None:
            pred_masks = [pred_masks]
        if anchor_points_instance_masks is not None:
            anchor_points_instance_masks = [anchor_points_instance_masks]

    num_images = len(gt_masks)

    # Normalize ppm to list
    if isinstance(pixel_per_micrometer, (int, float)):
        ppm_list = [float(pixel_per_micrometer)] * num_images
    else:
        if len(pixel_per_micrometer) != num_images:
            raise ValueError("pixel_per_micrometer list must match number of images.")
        ppm_list = [float(v) for v in pixel_per_micrometer]

    if pred_masks is None and anchor_points_instance_masks is None:
        pred_masks = [None] * num_images

    # Containers
    all_gt_lengths_px, all_pred_lengths_px = [], []
    all_gt_curv_px, all_pred_curv_px = [], []

    all_gt_lengths_um, all_pred_lengths_um = [], []
    all_gt_curv_um, all_pred_curv_um = [], []

    all_gt_counts, all_pred_counts = [], []

    # --------------------------------------------------------
    # PROCESS EACH IMAGE
    # --------------------------------------------------------
    for i in range(num_images):
        ppm = ppm_list[i]  # pixels per micrometer

        gt_mask = gt_masks[i]
        pred_mask = pred_masks[i] if pred_masks is not None else None
        anchor_points = (
            anchor_points_instance_masks[i] if anchor_points_instance_masks is not None else None
        )

        gt_instance_masks = as_instance_stack(gt_mask)
        all_gt_counts.append(gt_instance_masks.shape[0])

        # --- GT ---
        lengths_gt_px = np.array(_get_length_distribution(gt_instance_masks))
        curv_gt_px = np.array(_get_average_curvatures(gt_instance_masks, spline_s=spline_s))

        all_gt_lengths_px.extend(lengths_gt_px)
        all_gt_curv_px.extend(curv_gt_px)

        # convert to micrometers
        all_gt_lengths_um.extend(lengths_gt_px / ppm)
        all_gt_curv_um.extend(curv_gt_px * ppm)

        # --- PRED ---
        if pred_mask is not None:
            pred_instance_masks = as_instance_stack(pred_mask)
            all_pred_counts.append(pred_instance_masks.shape[0])

            lengths_pred_px = np.array(_get_length_distribution(pred_instance_masks))
            curv_pred_px = np.array(_get_average_curvatures(pred_instance_masks, spline_s=spline_s))

        elif anchor_points is not None:
            all_pred_counts.append(len(anchor_points))

            lengths_pred_px = np.array(_get_length_distribution_anchor_points(anchor_points))
            curv_pred_px = np.array(
                _get_average_curvatures(
                    instance_masks=None,
                    anchor_points_instance_masks=anchor_points,
                    spline_s=spline_s,
                    mask_shape=gt_instance_masks[0].shape,
                )
            )
        else:
            all_pred_counts.append(0)
            lengths_pred_px, curv_pred_px = np.array([]), np.array([])

        all_pred_lengths_px.extend(lengths_pred_px)
        all_pred_curv_px.extend(curv_pred_px)

        # convert to micrometers
        all_pred_lengths_um.extend(lengths_pred_px / ppm)
        all_pred_curv_um.extend(curv_pred_px * ppm)

    # Convert to numpy arrays
    gt_counts = np.array(all_gt_counts)
    pred_counts = np.array(all_pred_counts)

    gt_lengths_px = np.array(all_gt_lengths_px)
    pred_lengths_px = np.array(all_pred_lengths_px)
    gt_curv_px = np.array(all_gt_curv_px)
    pred_curv_px = np.array(all_pred_curv_px)

    gt_lengths_um = np.array(all_gt_lengths_um)
    pred_lengths_um = np.array(all_pred_lengths_um)
    gt_curv_um = np.array(all_gt_curv_um)
    pred_curv_um = np.array(all_pred_curv_um)

    # Raw arrays
    raw_arr_data = {
        "count": {"gt": gt_counts, "pred": pred_counts},
        "length": {"gt": gt_lengths_px, "pred": pred_lengths_px},
        "curvature": {"gt": gt_curv_px, "pred": pred_curv_px},
        "length_um": {"gt": gt_lengths_um, "pred": pred_lengths_um},
        "curvature_um": {"gt": gt_curv_um, "pred": pred_curv_um},
    }

    out = {
        # KL divs
        "Length_KL": np.inf,
        "Curvature_KL": np.inf,
    }

    calculate_downstream_metrics_from_gathered_arrays(
        out,
        gt_counts,
        pred_counts,
        gt_lengths_px,
        pred_lengths_px,
        gt_curv_px,
        pred_curv_px,
        gt_lengths_um,
        pred_lengths_um,
        gt_curv_um,
        pred_curv_um,
    )

    plots = {}
    turn_all_distributions_into_scores_and_plots(
        out,
        plots,
        gt_lengths_um,
        pred_lengths_um,
        gt_curv_um,
        pred_curv_um,
        save_histograms_path,
    )

    if return_histogram_data:
        return out, plots, raw_arr_data

    return out


def compute_downstream_metrics_from_raw_distribution_files(
    raw_distributions_data_path: str,
    save_histograms_path: Optional[str] = None,
    return_histogram_data: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], dict]]:
    """
    Compute biological downstream metrics from previously saved raw distribution data file.

    Args:
        raw_distributions_data_path: Path to the raw distribution data file (npz format).
        save_histograms_path: If provided, saves histogram plots to this path (only suffix (.png resp. .pdf) missing).
        return_histogram_data: If True, returns histogram data along with metrics.

    Returns:
        A dictionary of biological downstream values.
        If `return_histogram_data=True` the function will return a tuple (metrics_dict, histograms_dict)
        where `histograms_dict` contains keys 'length' and 'curvature' mapping to tuples
        (bins, gt_hist, pred_hist, x_axis_label, x_max, kl_val) that can be used to recreate plots.
    """
    # Check if file exist, if not return empty dict
    if not os.path.exists(raw_distributions_data_path):
        logger.warning(f"Raw distributions data file {raw_distributions_data_path} does not exist.")
        return {} if not return_histogram_data else ({}, {})

    # Get length and curvature distributions
    raw_arr_data = np.load(raw_distributions_data_path, allow_pickle=True).item()

    # Actually compute the metrics
    out = {
        # KL divs
        "Length_KL": np.inf,
        "Curvature_KL": np.inf,
    }

    calculate_downstream_metrics_from_gathered_arrays(
        out,
        raw_arr_data["count"]["gt"],
        raw_arr_data["count"]["pred"],
        raw_arr_data["length"]["gt"],
        raw_arr_data["length"]["pred"],
        raw_arr_data["curvature"]["gt"],
        raw_arr_data["curvature"]["pred"],
        raw_arr_data["length_um"]["gt"],
        raw_arr_data["length_um"]["pred"],
        raw_arr_data["curvature_um"]["gt"],
        raw_arr_data["curvature_um"]["pred"],
    )

    plots = {}
    turn_all_distributions_into_scores_and_plots(
        out,
        plots,
        raw_arr_data["length_um"]["gt"],
        raw_arr_data["length_um"]["pred"],
        raw_arr_data["curvature_um"]["gt"],
        raw_arr_data["curvature_um"]["pred"],
        save_histograms_path,
    )

    if return_histogram_data:
        return out, plots
    return out


def calculate_downstream_metrics_from_gathered_arrays(
    results_dict,
    gt_counts: np.array,
    pred_counts: np.array,
    gt_lengths_px: np.array,
    pred_lengths_px: np.array,
    gt_curv_px: np.array,
    pred_curv_px: np.array,
    gt_lengths_um: np.array,
    pred_lengths_um: np.array,
    gt_curv_um: np.array,
    pred_curv_um: np.array,
):
    """
    Compute biological downstream metrics given raw arrays, except count.
    """

    avg_count_gt = np.mean(gt_counts)
    std_count_gt = np.std(gt_counts)
    avg_count_pred = np.mean(pred_counts)
    std_count_pred = np.std(pred_counts)
    abs_err = abs(avg_count_gt - avg_count_pred)
    rel_err = abs_err / (avg_count_gt + 1e-6)

    results_dict.update(
        {
            # --- Count ---
            "Avg Count GT": float(avg_count_gt),
            "Std Count GT": float(std_count_gt),
            "Avg Count Pred": float(avg_count_pred),
            "Std Count Pred": float(std_count_pred),
            "Count Abs Err": float(abs_err),
            "Count Rel Err": float(rel_err),
            # --- Length in pixels ---
            "Avg Length GT": float(np.nanmean(gt_lengths_px)) if gt_lengths_px.size else np.inf,
            "Std Length GT": float(np.nanstd(gt_lengths_px)) if gt_lengths_px.size else np.inf,
            "Avg Length Pred": (
                float(np.nanmean(pred_lengths_px)) if pred_lengths_px.size else np.inf
            ),
            "Std Length Pred": (
                float(np.nanstd(pred_lengths_px)) if pred_lengths_px.size else np.inf
            ),
            "Length Avg Abs Err": (
                float(abs(np.nanmean(gt_lengths_px) - np.nanmean(pred_lengths_px)))
                if gt_lengths_px.size and pred_lengths_px.size
                else np.inf
            ),
            # --- Curvature in pixels ---
            "Avg Curvature GT": float(np.nanmean(gt_curv_px)) if gt_curv_px.size else np.inf,
            "Std Curvature GT": float(np.nanstd(gt_curv_px)) if gt_curv_px.size else np.inf,
            "Avg Curvature Pred": float(np.nanmean(pred_curv_px)) if pred_curv_px.size else np.inf,
            "Std Curvature Pred": (float(np.nanstd(pred_curv_px)) if pred_curv_px.size else np.inf),
            "Curvature Avg Abs Err": (
                float(abs(np.nanmean(gt_curv_px) - np.nanmean(pred_curv_px)))
                if gt_curv_px.size and pred_curv_px.size
                else np.inf
            ),
            # --- Length in micrometers ---
            "Avg Length GT (um)": (
                float(np.nanmean(gt_lengths_um)) if gt_lengths_um.size else np.inf
            ),
            "Std Length GT (um)": float(np.nanstd(gt_lengths_um)) if gt_lengths_um.size else np.inf,
            "Avg Length Pred (um)": (
                float(np.nanmean(pred_lengths_um)) if pred_lengths_um.size else np.inf
            ),
            "Std Length Pred (um)": (
                float(np.nanstd(pred_lengths_um)) if pred_lengths_um.size else np.inf
            ),
            "Length Avg Abs Err (um)": (
                float(abs(np.nanmean(gt_lengths_um) - np.nanmean(pred_lengths_um)))
                if gt_lengths_um.size and pred_lengths_um.size
                else np.inf
            ),
            # --- Curvature in micrometers ---
            "Avg Curvature GT (1/um)": float(np.nanmean(gt_curv_um)) if gt_curv_um.size else np.inf,
            "Std Curvature GT (1/um)": float(np.nanstd(gt_curv_um)) if gt_curv_um.size else np.inf,
            "Avg Curvature Pred (1/um)": (
                float(np.nanmean(pred_curv_um)) if pred_curv_um.size else np.inf
            ),
            "Std Curvature Pred (1/um)": (
                float(np.nanstd(pred_curv_um)) if pred_curv_um.size else np.inf
            ),
            "Curvature Avg Abs Err (1/um)": (
                float(abs(np.nanmean(gt_curv_um) - np.nanmean(pred_curv_um)))
                if gt_curv_um.size and pred_curv_um.size
                else np.inf
            ),
        }
    )


def turn_all_distributions_into_scores_and_plots(
    out: Dict[str, float],
    plots: Dict[str, Any],
    gt_lengths_um: np.ndarray,
    pred_lengths_um: np.ndarray,
    gt_curv_um: np.ndarray,
    pred_curv_um: np.ndarray,
    save_histograms_path: Optional[str] = None,
):
    """Histograms plotted in micrometers. Note that out and plots will be overwritten (pointers!)."""

    turn_distributions_into_scores_and_plots(
        "Length",
        "µm",
        gt_lengths_um,
        pred_lengths_um,
        -2,
        33,  # SynthMT: 33, Real: 25
        out,
        plots,
        save_histograms_path,
    )
    turn_distributions_into_scores_and_plots(
        "Curvature",
        "1/µm",
        gt_curv_um,
        pred_curv_um,
        -2,
        25,  # SynthMT: 25, Real: 25
        out,
        plots,
        save_histograms_path,
    )


def turn_distributions_into_scores_and_plots(
    name: str,
    unit: str,
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    x_min: float,
    x_max: float,
    out: Dict[str, float],
    plots: dict,
    save_histograms_path: Optional[str] = None,
):
    name_lower = name.lower()
    hists = _compute_histogram_distributions(gt_arr, pred_arr)
    if hists:
        gt_dist, pred_dist, bins, gt_hist, pred_hist = hists
        x_axis_label = f"{name} ({unit})"
        out[f"{name}_KL"] = round(float(entropy(pred_dist, gt_dist)), 3)
        kl_val = out[f"{name}_KL"]
        plots[f"{name_lower}"] = {
            "gt_dist": gt_dist,
            "pred_dist": pred_dist,
            "bins": bins,
            "x_axis_label": x_axis_label,
            "x_min": x_min,
            "x_max": x_max,
            "kl_val": kl_val,
        }
        if save_histograms_path is not None:
            save_path = os.path.join(
                os.path.dirname(save_histograms_path),
                "length_" + os.path.basename(save_histograms_path),
            )
            plot_histogram_distributions(
                gt_dist,
                pred_dist,
                bins,
                x_axis_label=x_axis_label,
                x_min=x_min,
                x_max=x_max,
                kl_val=kl_val,
                save_path=save_path,
            )


def plot_histogram_distributions(
    gt_dist,
    pred_dist,
    bins,
    x_axis_label,
    x_min,
    x_max,
    kl_val,
    gt_label="Ground Truth",
    pred_label="Prediction",
    save_path: str = "",
    ax: Optional[Axes] = None,
    title_suffix="",
    draw_x_label=True,
    draw_y_label=True,
    draw_y_axisticks=True,
    draw_legend=True,
    y_min=None,
    y_max=None,
):
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        created_fig = True
    else:
        fig = ax.figure
    # side-by-side bars
    color_blue = "#82B366"
    color_purple = "#AC37E5"
    ax.bar(
        bin_centers - width * 0.18,
        gt_dist,
        width=width * 0.35,
        label=gt_label,
        # alpha=0.9,
        color=color_blue,
        edgecolor="none",
    )
    ax.bar(
        bin_centers + width * 0.18,
        pred_dist,
        width=width * 0.35,
        label=pred_label,
        # alpha=0.9,
        color=color_purple,
        edgecolor="none",
    )

    # overlay line traces for clarity
    # ax.plot(bin_centers, dist_gt, color="C0", lw=1)
    # ax.plot(bin_centers, dist_pred, color="C1", lw=1)

    ax.set_xlim(x_min, x_max)
    if draw_x_label:
        ax.set_xlabel(x_axis_label)
    else:
        ax.tick_params(labelbottom=False)
    if draw_y_label:
        ax.set_ylabel("Probability Mass")
    elif draw_y_axisticks:
        pass
        # ax.tick_params(labelleft=True)
    else:
        ax.tick_params(labelleft=False)
    if title_suffix != "":
        # Replace * in this by + HPO
        title_suffix = title_suffix.replace("*", "+ HPO")
        title = rf"{title_suffix} $\mathrm{{KL}} = {kl_val:.3f}$"
    else:
        title = rf"$\mathrm{{KL}} = {kl_val:.3f}$"  # (\text{{pred}}\parallel\text{{gt}})
    ax.set_title(title)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    if draw_legend:
        ax.legend()
    # ax.grid(alpha=0.2)
    # ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

    if save_path != "":
        fig.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
        # If we created the figure for saving we close it to free resources
        if created_fig:
            plt.close(fig)
    return fig, ax


# -------------------------
# Tests for/with TARDIS
# -------------------------
def test_curvature_function():
    """
    Test the curvature calculation in _get_average_curvatures on two cases:
    1. Straight line (should yield near-zero curvature)
    2. Half circular arc (should yield curvature ~1/radius)
    For synthetic, clean shapes, s=None gives the most accurate curvature (forces interpolation).
    For real/noisy masks, s=1 (the TARDIS default) might be better.
    NOTE: We ran it across predictions from TARDIS across 500 images and did not see noticable differences.
    """
    # Test 1: Straight line
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:91, 10] = 1  # vertical line from (10,10) to (90,10)
    avg_curvature_none = _get_average_curvatures(
        np.expand_dims(mask, axis=0), debug_plot=True, spline_s=None
    )
    print("Straight line mean curvature (s=None, best for synthetic):", avg_curvature_none)
    avg_curvature_default = _get_average_curvatures(np.expand_dims(mask, axis=0), debug_plot=True)
    print("Straight line mean curvature (s=1, default):", avg_curvature_default)

    # Test 2: Half circular arc
    mask = np.zeros((100, 100), dtype=np.uint8)
    center = (50, 50)
    radius = 30
    theta = np.linspace(0, np.pi, 500)  # half arc, more points
    rr = (center[0] + radius * np.cos(theta)).astype(int)
    cc = (center[1] + radius * np.sin(theta)).astype(int)
    mask[rr, cc] = 1
    avg_curvature_snone = _get_average_curvatures(
        np.expand_dims(mask, axis=0), debug_plot=True, spline_s=None
    )
    print(
        "Half circle arc mean curvature (s=None):",
        avg_curvature_snone,
        "Expected:",
        1 / radius,
    )
    avg_curvature_s1 = _get_average_curvatures(np.expand_dims(mask, axis=0), debug_plot=True)
    print(
        "Half circle arc mean curvature (s=1):",
        avg_curvature_s1,
        "Expected:",
        1 / radius,
    )
    avg_curvature_s0 = _get_average_curvatures(np.expand_dims(mask, axis=0), debug_plot=True)
    print(
        "Half circle arc mean curvature (s=0):",
        avg_curvature_s0,
        "Expected:",
        1 / radius,
    )


# test_curvature_function()

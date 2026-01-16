import logging
from typing import Optional, List, Any, Callable

import lpips
import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.spatial.distance import jensenshannon, mahalanobis
from scipy.stats import chi2_contingency
from sklearn.metrics.pairwise import cosine_similarity

from synth_mt.config.tuning import TuningConfig

logger = logging.getLogger(__name__)


def similarity(
    tuning_cfg: TuningConfig,
    ref_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    # --- Optional pre-computed values for optimization ---
    ref_mean: Optional[np.ndarray] = None,
    ref_inv_cov: Optional[np.ndarray] = None,
    ref_cov: Optional[np.ndarray] = None,
    ref_hist_bins: Optional[List[np.ndarray]] = None,
    ref_hist: Optional[np.ndarray] = None,
    ref_prob: Optional[np.ndarray] = None,
    ref_kid_term1: Optional[float] = None,
) -> float:
    """
    Core evaluation logic: Computes similarity using the specified metric.
    Handles both embedding vectors and raw image data (for LPIPS).
    Returns an array of scores. For instance-based metrics, this is one score per synthetic image.
    For distribution-based metrics, this is a single score in an array.
    """

    similarity_metric = getattr(tuning_cfg, "similarity_metric", "cosine")
    aggregation_method = getattr(tuning_cfg, "aggregation_method", "mean")
    logger.debug(f"--- [METRIC START] ---")
    logger.debug(f"Metric: '{similarity_metric}', Aggregation: '{aggregation_method}'")

    try:
        # --- LPIPS Metric (Image-based) ---
        if "lpips" in similarity_metric:
            model_net = similarity_metric.split("-")[-1]
            # For LPIPS, ref_embeddings and synthetic_embeddings are arrays of images
            score = compute_lpips_score(model_net, synthetic_embeddings, ref_embeddings)

        # --- Embedding-based Metrics ---
        else:
            if ref_embeddings.shape[0] == 0:
                logger.warning("Reference embeddings are empty. Cannot compute similarity.")
                return -float("inf")
            if synthetic_embeddings.shape[0] == 0:
                logger.warning("Synthetic embeddings are empty. Cannot compute similarity.")
                return -float("inf")

            if similarity_metric == "mahalanobis":
                if ref_inv_cov is None:
                    raise ValueError(
                        "Inverse covariance matrix is required for Mahalanobis distance."
                    )
                # Returns a score per synthetic image
                score = compute_mahalanobis_score(
                    synthetic_embeddings, ref_embeddings, ref_inv_cov, aggregation_method
                )
            elif similarity_metric == "cosine":
                # Returns a score per synthetic image
                score = compute_cosine_score(
                    synthetic_embeddings, ref_embeddings, aggregation_method
                )
            elif similarity_metric == "fid":
                if ref_mean is None or ref_cov is None:
                    raise ValueError("Reference mean and covariance are required for FID.")
                score = compute_frechet_distance(synthetic_embeddings, ref_mean, ref_cov)
            elif similarity_metric == "kid":
                score = compute_kernel_inception_distance(
                    synthetic_embeddings, ref_embeddings, ref_kid_term1
                )
            elif similarity_metric == "ndb":
                if ref_hist_bins is None or ref_hist is None:
                    raise ValueError("Reference histogram and bins are required for NDB.")
                score = compute_ndb_score(synthetic_embeddings, ref_hist_bins, ref_hist)
            elif similarity_metric == "jsd":
                if ref_hist_bins is None or ref_prob is None:
                    raise ValueError(
                        "Reference histogram bins and probabilities are required for JSD."
                    )
                score = compute_js_divergence_score(synthetic_embeddings, ref_hist_bins, ref_prob)
            else:
                raise ValueError(f"Unknown similarity metric: {similarity_metric}")

    except Exception as e:
        logger.error(f"Error during '{similarity_metric}' computation: {e}", exc_info=True)
        score = -float("inf")

    logger.debug(f"Final Score(s) for Trial: {score}")
    logger.debug(f"--- [METRIC END] ---\n")
    return score


def compute_cosine_score(
    synthetic_embeddings: np.ndarray, ref_embeddings: np.ndarray, aggregation: str
) -> float:
    """Computes mean cosine similarity, aggregating scores for each synthetic image against reference images."""
    logger.debug(f"[COSINE] Computing scores with '{aggregation}' aggregation.")
    agg_func: Callable[[np.ndarray], float] = {"max": np.max, "mean": np.mean, "min": np.min}[
        aggregation
    ]

    frame_scores = []
    for emb in synthetic_embeddings:
        # scores_vs_ref is 1D array of similarities against all reference embeddings
        scores_vs_ref = cosine_similarity(emb.reshape(1, -1), ref_embeddings)[0]
        # Aggregate the scores for the current synthetic frame
        agg_score = agg_func(scores_vs_ref)
        frame_scores.append(agg_score)

    mean_score = float(np.mean(frame_scores))
    logger.debug(f"[COSINE] Mean aggregated score: {mean_score:.6f}")
    return mean_score


def compute_mahalanobis_score(
    synthetic_embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_inv_cov: np.ndarray,
    aggregation: str,
) -> float:
    """Computes mean Mahalanobis distance, aggregating scores for each synthetic image."""
    logger.debug(f"[MAHALANOBIS] Computing distances with '{aggregation}' aggregation.")
    # Lower distance is better, so min/max logic is inverted for the score
    agg_func: Callable[[np.ndarray], float] = {"max": np.min, "mean": np.mean, "min": np.max}[
        aggregation
    ]

    frame_distances = []
    ref_mean = np.mean(ref_embeddings, axis=0)
    for synth_emb in synthetic_embeddings:
        # Calculate distance from the synthetic embedding to each reference embedding
        dist_vs_ref = np.array(
            [mahalanobis(synth_emb, ref_emb, ref_inv_cov) for ref_emb in ref_embeddings]
        )
        # Aggregate the distances for the current synthetic frame
        agg_dist = agg_func(dist_vs_ref)
        frame_distances.append(agg_dist)

    # Final score is the negated mean of the aggregated distances
    mean_distance = float(np.mean(frame_distances))
    final_score = -mean_distance
    logger.debug(
        f"[MAHALANOBIS] Mean aggregated distance: {mean_distance:.6f}, Final negated score: {final_score:.6f}"
    )
    return final_score


def compute_frechet_distance(
    synthetic_embeddings: np.ndarray, ref_mean: np.ndarray, ref_cov: np.ndarray
) -> float:
    """Computes the Fréchet distance between two sets of embeddings."""
    logger.debug("[FID] Computing Fréchet Distance...")
    mu1, sigma1 = ref_mean, ref_cov
    mu2 = np.mean(synthetic_embeddings, axis=0)
    sigma2 = np.cov(synthetic_embeddings, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return -float(fid)


def _polynomial_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Computes a polynomial kernel between two sets of embeddings."""
    gamma = 1.0 / X.shape[1] if X.shape[1] > 0 else 1.0
    return (gamma * (X @ Y.T) + 1.0) ** 3


def compute_kernel_inception_distance(
    synthetic_embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_kid_term1: Optional[float] = None,
) -> float:
    """Computes the Kernel Inception Distance between two sets of embeddings."""
    logger.debug("[KID] Computing Kernel Inception Distance...")
    m, n = ref_embeddings.shape[0], synthetic_embeddings.shape[0]
    if m < 2 or n < 2:
        return -float("inf")

    if ref_kid_term1 is None:
        k_xx = _polynomial_kernel(ref_embeddings, ref_embeddings)
        term1 = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    else:
        term1 = ref_kid_term1

    k_yy = _polynomial_kernel(synthetic_embeddings, synthetic_embeddings)
    k_xy = _polynomial_kernel(ref_embeddings, synthetic_embeddings)

    term2 = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    term3 = 2 * k_xy.sum() / (m * n)
    kid = term1 + term2 - term3
    if np.isnan(kid):
        kid = float("inf")

    return -float(kid)


def compute_ndb_score(
    synthetic_embeddings: np.ndarray,
    ref_hist_bins: List[np.ndarray],
    ref_hist: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Computes the Number of Statistically Different Bins score."""
    logger.debug("[NDB] Computing scores...")
    synth_hist, _ = np.histogramdd(synthetic_embeddings, bins=ref_hist_bins)
    meaningful_bins_indices = np.argwhere((ref_hist > 0) & (synth_hist > 0))
    if meaningful_bins_indices.shape[0] == 0:
        return 0.0

    statistically_different_bins = 0
    total_ref_samples = ref_hist.sum()
    total_synth_samples = synth_hist.sum()
    if total_ref_samples == 0 or total_synth_samples == 0:
        return -float("inf")

    for bin_idx_tuple in meaningful_bins_indices:
        bin_idx = tuple(bin_idx_tuple)
        ref_count = ref_hist[bin_idx]
        synth_count = synth_hist[bin_idx]
        contingency_table = np.array(
            [
                [ref_count, total_ref_samples - ref_count],
                [synth_count, total_synth_samples - synth_count],
            ]
        )
        try:
            _, p_value, _, _ = chi2_contingency(contingency_table, correction=False)
            if p_value < alpha:
                statistically_different_bins += 1
        except ValueError:
            continue
    return -float(statistically_different_bins)


def compute_js_divergence_score(
    synthetic_embeddings: np.ndarray, ref_hist_bins: List[np.ndarray], ref_prob: np.ndarray
) -> float:
    """Computes the Jensen-Shannon Divergence score."""
    logger.debug("[JSD] Computing scores...")
    synth_hist, _ = np.histogramdd(synthetic_embeddings, bins=ref_hist_bins)
    if synth_hist.sum() == 0:
        return -float("inf")

    synth_prob = (synth_hist / synth_hist.sum()).flatten()
    synth_prob[synth_prob == 0] = 1e-10

    jsd = jensenshannon(ref_prob, synth_prob)
    return float(-jsd)


def compute_lpips_score(
    model_net: str, synthetic_image: np.ndarray, ref_images: np.ndarray
) -> float:
    """
    Calculates the mean LPIPS score for synthetic images against the average of reference images.
    Expects images as numpy arrays.
    """
    logger.debug(f"[LPIPS-{model_net.upper()}] Computing scores...")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    loss_fn = lpips.LPIPS(net=model_net).to(device)

    def to_tensor(img: np.ndarray) -> torch.Tensor:
        # Convert single image HWC to NCHW tensor for LPIPS
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return (tensor * 2) - 1  # Normalize to [-1, 1]

    # Convert all reference images to a single tensor batch
    ref_tensors = torch.cat([to_tensor(img) for img in ref_images], dim=0).to(device)
    avg_ref_tensor = ref_tensors.mean(dim=0, keepdim=True)

    target_tensor = to_tensor(synthetic_image).to(device)
    with torch.no_grad():
        dist = loss_fn(avg_ref_tensor, target_tensor)

    return -float(dist.item())


def precompute_matric_args(tuning_cfg: TuningConfig, ref_embeddings: np.ndarray) -> dict[str, Any]:
    """
    Pre-computes values based on the selected metric to avoid redundant calculations.
    """
    metric = getattr(tuning_cfg, "similarity_metric", "cosine")
    logger.debug(f"Pre-computing values for metric: '{metric}'")
    precomputed_args: dict[str, Any] = {}

    if "lpips" in metric or ref_embeddings.shape[0] == 0:
        return precomputed_args

    if metric == "mahalanobis":
        aggregation = getattr(tuning_cfg, "aggregation_method", "max")
        # If comparing to mean_ref, the covariance should be of the original distribution
        if aggregation == "mean_ref":
            logger.debug(
                "Pre-computing inverse covariance matrix for Mahalanobis (from original refs)."
            )
        cov = np.cov(ref_embeddings, rowvar=False)
        precomputed_args["ref_inv_cov"] = np.linalg.pinv(cov)
    elif metric == "fid":
        precomputed_args["ref_mean"] = np.mean(ref_embeddings, axis=0)
        precomputed_args["ref_cov"] = np.cov(ref_embeddings, rowvar=False)
    elif metric == "kid":
        if ref_embeddings.shape[0] > 1:
            k_xx = _polynomial_kernel(ref_embeddings, ref_embeddings)
            precomputed_args["ref_kid_term1"] = (k_xx.sum() - np.trace(k_xx)) / (
                ref_embeddings.shape[0] * (ref_embeddings.shape[0] - 1)
            )
    elif metric in ["ndb", "jsd"]:
        num_bins = int(np.cbrt(ref_embeddings.shape[0]))
        hist, bins = np.histogramdd(ref_embeddings, bins=num_bins)
        precomputed_args["ref_hist_bins"] = bins
        if metric == "ndb":
            precomputed_args["ref_hist"] = hist
        if metric == "jsd":
            prob = (hist / hist.sum()).flatten()
            prob[prob == 0] = 1e-10
            precomputed_args["ref_prob"] = prob

    logger.debug(f"Pre-computation for '{metric}' complete.")
    return precomputed_args

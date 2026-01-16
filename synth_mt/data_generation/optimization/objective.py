import logging

import numpy as np
import optuna


from .embeddings import ImageEmbeddingExtractor
from .metrics import similarity
from ...config.tuning import TuningConfig

logger = logging.getLogger(__name__)


def objective(
    trial: optuna.trial.Trial,
    tuning_cfg: TuningConfig,
    ref_embeddings: np.ndarray,
    embedding_extractor: ImageEmbeddingExtractor,
    **precomputed_kwargs,
) -> float:
    """
    Optuna objective function for hyperparameter tuning of synthetic microtubule data generation.

    Args:
        trial (optuna.trial.Trial): The Optuna trial object.
        tuning_cfg (TuningConfig): The overall tuning configuration.
        ref_embeddings (np.ndarray): Pre-computed embeddings from reference data.
        embedding_extractor (ImageEmbeddingExtractor): Initialized embedding extractor.
        **precomputed_kwargs: Dictionary of pre-computed metric arguments (e.g., ref_mean, ref_inv_cov).

    Returns:
        float: The similarity score for the current trial's synthetic data, to be maximized.
              Returns -inf if an error occurs during generation or evaluation.
    """
    logger.debug(f"--- Starting Optuna Trial {trial.number} ---")
    logger.debug(f"Trial parameters being evaluated: {trial.params}")

    try:
        # Generate synthetic data config for this trial
        logger.debug(f"Trial {trial.number}: Creating SyntheticDataConfig from trial parameters.")
        cfg = tuning_cfg.suggest_synthetic_config_from_trial(trial)

        # Extract embeddings from the newly generated synthetic data
        num_eval_frames = tuning_cfg.num_compare_frames
        logger.debug(
            f"Trial {trial.number}: Generating and extracting embeddings for {num_eval_frames} frames."
        )
        synthetic_embeddings = embedding_extractor.extract_from_synthetic_config(
            cfg, num_eval_frames
        )

        if synthetic_embeddings is None or synthetic_embeddings.size == 0:
            logger.warning(
                f"Trial {trial.number}: No synthetic embeddings extracted. Returning -inf."
            )
            return -float("inf")  # Cannot proceed without embeddings

        # Evaluate this new configuration against the reference embeddings.
        logger.debug(
            f"Trial {trial.number}: Computing similarity score using metric '{tuning_cfg.similarity_metric}'."
        )
        score = similarity(
            tuning_cfg=tuning_cfg,
            ref_embeddings=ref_embeddings,
            synthetic_embeddings=synthetic_embeddings,
            **precomputed_kwargs,
        )
        logger.debug(f"--- Optuna Trial {trial.number} completed. Score: {score:.6f} ---")

    except optuna.exceptions.TrialPruned as e:
        logger.info(f"Trial {trial.number} pruned: {e}")
        # Re-raise Optuna's pruning exception directly
        raise
    except ValueError as e:
        # Catch validation errors from suggest_synthetic_config_from_trial or other config issues
        logger.error(
            f"Trial {trial.number}: Configuration or data generation failed due to validation error: {e}",
            exc_info=False,
        )
        # Log exc_info=True if you need full traceback for all ValueError types in debug logs
        score = -float("inf")  # Assign a very bad score
    except Exception as e:
        # Catch any other unexpected errors during the trial
        logger.error(
            f"Trial {trial.number}: An unexpected error occurred during objective evaluation: {e}",
            exc_info=True,
        )
        score = -float("inf")  # Assign a very bad score

    return score

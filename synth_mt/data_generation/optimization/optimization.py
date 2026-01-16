import logging
import os
from functools import partial

import optuna
from optuna.trial import TrialState

from .embeddings import ImageEmbeddingExtractor
from .metrics import precompute_matric_args
from .objective import objective
from ...config.tuning import TuningConfig

logger = logging.getLogger(__name__)


def run_optimization(tuning_config_path: str):
    """
    Runs the Optuna optimization study and saves the results.

    This function performs the expensive model setup and optimization, then saves
    the best configuration and the Optuna study database for later evaluation.
    """
    logger.info(f"\n{'=' * 80}\nStarting OPTIMIZATION for: {tuning_config_path}\n{'=' * 80}")

    try:
        logger.debug("--- Step 1: Loading tuning configuration ---")
        tuning_cfg = TuningConfig.load(tuning_config_path)
        tuning_cfg.validate()
        tuning_cfg.to_json(tuning_config_path)

    except Exception as e:
        logger.critical(
            f"An unexpected error occurred during tuning config loading/validation: {e}",
            exc_info=True,
        )
        raise e

    # Ensure tuning_cfg is available before proceeding
    if tuning_cfg is None:
        logger.critical("Tuning configuration not available. Exiting optimization.")
        return

    logger.debug("Performing model setup and reference embedding extraction")
    embedding_extractor = ImageEmbeddingExtractor(tuning_cfg)
    ref_embeddings = embedding_extractor.extract_from_references()
    precomputed_kwargs = precompute_matric_args(tuning_cfg, ref_embeddings)

    # Ensure essential elements are available before proceeding to optimization
    if embedding_extractor is None or ref_embeddings is None:
        logger.critical(
            "Essential components (embedding extractor or reference embeddings) are missing."
        )
        return

    logger.debug("--- Step 3: Running Optuna optimization ---")
    db_filename = f"{tuning_cfg.output_config_id}.db"
    db_filepath = os.path.join(tuning_cfg.temp_dir, db_filename)

    os.makedirs(tuning_cfg.temp_dir, exist_ok=True)
    logger.debug(f"Ensured temporary directory exists: {tuning_cfg.temp_dir}")

    storage_uri = f"sqlite:///{db_filepath}"
    logger.debug(f"Using Optuna storage URI: {storage_uri}")

    sampler = optuna.samplers.TPESampler()

    # Ensure a clean study if requested
    if not tuning_cfg.load_if_exists:
        if os.path.exists(db_filepath):
            logger.info(f"Deleting existing Optuna study database: {db_filepath}")
            os.remove(db_filepath)

    study = optuna.create_study(
        sampler=sampler,
        study_name=tuning_cfg.output_config_id,
        storage=storage_uri,
        direction=tuning_cfg.direction,
        load_if_exists=tuning_cfg.load_if_exists,
    )
    logger.debug(
        f"Optuna study '{tuning_cfg.output_config_id}' created/loaded. "
        f"Direction: '{tuning_cfg.direction}', "
        f"Load if exists: {tuning_cfg.load_if_exists}."
    )

    n_existing_trials = len(study.get_trials(states=[TrialState.COMPLETE]))
    n_trials_to_run = tuning_cfg.num_trials - n_existing_trials

    if n_trials_to_run > 0:
        logger.info(f"Starting optimization for {n_trials_to_run} trials.")
        # Use partial to pass the pre-computed objects to the objective function
        objective_fcn = partial(
            objective,
            tuning_cfg=tuning_cfg,
            ref_embeddings=ref_embeddings,
            embedding_extractor=embedding_extractor,
            **precomputed_kwargs,
        )

        study.optimize(objective_fcn, n_trials=n_trials_to_run)
        logger.info(f"Optimization finished after {n_trials_to_run} new trials.")
    else:
        logger.info(
            f"Study already has {n_existing_trials} trials, which meets or exceeds the target of {tuning_cfg.num_trials}. No new trials will be run."
        )

    if study.best_trial:
        logger.info(
            f"Best trial found (Trial {study.best_trial.number}): Value = {study.best_trial.value:.6f}"
        )
    else:
        logger.warning("No best trial found. The study might have had no completed trials.")

    logger.info(f"\n{'=' * 80}\nOPTIMIZATION process completed.\n{'=' * 80}")

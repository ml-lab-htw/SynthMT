import logging
import os
from typing import List, Tuple

import numpy as np
import optuna

from ..video import generate_video, generate_frames
from ...config.synthetic_data import SyntheticDataConfig
from ...config.tuning import TuningConfig

logger = logging.getLogger(__name__)


def evaluate_tuning_cfg(tuning_config_path: str, output_dir: str):
    logger.debug(f"{'=' * 80}\nStarting EVALUATION for: {tuning_config_path}\n{'=' * 80}")

    logger.debug("--- Loading configurations and study results ---")
    tuning_cfg = TuningConfig.load(tuning_config_path)

    # Ensure folders exist for output and temporary files
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tuning_cfg.temp_dir, exist_ok=True)

    # Load the completed Optuna study from its database file
    study_db_path = os.path.join(tuning_cfg.temp_dir, f"{tuning_cfg.output_config_id}.db")

    if not os.path.exists(study_db_path):
        logger.info(f"Study database file not found: {study_db_path}")
        return tuning_cfg.output_config_id, 0, 0.0

    full_study_db_uri = f"sqlite:///{study_db_path}"
    logger.debug(f"Attempting to load Optuna study from: {full_study_db_uri}")

    study = optuna.load_study(study_name=tuning_cfg.output_config_id, storage=full_study_db_uri)
    logger.debug(f"Loaded Optuna study '{tuning_cfg.output_config_id}' from: {full_study_db_uri}")

    # Choose top-N
    top_n = tuning_cfg.output_config_num_best
    top_trials = []

    if top_n > 1:
        # trials = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
        trials = study.trials
        sorted_trials = sorted(
            [t for t in trials if t.value is not None], key=lambda t: t.value, reverse=True
        )
        number_of_trials = len(sorted_trials)
        if number_of_trials > top_n:
            top_trials = sorted_trials[:top_n]
    else:
        top_trials = [study.best_trial]
        number_of_trials = 1  # TODO: Calculate actual number of trials

    for i, trial in enumerate(top_trials):
        logger.info(f"Trial {i + 1}/{top_n}: Value = {trial.value:.4f}")

        current_cfg = SyntheticDataConfig.from_trial(trial)

        current_cfg.num_frames = tuning_cfg.output_config_num_frames
        current_cfg.id = f"{tuning_cfg.output_config_id}_rank_{i + 1}"
        current_cfg.generate_mt_mask = True
        current_cfg.generate_seed_mask = False

        evaluate_synthetic_data_cfg(
            cfg=current_cfg,
            tuning_cfg=tuning_cfg,
            output_dir=output_dir,
            is_for_expert_validation=True,
        )

    logger.debug("Evaluation complete.")
    return study, number_of_trials, top_trials[0].number


def evaluate_synthetic_data_cfg(
    cfg: SyntheticDataConfig,
    tuning_cfg: TuningConfig,
    output_dir: str | None = None,
    is_for_expert_validation: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Evaluates a specific SyntheticDataConfig against reference data.
    """

    frames: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    if output_dir is None:
        frame_generator = generate_frames(
            cfg,
            cfg.num_frames,
            return_mt_mask=cfg.generate_mt_mask,
            return_seed_mask=cfg.generate_seed_mask,
        )

        for frame, gt_data, mt_mask, seed_mask, frame_idx in frame_generator:
            frames.append(frame)
            masks.append(mt_mask)
    else:
        frames = generate_video(
            cfg,
            output_dir,
            num_png_frames=tuning_cfg.output_config_num_png,
            is_for_expert_validation=is_for_expert_validation,
        )

    return frames, masks

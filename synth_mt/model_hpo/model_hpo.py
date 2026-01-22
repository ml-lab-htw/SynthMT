import argparse
import json
import os
from pathlib import Path

import numpy as np
import optuna
from optuna import Trial
from optuna.trial import TrialState

from synth_mt.benchmark.dataset import BenchmarkDataset
from synth_mt.benchmark.metrics import calculate_downstream_metrics, calculate_segmentation_metrics
from synth_mt.benchmark.models.anchor_point_model import AnchorPointModel
from synth_mt.benchmark.models.factory import ModelFactory, setup_model_factory
from synth_mt.utils.logger import setup_logging
from synth_mt.utils.matlab import matlab_engine
from synth_mt.utils.postprocessing import (
    filter_anchor_points,
    filter_instance_masks,
    get_area_length_ranges,
)
from synth_mt.utils.preprocessing import process_image

logger = setup_logging(log_dir=".logs")


def define_search_space(trial, model_name):
    """Defines the hyperparameter search space for a given model using Optuna."""
    params = {}

    # Shared preprocessing parameters
    params["percentile_min"] = trial.suggest_float("percentile_min", 0, 5)
    params["percentile_max"] = trial.suggest_float("percentile_max", 95, 100)
    params["sharpen_radius"] = trial.suggest_float("sharpen_radius", 0, 10)
    params["smooth_radius"] = trial.suggest_float("smooth_radius", 0, 10)
    params["rescale_using_percentiles"] = trial.suggest_categorical(
        "rescale_using_percentiles", [True, False]
    )
    params["invert"] = trial.suggest_categorical("invert", [True, False])
    params["histogram_normalization"] = trial.suggest_categorical(
        "histogram_normalization", [True, False]
    )
    params["clip_to_percentiles"] = trial.suggest_categorical("clip_to_percentiles", [True, False])

    # Model-specific parameters
    model_name = model_name.lower()
    if model_name == "cellposesam":
        params["cellprob_threshold"] = trial.suggest_float("cellprob_threshold", 0.0, 1.0)
        params["diameter"] = trial.suggest_int("diameter", 15, 40)
        params["flow_threshold"] = trial.suggest_float("flow_threshold", 0.0, 1.0)
        params["grayscale"] = trial.suggest_categorical("grayscale", [True, False])

    elif model_name == "cellsam":
        params["bbox_threshold"] = trial.suggest_float("bbox_threshold", 0.0, 1.0)
        params["grayscale"] = trial.suggest_categorical("grayscale", [True, False])

    elif model_name == "microsam":
        params["boundary_distance_threshold"] = trial.suggest_float(
            "boundary_distance_threshold", 0.0, 1.0
        )
        params["center_distance_threshold"] = trial.suggest_float(
            "center_distance_threshold", 0.0, 1.0
        )
        params["distance_smoothing"] = trial.suggest_float("distance_smoothing", 0.0, 5.0)
        params["foreground_smoothing"] = trial.suggest_float("foreground_smoothing", 0.0, 3.0)
        params["foreground_threshold"] = trial.suggest_float("foreground_threshold", 0.0, 1.0)
        params["grayscale"] = trial.suggest_categorical("grayscale", [True, False])

    elif model_name == "sam":
        params["pred_iou_thresh"] = trial.suggest_float("pred_iou_thresh", 0.5, 0.99)
        params["stability_score_thresh"] = trial.suggest_float("stability_score_thresh", 0.5, 0.99)
        params["grayscale"] = trial.suggest_categorical("grayscale", [True, False])

    elif model_name == "sam2":
        params["pred_iou_thresh"] = trial.suggest_float("pred_iou_thresh", 0.0, 1.0)
        params["stability_score_thresh"] = trial.suggest_float("stability_score_thresh", 0.0, 1.0)
        params["grayscale"] = trial.suggest_categorical("grayscale", [True, False])

    elif model_name == "sam3text":
        params["threshold"] = trial.suggest_float("threshold", 0.0, 1.0)
        params["mask_threshold"] = trial.suggest_float("mask_threshold", 0.0, 1.0)
        params["text_prompt_option"] = trial.suggest_int("text_prompt_option", 0, 9)
        params["grayscale"] = trial.suggest_categorical("grayscale", [True, False])

    elif model_name == "sam3":
        params["pred_iou_thresh"] = trial.suggest_float("pred_iou_thresh", 0.0, 1.0)
        params["stability_score_thresh"] = trial.suggest_float("stability_score_thresh", 0.0, 1.0)
        params["grayscale"] = trial.suggest_categorical("grayscale", [True, False])

    elif model_name == "fiesta":
        params["background_filter"] = trial.suggest_categorical("background_filter", [True, False])
        params["binary_image_processing"] = trial.suggest_categorical(
            "binary_image_processing", ["average", "none"]
        )
        params["dynamicfil"] = trial.suggest_categorical("dynamicfil", [True, False])
        params["focus_correction"] = trial.suggest_categorical("focus_correction", [True, False])
        params["fwhm_estimate"] = trial.suggest_float("fwhm_estimate", 2.6, 10.0)
        params["height_threshold"] = trial.suggest_float("height_threshold", 0.0, 2.0)
        params["min_cod"] = trial.suggest_float("min_cod", 0.0, 0.1)
        params["reduce_fit_box"] = trial.suggest_float("reduce_fit_box", 1.0, 3.0)
        params["grayscale"] = True

    elif model_name == "stardist":
        params["pretrained"] = trial.suggest_categorical(
            "pretrained", ["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"]
        )
        params["prob_thresh"] = trial.suggest_float("prob_thresh", 0.0, 1.0)
        params["nms_thresh"] = trial.suggest_float("nms_thresh", 0.0, 1.0)
        params["grayscale"] = True

    elif model_name == "tardis":
        params["cnn_threshold"] = trial.suggest_float("cnn_threshold", 0.0, 1.0)
        params["dist_threshold"] = trial.suggest_float("dist_threshold", 0.0, 1.0)
        params["grayscale"] = True

    else:
        raise ValueError("Unknown model: {}".format(model_name))

    return params


def objective_function(
    trial: Trial,
    factory: ModelFactory,
    model_name: str,
    dataset: BenchmarkDataset,
    postprocessing_props: dict,
    metric: str = "IoU",
    temp_dir: str = ".temp",
    models_dir: str = ".models",
    use_skeleton: bool = True,
):
    """
    Objective function for Optuna hyperparameter optimization.
    Returns the mean IoU.
    """

    if "SK" in metric and not use_skeleton:
        use_skeleton = True

    if "IoU" in metric:
        default_rv = 0.0
    elif "KL" in metric:
        default_rv = np.inf
    else:
        raise ValueError("Unknown metric: {}".format(metric))

    try:

        params = define_search_space(trial, model_name)
        params["save_dir"] = models_dir
        params["work_dir"] = temp_dir

        model = factory.create_model(model_name, **params)
        model.load_model()

        preprocess_params = {
            "grayscale": params["grayscale"],
            "sharpen_radius": params["sharpen_radius"],
            "smooth_radius": params["smooth_radius"],
            "percentile_min": params["percentile_min"],
            "percentile_max": params["percentile_max"],
            "clip_to_percentiles": params["clip_to_percentiles"],
            "rescale_using_percentiles": params["rescale_using_percentiles"],
            "invert": params["invert"],
            "histogram_normalization": params["histogram_normalization"],
        }

        processed_images = []
        all_gt_masks = []
        for i in range(len(dataset)):
            image, gt_mask, _ = dataset[i]
            processed_image = process_image(image, **preprocess_params)
            processed_images.append(processed_image)
            all_gt_masks.append(gt_mask)

        # Batch prediction
        predictions = model.predict_batch(processed_images)
        all_pred_masks = []
        all_anchor_points = []

        if isinstance(model, AnchorPointModel):
            for anchor_points in predictions:
                anchor_points = filter_anchor_points(
                    anchor_points,
                    min_length=postprocessing_props["min_length"],
                    max_length=postprocessing_props["max_length"],
                    image_resolution=processed_images[0].shape[
                        :2
                    ],  # Assuming all images have same resolution
                    border_margin=-1,
                )
                all_anchor_points.append(anchor_points)
            all_pred_masks = [None] * len(dataset)
        else:
            for pred_mask in predictions:
                pred_mask = filter_instance_masks(
                    pred_mask,
                    min_area=postprocessing_props["min_area"],
                    max_area=postprocessing_props["max_area"],
                    min_length=postprocessing_props["min_length"],
                    max_length=postprocessing_props["max_length"],
                    border_margin=-1,
                )
                all_pred_masks.append(pred_mask)
            all_anchor_points = [None] * len(dataset)

        if metric == "KL":
            down_metrics = calculate_downstream_metrics(
                pred_masks=all_pred_masks,
                gt_masks=all_gt_masks,
                anchor_points_instance_masks=all_anchor_points,
            )
            mean_score = down_metrics["Length_KL"]
        elif "IoU" in metric:
            seg_metrics, _ = calculate_segmentation_metrics(
                pred_masks=all_pred_masks,
                gt_masks=all_gt_masks,
                anchor_points_instance_masks=all_anchor_points,
                use_skeletonized_version=use_skeleton,
            )
            if use_skeleton:
                mean_score = seg_metrics["SKIoU_mean"]
            else:
                mean_score = seg_metrics["IoU/T_mean"]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return mean_score

    except Exception as e:
        logger.warning(
            f"Trial {trial.number} for {model_name} -> Failed with error: {e}. Reporting as failed."
        )
        return default_rv
        # raise e


def main():
    parser = argparse.ArgumentParser(description="Optimize model hyperparameters.")
    parser.add_argument("config_path", type=str, help="Path to the optimization config JSON file.")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    DATASET_PATH = config["DATASET_PATH"]
    N_SAMPLES = config["N_SAMPLES"]
    N_TRIALS = config["N_TRIALS"]
    TEMP_DIR = config["TEMP_DIR"]
    MODEL_DIR = config["MODEL_DIR"]
    MODELS = config["MODELS"]
    METRIC = config["METRIC"]
    LOAD_IF_EXISTS = config["LOAD_IF_EXISTS"]
    USE_SKELETON = config["USE_SKELETON"]

    # minimize for KL, maximize for IoU
    if METRIC == "KL":
        direction = "minimize"
    elif METRIC == "IoU":
        direction = "maximize"
    else:
        raise ValueError(f"Unknown metric: {METRIC}")

    logger.info(f"Loading dataset from: {DATASET_PATH} with {N_SAMPLES} samples.")
    dataset = BenchmarkDataset(
        DATASET_PATH,
        num_samples=N_SAMPLES,
        mask_mapping=("images", "image_masks"),
    )
    dataset_full = BenchmarkDataset(
        DATASET_PATH,
        num_samples=-1,
        mask_mapping=("images", "image_masks"),
    )

    logger.info(f"HPO will be performed on {len(dataset)} images:")
    for i in range(len(dataset)):
        image_path = dataset.get_image_path(i)
        logger.info({image_path})

    factory = setup_model_factory()
    available_models = factory.get_available_models()

    # Create a directory for Optuna studies
    temp_dir = Path(TEMP_DIR)
    logger.info(f"Creating Optuna study directory: {temp_dir}")
    temp_dir.mkdir(exist_ok=True)

    logger.info(f"Requested models for optimization: {[m.lower() for m in MODELS]}")
    logger.info(f"Available models for optimization: {available_models}")
    optimizable_models = [m for m in MODELS if m.lower() in available_models]

    logger.info(f"Number of trials: {N_TRIALS}")
    logger.info(f"Metric: {METRIC}")
    logger.info(f"Use skeleton for IoU calculation: {USE_SKELETON}")
    logger.info(f"Optimization direction: {direction}")

    for model_name in optimizable_models:
        logger.info(f"--- Optimizing model: {model_name} ---")

        study_name = f"{model_name.lower()}_{METRIC}_{N_SAMPLES}"

        if USE_SKELETON:
            study_name += "_skel"

        db_filepath = f"{temp_dir / study_name}.db"
        storage_url = f"sqlite:///{db_filepath}"

        # Ensure a clean study if requested
        if not LOAD_IF_EXISTS:
            if os.path.exists(db_filepath):
                logger.info(f"Deleting existing Optuna study database: {db_filepath}")
                os.remove(db_filepath)

        logger.info(f"Optuna storage URL: {storage_url}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction=direction,  # Minimize the score (e.g., KL divergence)
            load_if_exists=True,
        )

        n_existing_trials = len(study.get_trials(states=[TrialState.COMPLETE]))
        n_trials_to_run = N_TRIALS - n_existing_trials

        factory = setup_model_factory()
        min_area, max_area, min_length, max_length = get_area_length_ranges(dataset_full)
        postprocessing_props = {
            "min_area": min_area,
            "max_area": max_area,
            "min_length": min_length,
            "max_length": max_length,
        }

        def run_optimization():
            if n_trials_to_run > 0:
                logger.info(f"Starting optimization for {n_trials_to_run} trials.")
                study.optimize(
                    lambda trial: objective_function(
                        trial,
                        factory,
                        model_name,
                        dataset,
                        postprocessing_props,
                        METRIC,
                        TEMP_DIR,
                        MODEL_DIR,
                        USE_SKELETON,
                    ),
                    n_trials=n_trials_to_run,
                )
            else:
                logger.info(
                    f"n_trials={n_existing_trials} > {N_TRIALS}. No new trials will be run."
                )

        if model_name.lower() == "fiesta":
            with matlab_engine():
                run_optimization()
        else:
            run_optimization()

        found_any_valid = any(
            trial.value is not None and not np.isnan(trial.value) for trial in study.trials
        )

        if not found_any_valid:
            logger.warning(f"No valid trials for '{model_name}'. Skipping saving best parameters.")
            continue

        best_params = study.best_params
        best_params["save_dir"] = MODEL_DIR
        best_params["work_dir"] = TEMP_DIR

        best_score = study.best_value

        logger.info(f"Best score for {study_name}: {best_score:.4f} ({METRIC})")
        logger.info(f"Best parameters for {study_name}: {best_params}")

        # Save the best parameters to a JSON file
        config_path = Path(f"mt/benchmark/models/{study_name}.json")
        logger.info(f"Saving best parameters to {config_path}")

        model_instance = factory.create_model(model_name, **best_params)
        model_instance.save(str(config_path))


if __name__ == "__main__":
    main()

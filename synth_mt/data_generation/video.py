import concurrent.futures
import logging
import os
import random
from typing import List, Tuple, Optional, Dict, Any

import albumentations as A
import imageio
import numpy as np
from tqdm import tqdm

from synth_mt.config.synthetic_data import SyntheticDataConfig
from synth_mt.data_generation import utils
from synth_mt.data_generation.microtubule import Microtubule
from synth_mt.data_generation.spots import SpotGenerator
from synth_mt.file_io import OutputManager

logger = logging.getLogger(__name__)


def draw_mt(
    mt: Microtubule,
    cfg: SyntheticDataConfig,
    frame,
    frame_idx,
    return_seed_mask,
    jitter,
):
    mt.step(cfg)
    mt.base_point += jitter
    local_frame = np.zeros_like(frame)
    local_mt_mask = np.zeros(cfg.img_size, dtype=np.uint16)
    local_seed_mask = np.zeros(cfg.img_size, dtype=np.uint16)
    gt_info = mt.draw(
        frame=local_frame,
        mt_mask=local_mt_mask,
        cfg=cfg,
        seed_mask=(local_seed_mask if frame_idx == 0 and return_seed_mask else None),
    )
    mt.base_point -= jitter
    return local_frame, local_mt_mask, local_seed_mask, gt_info


def render_frame(
    cfg: SyntheticDataConfig,
    mts: List[Microtubule],
    frame_idx: int,
    fixed_spot_generator: SpotGenerator,
    moving_spot_generator: SpotGenerator,
    aug_pipeline: Optional[A.Compose] = None,
    return_mt_mask: bool = False,
    return_seed_mask: bool = False,
    debug_steps: bool = False,
) -> Tuple[
    np.ndarray,
    List[Dict[str, Any]],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    logger.debug(f"Rendering frame {frame_idx} for series ID {cfg.id}...")

    # ─── Initialization ──────────────────────────────────────────
    # Initialize background as a float32 array in the range [0, 1]
    frame = np.full((*cfg.img_size, 3), cfg.background_level, dtype=np.float32)
    mt_mask = (
        np.zeros((len(mts), *cfg.img_size), dtype=np.uint16)
        if return_mt_mask
        else None
    )

    if debug_steps:
        # save intermediate frames for debugging
        debug_output_dir = "debug_output"
        debug_dir = os.path.join(debug_output_dir, f"series_{cfg.id}", f"frame_{frame_idx:04d}")
        os.makedirs(debug_dir, exist_ok=True)
        imageio.imwrite(
            os.path.join(debug_dir, "01_initial_frame.png"),
            im_to_int(frame),
            compress_level=6,
        )

    # seed_mask is only for the first frame and if requested
    seed_mask = None
    if frame_idx == 0 and return_seed_mask:
        seed_mask = (
            np.zeros((len(mts), *cfg.img_size), dtype=np.uint16)
            if return_seed_mask
            else None
        )

    gt_data: List[Dict[str, Any]] = []

    # Jitter is applied to microtubule base points
    jitter = (
        np.random.normal(0, cfg.jitter_px, 2).astype(np.float32)
        if cfg.jitter_px > 0
        else np.zeros(2, dtype=np.float32)
    )
    if cfg.jitter_px > 0:
        logger.debug(f"Frame {frame_idx}: Applying jitter: {jitter.tolist()}")

    # ─── Simulate and Draw Microtubules (Parallelized) ──────────
    args = [
        (mt, cfg, frame, frame_idx, return_seed_mask, jitter) for mt in mts
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(draw_mt, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Sum all microtubule results into main frame/mask
    for i, (local_frame, local_mt_mask, local_seed_mask, gt_info) in enumerate(results):
        frame += local_frame
        if mt_mask is not None and local_mt_mask is not None:
            mt_mask[i] = local_mt_mask
        if seed_mask is not None and local_seed_mask is not None:
            seed_mask[i] = local_seed_mask
        gt_data.extend(gt_info)

    if debug_steps:
        imageio.imwrite(
            os.path.join(debug_dir, "02_after_drawing_mts.png"),
            im_to_int(frame),
            compress_level=6,
        )

    # ─── Add Ancillary Objects (Spots) ───────────────────────────
    try:
        frame = fixed_spot_generator.apply(frame)
        logger.debug(f"Frame {frame_idx}: Fixed spots applied.")
        if debug_steps:
            imageio.imwrite(
                os.path.join(debug_dir, "03_after_fixed_spots.png"),
                im_to_int(frame),
                compress_level=6,
            )
        frame = moving_spot_generator.apply(frame)
        logger.debug(f"Frame {frame_idx}: Moving spots applied.")
        if debug_steps:
            imageio.imwrite(
                os.path.join(debug_dir, "04_after_moving_spots.png"),
                im_to_int(frame),
                compress_level=6,
            )
        frame = SpotGenerator.apply_random_spots(frame, cfg.random_spots)
        logger.debug(f"Frame {frame_idx}: Random spots applied.")
        moving_spot_generator.update()  # Update moving spot positions for next frame
        logger.debug(f"Frame {frame_idx}: Moving spot generator updated for next frame.")
        if debug_steps:
            imageio.imwrite(
                os.path.join(debug_dir, "05_after_random_spots.png"),
                im_to_int(frame),
                compress_level=6,
            )
    except Exception as e:
        logger.error(f"Frame {frame_idx}: Error applying spots: {e}", exc_info=True)

    # ─── Apply Photophysics and Camera Effects ───────────────────
    logger.debug(f"Frame {frame_idx}: Applying photophysics and camera effects.")
    try:
        vignette = utils.compute_vignette(cfg)
        frame *= vignette[..., np.newaxis]
        logger.debug(f"Frame {frame_idx}: Applied vignetting.")

        if debug_steps:
            imageio.imwrite(
                os.path.join(debug_dir, "06_after_vignetting.png"),
                im_to_int(frame),
                compress_level=6,
            )

        # Apply red channel noise if specified
        if cfg.red_channel_noise_std > 0.0:
            red_noise = np.random.normal(0, cfg.red_channel_noise_std, frame.shape[:2]).astype(
                np.float32
            )
            red_ch_idx = 2  # OpenCV uses BGR format
            frame[..., red_ch_idx] += red_noise
            frame[..., red_ch_idx] = np.clip(
                frame[..., red_ch_idx], 0, 1
            )  # Ensure red channel stays in [0, 1]
            logger.debug(
                f"Frame {frame_idx}: Applied red channel noise (std={cfg.red_channel_noise_std:.4f})."
            )
            if debug_steps:
                imageio.imwrite(
                    os.path.join(debug_dir, "07_after_red_channel_noise.png"),
                    im_to_int(frame),
                    compress_level=6,
                )
        else:
            logger.debug(
                f"Frame {frame_idx}: Skipping red channel noise (std={cfg.red_channel_noise_std:.4f})."
            )

        if cfg.quantum_efficiency > 0:
            frame[frame < 0] = 0  # Clamp negative values before Poisson noise
            frame = np.random.poisson(frame * cfg.quantum_efficiency) / cfg.quantum_efficiency
            logger.debug(
                f"Frame {frame_idx}: Applied Poisson noise (QE={cfg.quantum_efficiency:.2f})."
            )
            if debug_steps:
                imageio.imwrite(
                    os.path.join(debug_dir, "08_after_poisson_noise.png"),
                    im_to_int(frame),
                    compress_level=6,
                )
        else:
            logger.debug(
                f"Frame {frame_idx}: Skipping Poisson noise (QE={cfg.quantum_efficiency:.2f})."
            )

        if cfg.gaussian_noise > 0.0:
            frame += np.random.normal(0, cfg.gaussian_noise, frame.shape).astype(np.float32)
            logger.debug(
                f"Frame {frame_idx}: Applied Gaussian noise (std={cfg.gaussian_noise:.4f})."
            )
            if debug_steps:
                imageio.imwrite(
                    os.path.join(debug_dir, "09_after_gaussian_noise.png"),
                    im_to_int(frame),
                    compress_level=6,
                )
        else:
            logger.debug(
                f"Frame {frame_idx}: Skipping Gaussian noise (std={cfg.gaussian_noise:.4f})."
            )

        frame = utils.apply_global_blur(frame, cfg)
        logger.debug(f"Frame {frame_idx}: Applied global blur (sigma={cfg.global_blur_sigma:.2f}).")

        if debug_steps:
            imageio.imwrite(
                os.path.join(debug_dir, "10_after_global_blur.png"),
                im_to_int(frame),
                compress_level=6,
            )
        # show_frame(frame, title="Before Albumentations")

        if cfg.global_contrast > 0.0:
            frame = utils.apply_contrast(frame, cfg.global_contrast)
            logger.debug(
                f"Frame {frame_idx}: Applied contrast adjustment (factor={cfg.global_contrast:.2f})."
            )
            if debug_steps:
                imageio.imwrite(
                    os.path.join(debug_dir, "11_after_contrast_adjustment.png"),
                    im_to_int(frame),
                    compress_level=6,
                )

        if cfg.global_brightness > 0.0:
            frame = utils.apply_brightness(frame, cfg.global_brightness)
            logger.debug(
                f"Frame {frame_idx}: Applied brightness adjustment (factor={cfg.global_brightness:.2f})."
            )
            if debug_steps:
                imageio.imwrite(
                    os.path.join(debug_dir, "12_after_brightness_adjustment.png"),
                    im_to_int(frame),
                    compress_level=6,
                )

        # show_frame(frame, title="After Albumentations")

    except Exception as e:
        logger.error(
            f"Frame {frame_idx}: Error applying photophysics/camera effects: {e}", exc_info=True
        )

    # ─── Apply Augmentations ────────────────────────────────
    if aug_pipeline and cfg.albumentations and cfg.albumentations["p"] > 0:
        logger.debug(f"Frame {frame_idx}: Applying Albumentations (p={cfg.albumentations['p']:.2f}).")
        try:
            # Albumentations expects uint8 or float, ensure float [0,1]
            # Convert back to original frame type if needed
            # For simplicity, passing float32 as is.
            masks_to_augment = []
            if mt_mask is not None:
                masks_to_augment.append(mt_mask)
            if seed_mask is not None:
                masks_to_augment.append(seed_mask)

            if masks_to_augment:
                # Albumentations typically handles masks as (H, W) or (H, W, C).
                # For instance masks, it's better to apply geometric transformations per mask.
                # Here, we transpose to (H, W, N) and back.
                transposed_masks = [m.transpose(1, 2, 0) for m in masks_to_augment]
                augmented = aug_pipeline(image=frame, masks=transposed_masks)
                frame = augmented["image"]
                augmented_masks = [m.transpose(2, 0, 1) for m in augmented["masks"]]

                mask_idx = 0
                if mt_mask is not None:
                    mt_mask = augmented_masks[mask_idx]
                    mask_idx += 1
                if seed_mask is not None:
                    seed_mask = augmented_masks[mask_idx]
            else:
                augmented = aug_pipeline(image=frame)
                frame = augmented["image"]

            logger.debug(f"Frame {frame_idx}: Albumentations applied.")

        except Exception as e:
            logger.error(f"Frame {frame_idx}: Error applying Albumentations: {e}", exc_info=True)
            # Log error but don't stop rendering the frame
    elif aug_pipeline and cfg.albumentations and cfg.albumentations["p"] == 0:
        logger.debug(
            f"Frame {frame_idx}: Albumentations pipeline exists but master probability (p) is 0. Skipping."
        )
    else:
        logger.debug(f"Frame {frame_idx}: No Albumentations pipeline or config. Skipping.")

    # ─── Finalization and Formatting ─────────────────────────────
    logger.debug(f"Frame {frame_idx}: Finalizing frame and ground truth.")
    try:
        frame = utils.annotate_frame(frame, cfg, frame_idx)
        logger.debug(f"Frame {frame_idx}: Annotations applied.")
    except Exception as e:
        logger.error(f"Frame {frame_idx}: Error annotating frame: {e}", exc_info=True)

    frame_uint8 = im_to_int(frame)
    logger.debug(f"Frame {frame_idx}: Converted to uint8 (clipping applied).")

    # Add frame index to the ground truth data.
    for entry in gt_data:
        entry["frame_index"] = frame_idx
    logger.debug(
        f"Frame {frame_idx}: Ground truth data updated with frame_index. Total segments: {len(gt_data)}."
    )

    final_mt_mask = mt_mask if return_mt_mask else None
    final_seed_mask = seed_mask if return_seed_mask else None

    if debug_steps:
        imageio.imwrite(
            os.path.join(debug_dir, "13_final_frame.png"),
            frame_uint8,
            compress_level=6,
        )

        mask_export = np.max(final_mt_mask, axis=0)
        imageio.imwrite(
            os.path.join(debug_dir, "14_final_mt_mask.png"),
            im_to_int(mask_export > 0),
            compress_level=6,
        )
    # sort mt_mask and seed_mask by instance_id if they exist
    final_mt_mask = sort_instance_stack(final_mt_mask)
    final_seed_mask = sort_instance_stack(final_seed_mask)

    logger.debug(f"Frame {frame_idx} rendering complete.")
    return frame_uint8, gt_data, final_mt_mask, final_seed_mask


def im_to_int(frame: np.ndarray) -> np.ndarray:
    frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return frame_uint8


def sort_instance_stack(instance_stack: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if instance_stack is not None:
        instance_ids = [np.max(instance_stack[i]) for i in range(instance_stack.shape[0])]
        sorted_indices = np.argsort(instance_ids)
        instance_stack = instance_stack[sorted_indices]
    return instance_stack


def generate_frames(
    cfg: SyntheticDataConfig,
    num_frames: int,
    return_mt_mask: bool = False,
    return_seed_mask: bool = False,
):
    """
    Generates a sequence of synthetic video frames and associated data.
    This is a generator function, yielding one frame's data at a time.
    """
    logger.debug(f"Preparing to generate {num_frames} frames for series ID {cfg.id}...")

    mts: List[Microtubule] = []
    try:
        start_points = utils.build_motion_seeds(cfg)
        logger.debug(f"Built {len(start_points)} initial motion seeds for microtubules.")
        for idx, start_pt in enumerate(start_points, start=1):
            mts.append(
                Microtubule(
                    cfg=cfg,
                    base_point=start_pt,
                    instance_id=idx,
                )
            )
        logger.debug(f"Initialized {len(mts)} microtubules.")
    except Exception as e:
        logger.critical(f"Failed to initialize microtubules: {e}", exc_info=True)
        # If MTs cannot be initialized, we cannot generate frames. Raise or return empty.
        raise

    try:
        fixed_spot_generator = SpotGenerator(cfg.fixed_spots, cfg.img_size)
        logger.debug(f"Initialized fixed spot generator with {cfg.fixed_spots.count} spots.")
        moving_spot_generator = SpotGenerator(cfg.moving_spots, cfg.img_size)
        logger.debug(f"Initialized moving spot generator with {cfg.moving_spots.count} spots.")
    except Exception as e:
        logger.critical(f"Failed to initialize spot generators: {e}", exc_info=True)
        raise  # Critical setup failure

    aug_pipeline: Optional[A.Compose] = None
    try:
        if cfg.albumentations:
            aug_pipeline = utils.build_albumentations_pipeline(cfg.albumentations)
        else:
            logger.debug(
                "Albumentations configuration is None. No augmentation pipeline will be built."
            )
    except Exception as e:
        logger.error(
            f"Error building Albumentations pipeline: {e}. Augmentations will be skipped.",
            exc_info=True,
        )
        aug_pipeline = None  # Ensure it's None if building fails

    # For each frame, step each microtubule and draw it:
    for frame_idx in range(num_frames):
        try:
            frame, gt_data, mt_mask, seed_mask = render_frame(
                cfg=cfg,
                mts=mts,
                frame_idx=frame_idx,
                fixed_spot_generator=fixed_spot_generator,
                moving_spot_generator=moving_spot_generator,
                aug_pipeline=aug_pipeline,
                return_mt_mask=return_mt_mask,
                return_seed_mask=return_seed_mask,
            )
            logger.debug(f"Frame {frame_idx} yielded.")
            yield frame, gt_data, mt_mask, seed_mask, frame_idx
        except Exception as e:
            logger.error(
                f"Error rendering frame {frame_idx}: {e}. Skipping this frame.", exc_info=True
            )


def generate_video(
    cfg: SyntheticDataConfig,
    base_output_dir: str,
    num_png_frames: int = 0,
    is_for_expert_validation: bool = False,
) -> List[np.ndarray]:
    """
    Generates a full synthetic video, saves it, and optionally exports ground truth data.
    """
    logger.debug(
        f"Generating and writing {cfg.num_frames} frames for Series {cfg.id} into '{base_output_dir}'..."
    )

    output_manager_main = OutputManager(cfg, os.path.join(base_output_dir, "full"))

    if is_for_expert_validation:
        output_manager_validation_set = OutputManager(
            cfg, os.path.join(base_output_dir, "small")
        )
        validation_image_idx = random.randint(0, cfg.num_frames - 1)
    else:
        output_manager_validation_set = None
        validation_image_idx = -1  # No validation image for non-expert validation

    export_all_frames = (0 == num_png_frames) or (num_png_frames >= cfg.num_frames)
    export_current = True
    frame_idx_export = []

    if not export_all_frames:
        frame_idx_export = random.sample(range(cfg.num_frames), num_png_frames)

    frames: List[np.ndarray] = []
    frame_generator = generate_frames(
        cfg,
        cfg.num_frames,
        return_mt_mask=cfg.generate_mt_mask,
        return_seed_mask=cfg.generate_seed_mask,
    )

    for frame_rgb, frame_gt, mt_mask, mt_seed_mask, frame_idx in tqdm(
        frame_generator, total=cfg.num_frames, desc=f"Series {cfg.id} frames"
    ):

        frames.append(frame_rgb)
        if not export_all_frames:
            export_current = frame_idx in frame_idx_export

        output_manager_main.append(
            frame_idx, frame_rgb, mt_mask, mt_seed_mask, frame_gt, export_current
        )

        if is_for_expert_validation:
            export_png = frame_idx == validation_image_idx
            output_manager_validation_set.append(
                frame_idx, frame_rgb, mt_mask, mt_seed_mask, frame_gt, export_png
            )

    if output_manager_main:
        output_manager_main.close()

    if output_manager_validation_set:
        output_manager_validation_set.close()

    return frames

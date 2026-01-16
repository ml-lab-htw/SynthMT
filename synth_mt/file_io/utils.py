import json
import logging
import os
from typing import List, Tuple, Any

import cv2
import numpy as np
import tifffile
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Custom JSON encoder to handle non-serializable types like sets and numpy arrays.
class CustomJsonEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle common scientific data types.
    - Converts sets to lists.
    - Converts numpy integers and floats to standard Python types.
    - Converts numpy arrays to lists.
    """

    def default(self, obj: Any) -> Any:
        try:
            if isinstance(obj, set):
                logger.debug(f"JSONEncoder: Converting set to list.")
                return list(obj)
            if isinstance(obj, np.integer):
                logger.debug(f"JSONEncoder: Converting numpy integer {obj} to int.")
                return int(obj)
            if isinstance(obj, np.floating):
                logger.debug(f"JSONEncoder: Converting numpy float {obj} to float.")
                return float(obj)
            if isinstance(obj, np.ndarray):
                logger.debug(f"JSONEncoder: Converting numpy array of shape {obj.shape} to list.")
                return obj.tolist()
            # Let the base class default method raise the TypeError for other types.
            return super().default(obj)
        except Exception as e:
            logger.error(
                f"JSONEncoder: Error serializing object of type {type(obj)}: {e}", exc_info=True
            )
            raise  # Re-raise to prevent partial/corrupted JSON output


def fiji_auto_contrast(img, low_percentile=0.1, high_percentile=99.6):
    """
    Apply Fiji-style auto contrast: stretch histogram so that low/high percentiles map to 0/1.
    """
    p_low, p_high = np.percentile(img, [low_percentile, high_percentile])
    if p_high == p_low:
        return np.zeros_like(img, dtype=np.float32)
    img_stretched = np.clip(img, p_low, p_high)
    img_stretched = (img_stretched - p_low) / (p_high - p_low)
    return img_stretched.astype(np.float32)


def fiji_auto_contrast_brightness(img, low_percentile=0.4, high_percentile=99.6, target_mean=0.5):
    """
    Apply Fiji-style auto contrast and auto brightness:
    - Stretch histogram so that low/high percentiles map to 0/1.
    - Then scale so mean intensity is target_mean (default 0.5).
    """
    img_stretched = fiji_auto_contrast(img, low_percentile, high_percentile)
    mean_val = np.mean(img_stretched)
    if mean_val > 0:
        img_bright = np.clip(img_stretched / mean_val * target_mean, 0, 1)
    else:
        img_bright = img_stretched
    return img_bright.astype(np.float32)


def process_tiff_video(
    video_path: str,
    num_crops: int = 3,
    crop_size: Tuple[int, int] = (512, 512),
    auto_brightness: bool = True,
) -> List[List[np.ndarray]]:
    """
    Reads a TIFF video, performs repeated random cropping, applies contrast-stretching
    normalization as described in the paper, and returns processed videos.

    Args:
        video_path (str): The file path to the TIFF video.
        num_crops (int): The number of different random video crops to generate.
        crop_size (Tuple[int, int]): The (height, width) for the random crops.
        auto_brightness (bool): If True, applies auto brightness after auto contrast.

    Returns:
        List[List[np.ndarray]]: List of videos (list of frames). Each frame is a
                                NumPy array (crop_h, crop_w, 3) with float values in [0, 1].
    """
    # --- Step 1: Reading and Validation (same as before) ---
    try:
        data = tifffile.imread(video_path)
    except Exception as e:
        logger.error(f"Failed to read TIFF file {video_path}: {e}")
        return []

    if num_crops <= 1:
        logger.warning(f"num_crops is set to {num_crops}. No cropping will be performed.")
        return [[data]]

    logger.debug(f"Loaded TIFF stack of shape {data.shape}.")
    if data.ndim not in [3, 4]:
        logger.error(f"Unsupported TIFF dimension: {data.ndim}.")
        return []

    num_frames = data.shape[0]
    orig_h, orig_w = data.shape[-2:]
    crop_h, crop_w = crop_size
    if orig_h < crop_h or orig_w < crop_w:
        logger.error(f"Crop size {crop_size} is larger than frame size {(orig_h, orig_w)}.")
        return []

    # --- Step 2: Calculate global max for normalization ---
    global_max = np.max(data)
    if global_max == 0:
        logger.error("Global max is zero. Cannot normalize.")
        return []

    all_cropped_videos = []

    # --- Step 3: Cropping Loop ---
    for i in tqdm(range(num_crops), desc="Reading TIFF", unit="crop"):

        top = np.random.randint(0, orig_h - crop_h + 1)
        left = np.random.randint(0, orig_w - crop_w + 1)

        logger.info(f"Generating crop #{i + 1}/{num_crops} at (top={top}, left={left})")

        cropped_data = (
            data[:, :, top : top + crop_h, left : left + crop_w]
            if data.ndim == 4
            else data[:, top : top + crop_h, left : left + crop_w]
        )

        current_cropped_frames = []
        for t in range(num_frames):
            if data.ndim == 4:
                channels_to_process = [
                    cropped_data[t, c, :, :] for c in range(min(2, cropped_data.shape[1]))
                ]
            else:
                channels_to_process = [cropped_data[t, :, :]]

            if len(channels_to_process) == 2:
                # Identify main and red channel by intensity
                mean0 = np.mean(channels_to_process[0])
                mean1 = np.mean(channels_to_process[1])

                if mean0 > mean1:
                    main_chan = channels_to_process[0]
                    red_chan = channels_to_process[1]
                else:
                    main_chan = channels_to_process[1]
                    red_chan = channels_to_process[0]

                # Apply Fiji auto contrast
                main_chan_norm = fiji_auto_contrast(
                    main_chan, low_percentile=0.1, high_percentile=90.6
                )
                # main_chan_norm = cv2.equalizeHist((255 * main_chan).astype(np.uint8))
                red_chan_norm = fiji_auto_contrast(
                    red_chan, low_percentile=50, high_percentile=99.6
                )

                # Use OpenCV to convert grayscale to RGB
                main_chan_rgb = cv2.cvtColor(main_chan_norm, cv2.COLOR_GRAY2RGB)
                # Add red channel to R
                main_chan_rgb[..., 0] = np.clip(
                    np.maximum(main_chan_rgb[..., 0], red_chan_norm), 0, 1
                )
                rgb_frame = main_chan_rgb

                # --- Apply auto brightness correction to merged RGB ---
                if auto_brightness:
                    mean_val = np.mean(rgb_frame)
                    if mean_val > 0:
                        rgb_frame = np.clip(rgb_frame / mean_val * 0.8, 0, 1)

            elif len(channels_to_process) == 1:
                if auto_brightness:
                    gray_chan = fiji_auto_contrast_brightness(channels_to_process[0])
                else:
                    gray_chan = fiji_auto_contrast(channels_to_process[0])
                rgb_frame = np.stack([gray_chan, gray_chan, gray_chan], axis=-1)

            else:
                # Fallback for >2 channels: use first three, Fiji auto contrast (+ optional brightness)
                if auto_brightness:
                    norm_chans = [
                        fiji_auto_contrast_brightness(chan) for chan in channels_to_process[:3]
                    ]
                else:
                    norm_chans = [fiji_auto_contrast(chan) for chan in channels_to_process[:3]]
                while len(norm_chans) < 3:
                    norm_chans.append(np.zeros_like(norm_chans[0]))
                rgb_frame = np.stack(norm_chans, axis=-1)

            # plt.imshow(rgb_frame)
            # plt.axis('off')
            # plt.title(f"Crop {i+1}, Frame {t+1}")
            # plt.show()

            current_cropped_frames.append(rgb_frame)

        all_cropped_videos.append(current_cropped_frames)

    logger.info("Processing complete.")
    return all_cropped_videos


def extract_frames(
    video_path: str, num_crops: int = 1, crop_size=(512, 512), preprocess: bool = True
) -> Tuple[List[List[np.ndarray]], int]:

    logger.debug(f"Extracting frames from: {video_path}")

    if not os.path.isfile(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    fps: int = 10

    try:
        if video_path.lower().endswith((".avi", ".mp4", ".mov", ".mkv")):
            frames, fps = process_avi_video(fps, video_path)

            # rescale frames to 512 x 512
            frames = [
                cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA) for frame in frames
            ]
            frames = [frames]

        elif video_path.lower().endswith((".tif", ".tiff")):
            frames = process_tiff_video(
                video_path=video_path,
                num_crops=num_crops,
                crop_size=crop_size,
            )

            # plt.imshow(frames[0][0])
            # plt.axis('off')
            # plt.title(f"First frame from {os.path.basename(video_path)}")
            # plt.show()

        else:
            msg = f"Unsupported video format '{os.path.splitext(video_path)[1]}'. Only AVI, MP4, MOV, MKV and TIFF/TIF files are supported."
            logger.error(msg)
            raise ValueError(msg)

    except Exception as e:
        logger.error(
            f"An error occurred while extracting frames from {video_path}: {e}", exc_info=True
        )
        raise e

    logger.debug(f"Finished extracting {len(frames)} frames with FPS {fps}.")
    return frames, fps


def process_avi_video(fps, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file with OpenCV: {video_path}")
        raise IOError(f"Could not open video file: {video_path}")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    logger.debug(f"Opened video with OpenCV. FPS: {fps}.")
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame.dtype != "uint8":
            frame = (255 * frame).astype("uint8")

        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame)
        frame_count += 1
    cap.release()
    logger.debug(f"Successfully extracted {frame_count} frames from OpenCV video file.")
    return frames, fps

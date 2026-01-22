import json
import logging
import os
from typing import Optional, Any

import cv2
import imageio
import numpy as np
import tifffile
from skimage.color import label2rgb

from synth_mt.config.synthetic_data import SyntheticDataConfig
from synth_mt.file_io.utils import CustomJsonEncoder

logger = logging.getLogger(__name__)


def sort_instance_mask(seed_instance_mask: np.ndarray):
    # sort the instances by their min y-coordinate to have a consistent coloring
    # (this is a bit of a hacky way to do it, but it works for now)
    y_mins = []
    for i in range(seed_instance_mask.shape[0]):
        coords = np.column_stack(np.where(seed_instance_mask[i] > 0))
        if coords.size > 0:
            y_mins.append((i, coords[:, 0].min()))
        else:
            y_mins.append((i, np.inf))  # If no pixels, set to infinity
    y_mins.sort(key=lambda x: x[1])
    sorted_labels = [label for label, _ in y_mins]

    # instance ordering for consistent coloring
    seed_instance_mask = seed_instance_mask[sorted_labels]
    return seed_instance_mask


def merge_instance_mask(instance_mask: np.ndarray):
    # Merge instance masks into a single mask with unique labels
    if instance_mask is not None:
        # Sort instances for consistent coloring
        instance_mask = sort_instance_mask(instance_mask)

        mask = np.zeros(instance_mask.shape[1:], dtype=np.uint8)
        for label in range(instance_mask.shape[0]):
            mask[instance_mask[label] > 0] = label + 1

    else:
        mask = None

    return mask


def export_full_tiff_video_maks(masks: list[Any], path: Any | None):
    if path:
        try:
            # Stack along a new time axis and save
            full_mask_stack = np.stack(masks, axis=0)

            # Save as compressed TIFF (lossless)
            tifffile.imwrite(
                path,
                full_mask_stack,
                compression="zstd",
                metadata={"axes": "TCYX"},
                ome=True,
            )

            # # read the file back to verify
            # with tifffile.TiffFile(path) as tif:
            #     full_mask_stack = tif.asarray()
            #     all_identical = np.all(full_mask_stack == full_mask_stack)
            #     print(f"Verification: All frames identical: {all_identical}")

            logger.debug(
                f"MT instance mask video saved to {path} with shape {full_mask_stack.shape}"
            )
        except Exception as e:
            logger.error(
                f"Failed to write stacked MT mask TIFF to '{path}': {e}",
                exc_info=True,
            )


class OutputManager:
    """
    Manages the creation and writing process for all video/image sequence outputs.

    This class centralizes file path generation and writer object handling
    (TIFF, MP4, GIF, PNG) to simplify the main video generation loop. It allows
    for granular control over which output formats are generated.
    """

    def __init__(
        self,
        cfg: SyntheticDataConfig,
        base_output_dir: str,
        write_video_tiff: bool = True,
        write_video_masks_tiff: bool = True,
        write_video_mp4: bool = True,
        write_video_masks_mp4: bool = True,
        write_video_gif: bool = True,
        write_video_masks_gif: bool = True,
        write_image_pngs: bool = True,
        write_image_tiff: bool = True,
        write_image_masks_tiff: bool = True,
        write_image_masks_png: bool = True,
        write_config: bool = True,
        write_gt: bool = True,
    ):
        """
        Initializes all file paths and writer objects based on the config.

        Args:
            cfg: The synthetic data configuration object.
            base_output_dir: The base directory where all outputs will be saved.
            write_video_tiff: If True, saves the full video sequence as a multi-page TIFF file.
            write_video_masks_tiff: If True, saves the corresponding video masks as a multi-page TIFF file.
            write_video_mp4: If True, saves a preview of the video as an MP4 file.
            write_video_masks_mp4: If True, saves a preview of the video masks as an MP4 file.
            write_video_gif: If True, saves a preview of the video as a GIF file.
            write_video_masks_gif: If True, saves a preview of the video masks as a GIF file.
            write_image_pngs: If True, saves individual frames as PNG images.
            write_image_tiff: If True, saves individual frames as TIFF images.
            write_image_masks_tiff: If True, saves individual frame masks as multi-page TIFF files.
            write_image_masks_png: If True, saves individual frame masks as colored PNG images.
            write_config: If True, saves the generation configuration to a JSON file.
            write_gt: If True, saves the ground truth data to a JSON file.
        """
        logger.debug(
            f"Initializing OutputManager for series ID: {cfg.id}, output directory: '{base_output_dir}'"
        )
        self.cfg = cfg
        self.base_output_dir = base_output_dir

        # Store output generation flags
        self.write_video_tiff = write_video_tiff
        self.write_video_masks_tiff = write_video_masks_tiff
        self.write_video_mp4 = write_video_mp4
        self.write_video_masks_mp4 = write_video_masks_mp4
        self.write_video_gif = write_video_gif
        self.write_video_masks_gif = write_video_masks_gif
        self.write_image_pngs = write_image_pngs
        self.write_image_tiff = write_image_tiff
        self.write_image_masks_tiff = write_image_masks_tiff
        self.write_image_masks_png = write_image_masks_png
        self.write_config = write_config
        self.write_gt = write_gt
        self.all_gt = []  # Store ground truth data if needed
        self.all_mt_masks = []  # Store all MT instance masks for final TIFF writing
        self.all_seed_masks = []  # Store all seed instance masks for final TIFF writing

        try:
            os.makedirs(base_output_dir, exist_ok=True)
            logger.debug(f"Ensured base output directory exists: {base_output_dir}")
        except OSError as e:
            logger.critical(
                f"Failed to create base output directory '{base_output_dir}': {e}", exc_info=True
            )
            raise

        self._initialize_paths()
        self._initialize_writers()
        logger.debug("OutputManager initialization complete.")

    def _initialize_paths(self):
        """Defines all output paths based on configuration."""
        base_name = f"series_{self.cfg.id}"
        self.paths = {}

        # Sequence file paths
        if self.write_video_tiff:
            self.paths["video_tiff"] = os.path.join(
                self.base_output_dir, "videos", f"{base_name}_video.tif"
            )
        if self.write_video_masks_tiff and self.cfg.generate_mt_mask:
            self.paths["mt_masks_tiff"] = os.path.join(
                self.base_output_dir, "video_masks", f"{base_name}_masks.tif"
            )

        # Preview MP4
        if self.write_video_mp4:
            self.paths["video_mp4"] = os.path.join(
                self.base_output_dir, "previews", f"{base_name}_video_preview.mp4"
            )
        if self.write_video_masks_mp4 and self.cfg.generate_mt_mask:
            self.paths["mt_masks_mp4"] = os.path.join(
                self.base_output_dir, "previews", f"{base_name}_masks_preview.mp4"
            )

        # Preview GIFs
        if self.write_video_gif:
            self.paths["video_gif"] = os.path.join(
                self.base_output_dir, "previews", f"{base_name}_video_preview.gif"
            )
        if self.write_video_masks_gif and self.cfg.generate_mt_mask:
            self.paths["mt_masks_gif"] = os.path.join(
                self.base_output_dir, "previews", f"{base_name}_masks_preview.gif"
            )

        # Seed masks
        if self.cfg.generate_seed_mask:
            if self.write_video_masks_tiff:
                self.paths["seed_masks_tiff"] = os.path.join(
                    self.base_output_dir, "video_masks", f"{base_name}_seed_masks.tif"
                )
            if self.write_video_masks_mp4:
                self.paths["seed_masks_mp4"] = os.path.join(
                    self.base_output_dir, "previews", f"{base_name}_seed_masks_preview.mp4"
                )
            if self.write_video_masks_gif:
                self.paths["seed_masks_gif"] = os.path.join(
                    self.base_output_dir, "previews", f"{base_name}_seed_masks_preview.gif"
                )

        # Single-frame PNGs
        if self.write_image_pngs:
            self.paths["video_png_dir"] = os.path.join(self.base_output_dir, "images")
        if self.write_image_tiff:
            self.paths["image_tiff_dir"] = os.path.join(self.base_output_dir, "images_tiff")
        if self.write_image_masks_tiff and self.cfg.generate_mt_mask:
            self.paths["mt_mask_img_dir"] = os.path.join(self.base_output_dir, "image_masks")
        if self.write_image_masks_tiff and self.cfg.generate_seed_mask:
            self.paths["seed_mask_img_dir"] = os.path.join(self.base_output_dir, "image_masks_seed")
        if self.write_image_masks_png and self.cfg.generate_mt_mask:
            self.paths["mt_mask_img_colored_dir"] = os.path.join(
                self.base_output_dir, "image_masks_colored"
            )
        if self.write_image_masks_png and self.cfg.generate_seed_mask:
            self.paths["seed_mask_img_colored_dir"] = os.path.join(
                self.base_output_dir, "image_masks_colored_seed"
            )

        # Config
        if self.write_config:
            self.paths["config_file"] = os.path.join(
                self.base_output_dir, "configs", f"{base_name}_config.json"
            )
            try:
                self.cfg.save(self.paths["config_file"])
                logger.debug(f"Configuration saved to {self.paths['config_file']}")
            except Exception as e:
                logger.error(f"Failed to save configuration file: {e}", exc_info=True)

        # Ground truth file
        if self.write_gt:
            self.paths["ground_truth_file"] = os.path.join(
                self.base_output_dir, "gt", f"{base_name}_ground_truth.json"
            )

        logger.debug(f"Output paths defined: {self.paths}")

        # Ensure all necessary directories exist
        for key, path in self.paths.items():
            if key.endswith("_dir"):
                try:
                    os.makedirs(path, exist_ok=True)
                    logger.debug(f"Ensured directory exists: {path}")
                except OSError as e:
                    logger.error(f"Failed to create directory '{path}': {e}", exc_info=True)
                    raise
            elif not os.path.exists(os.path.dirname(path)):
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    logger.debug(f"Created parent directory for file: {os.path.dirname(path)}")
                except OSError as e:
                    logger.error(
                        f"Failed to create parent directory '{os.path.dirname(path)}': {e}",
                        exc_info=True,
                    )
                    raise

    def _initialize_writers(self):
        """Initializes all writer objects."""
        self.writers = {}
        img_h, img_w = self.cfg.img_size
        mp4_fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        def _create_writer(writer_type, path_key, **kwargs):
            try:
                if writer_type == "tiff":
                    return tifffile.TiffWriter(self.paths[path_key])
                elif writer_type == "mp4":
                    writer = cv2.VideoWriter(
                        self.paths[path_key], mp4_fourcc, self.cfg.fps, (img_w, img_h), **kwargs
                    )
                    if not writer.isOpened():
                        raise IOError(f"OpenCV VideoWriter failed to open for {path_key}.")
                    return writer
                elif writer_type == "gif":
                    return imageio.get_writer(
                        self.paths[path_key], fps=self.cfg.fps, mode="I", loop=0
                    )
            except Exception as e:
                logger.error(
                    f"Failed to initialize {writer_type} writer for {path_key} at {self.paths.get(path_key)}: {e}",
                    exc_info=True,
                )
            return None

        if self.write_video_tiff:
            self.writers["video_tiff"] = _create_writer("tiff", "video_tiff")
        if self.write_video_mp4:
            self.writers["video_mp4"] = _create_writer("mp4", "video_mp4")
        if self.write_video_gif:
            self.writers["video_gif"] = _create_writer("gif", "video_gif")

        if self.cfg.generate_mt_mask:
            if self.write_video_masks_tiff:
                self.writers["mt_mask_tiff"] = _create_writer("tiff", "mt_masks_tiff")
            if self.write_video_masks_mp4:
                self.writers["mt_mask_mp4"] = _create_writer("mp4", "mt_masks_mp4")
            if self.write_video_masks_gif:
                self.writers["mt_mask_gif"] = _create_writer("gif", "mt_masks_gif")

        if self.cfg.generate_seed_mask:
            if self.write_video_masks_tiff:
                self.writers["seed_mask_tiff"] = _create_writer("tiff", "seed_masks_tiff")
            if self.write_video_masks_mp4:
                self.writers["seed_mask_mp4"] = _create_writer("mp4", "seed_masks_mp4")
            if self.write_video_masks_gif:
                self.writers["seed_mask_gif"] = _create_writer("gif", "seed_masks_gif")

        logger.debug(f"Writers initialized: {list(self.writers.keys())}")

    def append(
        self,
        frame_index: int,
        frame_img: Optional[np.ndarray],
        mt_instance_mask: Optional[np.ndarray],
        seed_instance_mask: Optional[np.ndarray],
        gt_data: list[dict[str, Any]],
        export_current_frame: bool = False,
    ):
        """
        Appends a single frame and its corresponding data to all active writers.

        Args:
            frame_index: The index of the current frame.
            frame_img: The main image data for the current frame.
            mt_instance_mask: The microtubule mask for the current frame.
            seed_instance_mask: The seed mask for the current frame.
            gt_data: A list of ground truth dictionaries for the current frame.
            export_current_frame: If True, saves the current frame as a single image file.
        """

        seed_mask = merge_instance_mask(seed_instance_mask)
        mt_mask = merge_instance_mask(mt_instance_mask)

        #####################################
        #####           VIDEO           #####
        #####################################

        # TIFF RGB VIDEO
        if self.write_video_tiff and "video_tiff" in self.writers and frame_img is not None:
            self.writers["video_tiff"].write(frame_img)

        # Collect MT instance masks
        if (
            self.write_video_masks_tiff
            and "mt_mask_tiff" in self.writers
            and mt_instance_mask is not None
        ):
            self.all_mt_masks.append(mt_instance_mask)

        # Collect seed instance masks
        if (
            self.write_video_masks_tiff
            and "seed_mask_tiff" in self.writers
            and seed_instance_mask is not None
        ):
            self.all_seed_masks.append(seed_instance_mask)

        # MP4 previews
        if self.write_video_mp4 and "video_mp4" in self.writers and frame_img is not None:
            self.writers["video_mp4"].write(cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))

        # MP4 MASKS COLORED VIDEO - the whole MTs
        if self.write_video_masks_mp4 and "mt_mask_mp4" in self.writers and mt_mask is not None:
            colored_mask = (label2rgb(mt_mask, bg_label=0) * 255).astype(np.uint8)
            self.writers["mt_mask_mp4"].write(cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))

        # MP4 MASKS COLORED VIDEO - the seeds
        if self.write_video_masks_mp4 and "seed_mask_mp4" in self.writers and seed_mask is not None:
            colored_mask = (label2rgb(seed_mask, bg_label=0) * 255).astype(np.uint8)
            self.writers["seed_mask_mp4"].write(cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))

        # GIF previews
        if self.write_video_gif and "video_gif" in self.writers and frame_img is not None:
            self.writers["video_gif"].append_data(frame_img)

        # GIF MASKS COLORED VIDEO - the whole MTs
        if self.write_video_masks_gif and "mt_mask_gif" in self.writers and mt_mask is not None:
            self.writers["mt_mask_gif"].append_data(
                (label2rgb(mt_mask, bg_label=0) * 255).astype(np.uint8)
            )

        # GIF MASKS COLORED VIDEO - the seeds
        if self.write_video_masks_gif and "seed_mask_gif" in self.writers and seed_mask is not None:
            self.writers["seed_mask_gif"].append_data(
                (label2rgb(seed_mask, bg_label=0) * 255).astype(np.uint8)
            )

        ######################################
        #####        SINGLE FRAMES       #####
        ######################################
        if export_current_frame:
            base_name = f"series_{self.cfg.id}_frame_{frame_index:04d}"
            # Save individual frames as PNG
            if self.write_image_pngs and "video_png_dir" in self.paths and frame_img is not None:
                imageio.imwrite(
                    os.path.join(self.paths["video_png_dir"], f"{base_name}.png"),
                    frame_img,
                    compress_level=6,
                )

            # Save individual frames as TIFF
            if self.write_image_tiff and "image_tiff_dir" in self.paths and frame_img is not None:
                tifffile.imwrite(
                    os.path.join(self.paths["image_tiff_dir"], f"{base_name}.tif"),
                    frame_img,
                    compression="zstd",
                    metadata={},
                    ome=True,
                )

            # Save individual instance stack masks as TIFF
            if (
                self.write_image_masks_tiff
                and "mt_mask_img_dir" in self.paths
                and mt_instance_mask is not None
            ):
                tifffile.imwrite(
                    os.path.join(self.paths["mt_mask_img_dir"], f"{base_name}.tif"),
                    mt_instance_mask.astype(np.uint16),
                    compression="zstd",
                    metadata={},
                    ome=True,
                )

            # Save individual instance stack masks as TIFF - the seeds
            if (
                self.write_image_masks_tiff
                and "seed_mask_img_dir" in self.paths
                and seed_instance_mask is not None
            ):
                tifffile.imwrite(
                    os.path.join(self.paths["seed_mask_img_dir"], f"{base_name}.tif"),
                    seed_instance_mask.astype(np.uint16),
                    compression="zstd",
                    metadata={},
                    ome=True,
                )

            # Save individual colored masks as PNG
            if (
                self.write_image_masks_png
                and "mt_mask_img_colored_dir" in self.paths
                and mt_mask is not None
            ):
                imageio.imwrite(
                    os.path.join(self.paths["mt_mask_img_colored_dir"], f"{base_name}.png"),
                    (label2rgb(mt_mask, bg_label=0) * 255).astype(np.uint8),
                )

            # Save individual colored masks as PNG - the seeds
            if (
                self.write_image_masks_png
                and "seed_mask_img_colored_dir" in self.paths
                and seed_mask is not None
            ):
                imageio.imwrite(
                    os.path.join(self.paths["seed_mask_img_colored_dir"], f"{base_name}.png"),
                    (label2rgb(seed_mask, bg_label=0) * 255).astype(np.uint8),
                    compress_level=6,
                )

        ##################################
        #####      GROUND TRUTH      #####
        ##################################
        if self.write_gt:
            self.all_gt.append(gt_data)

    def close(self):
        """Closes all active writers and saves any remaining data."""
        logger.debug("Closing all writers.")

        # Write all collected MT masks at once
        if self.all_mt_masks:
            export_full_tiff_video_maks(self.all_mt_masks, self.paths.get("mt_masks_tiff"))

        # Write all collected seed masks at once
        if self.all_seed_masks:
            export_full_tiff_video_maks(self.all_seed_masks, self.paths.get("seed_masks_tiff"))

        for name, writer in self.writers.items():
            if writer is None:
                continue
            # Skip the tiff writers we've handled separately
            if name in ["mt_mask_tiff", "seed_mask_tiff"]:
                continue
            try:
                if "mp4" in name:
                    writer.release()
                else:
                    writer.close()
                logger.debug(f"Successfully closed {name} writer.")
            except Exception as e:
                logger.error(f"Error closing {name} writer: {e}", exc_info=True)

        # Save ground truth data
        if self.write_gt and self.all_gt:
            gt_path = self.paths.get("ground_truth_file")
            if gt_path:
                try:
                    with open(gt_path, "w") as f:
                        json.dump(self.all_gt, f, cls=CustomJsonEncoder, indent=4)
                    logger.debug(f"Ground truth data saved to {gt_path}")
                except Exception as e:
                    logger.error(
                        f"Failed to save ground truth data to '{gt_path}': {e}", exc_info=True
                    )

        logger.debug("OutputManager closed.")

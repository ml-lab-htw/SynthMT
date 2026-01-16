from cachetools import cached, LRUCache
import numpy as np

# transformers provides Sam3Processor/Sam3Model in newer versions; guard import to keep static checks quiet
try:
    from transformers import Sam3Processor, Sam3Model, pipeline
except Exception as _e:  # pragma: no cover
    Sam3Processor = None
    Sam3Model = None
    _import_err = _e

import torch
from PIL import Image, ImageOps
import logging
import time

from synth_mt.benchmark.models.base import BaseModel

# Module logger
logger = logging.getLogger(__name__)

# Cache for the model and processor
# We can use a simple LRUCache here. maxsize=2 is a reasonable default if you expect
# to work with models on both CPU and GPU.
model_cache = LRUCache(maxsize=2)


@cached(model_cache)
def _load_sam3_model_and_processor(aig: bool = False, device: str = "cuda"):
    """
    Loads and caches the SAM3 model and processor.
    The 'device' argument is used as the cache key.
    """
    model_name = "facebook/sam3"

    if aig:
        model = pipeline("mask-generation", model=model_name, device=device)
        return model, None
    else:
        if Sam3Model is None or Sam3Processor is None:
            logger.error(
                "Sam3Model/Sam3Processor not available. Ensure 'transformers' is installed with SAM3 support."
            )
            raise ImportError("transformers.Sam3Model or Sam3Processor not available")

        logger.debug("Loading Sam3Model.from_pretrained('facebook/sam3') on device %s", device)
        start = time.time()
        model = Sam3Model.from_pretrained(model_name).to(device)
        processor = Sam3Processor.from_pretrained(model_name)
        elapsed = time.time() - start
        logger.debug("Loaded SAM3 model and processor in %.3f seconds", elapsed)
        return model, processor


class SAM3(BaseModel):

    def __init__(
        self,
        model_name: str = "SAM3",
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self._points_per_batch = 64
        """
        points_per_batch : int
            Sets the number of points run simultaneously by the model.
            Higher numbers may be faster but use more GPU memory.
            Default: 64
        """
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self._min_mask_region_area = 0
        """
        min_mask_region_area : int
            If >0, postprocessing will be applied to remove disconnected regions
            and holes in masks with area smaller than min_mask_region_area.
            Requires opencv.
            Default: 0
        """
        self._model = None

    def load_model(self):
        """Load the model for text-prompted segmentation."""
        logger.debug("Called load_model()")
        if self._model is not None:
            logger.debug("Model already loaded; skipping load_model")
            return

        try:
            # This will use the cached version if available for the current device
            self._model, _ = _load_sam3_model_and_processor(True, self._device)
        except Exception as e:
            logger.exception("Failed to load SAM3 model/processor: %s", e)
            raise

    def predict(self, image: np.ndarray) -> np.ndarray:

        if image.ndim not in (2, 3):
            logger.error("Invalid image dimensionality: %s", image.shape)
            raise ValueError(f"image must be 2D or 3D (H,W[,C]); got {image.shape}")

        if self._model is None:
            logger.debug("Model not loaded in predict(); calling load_model()")
            self.load_model()

        assert self._model is not None

        # Ensure the image is uint8
        if image.dtype != np.uint8:
            try:
                img_min = float(image.min())
                img_max = float(image.max())
                logger.debug("Normalizing image with min=%s and max=%s", img_min, img_max)
                if img_max == img_min:
                    # Avoid division by zero; produce a zero image
                    logger.warning(
                        "Image has zero dynamic range (max == min == %s). Producing zeros before casting to uint8.",
                        img_max,
                    )
                    image = np.zeros_like(image, dtype=np.uint8)
                else:
                    image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            except Exception as e:
                logger.exception("Error normalizing image to uint8: %s", e)
                raise

        pil_image = Image.fromarray(image)
        logger.debug("Converted image to PIL with size=%s (width,height)", pil_image.size)

        results = self._model(
            pil_image,
            points_per_batch=self._points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self._min_mask_region_area,
        )

        # Extract masks from the pipeline output
        if not results or "masks" not in results or len(results["masks"]) == 0:
            h, w = image.shape[:2]
            return np.empty((0, h, w), dtype=np.uint16)

        # Get the masks, bounding boxes, and scores
        masks_output = results.get("masks", [])
        logger.debug("Post-processed results: found masks count=%d", len(masks_output))

        # Check if we got any masks
        if len(masks_output) == 0:
            h, w = pil_image.size[1], pil_image.size[0]  # PIL images are (width, height)
            logger.debug("No masks found; returning empty array with shape (0,%d,%d)", h, w)
            return np.empty((0, h, w), dtype=np.uint16)

        # Convert masks to instance masks
        masks = []
        for i, mask_data in enumerate(masks_output):
            try:
                # mask_data may be a torch tensor or numpy array
                mask = (
                    np.array(mask_data.cpu()) if hasattr(mask_data, "cpu") else np.array(mask_data)
                )
            except Exception as e:
                logger.exception("Failed to convert mask #%d to numpy array: %s", i, e)
                raise

            # Ensure mask is boolean
            if mask.dtype != bool:
                mask_bool = mask > 0
            else:
                mask_bool = mask

            logger.debug(
                "Mask %d shape=%s dtype=%s true_fraction=%.4f",
                i,
                mask_bool.shape,
                mask_bool.dtype,
                float(mask_bool.mean()),
            )
            masks.append(mask_bool)

        if len(masks) == 0:
            h, w = pil_image.size[1], pil_image.size[0]
            logger.debug("All masks filtered out; returning empty array with shape (0,%d,%d)", h, w)
            return np.empty((0, h, w), dtype=np.uint16)

        # Stack masks and create instance IDs
        masks = np.array(masks, dtype=bool)  # Shape: (N, H, W)
        num_masks = masks.shape[0]
        logger.debug("Stacked %d masks with shape %s", num_masks, masks.shape)
        instance_ids = np.arange(1, num_masks + 1, dtype=np.uint16)

        # Use broadcasting to multiply each boolean mask by its ID
        labeled_masks = masks * instance_ids[:, np.newaxis, np.newaxis]

        logger.debug(
            "Returning labeled_masks with dtype=%s shape=%s unique_ids_sample=%s",
            labeled_masks.dtype,
            labeled_masks.shape,
            np.unique(labeled_masks)[:10],
        )

        return labeled_masks.astype(np.uint16)


class SAM3Text(BaseModel):

    # Predefined text prompts for microtubule segmentation
    TEXT_PROMPTS = {
        0: "thin line",
        1: "elongated structure",
        2: "straight black line",
        3: "linear biological filament",
        4: "narrow elongated line",
        5: "thin red-highlighted line",
        6: "thin bright-on-dark microstructure",
        7: "thin structure in noisy background",
        8: "small linear object",
        9: "thin linear structure among noise",
    }

    def __init__(
        self,
        model_name: str = "SAM3Text",
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        text_prompt_option: int = 0,
        **kwargs,
    ):
        logger.debug(
            "Initializing SAM3 model wrapper: model_name=%s, threshold=%s, mask_threshold=%s, text_prompt_option=%s, kwargs=%s",
            model_name,
            threshold,
            mask_threshold,
            text_prompt_option,
            {k: v for k, v in kwargs.items()},
        )
        super().__init__(model_name, **kwargs)

        self.text_prompt_option = text_prompt_option
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self._model = None
        self._processor = None

    def load_model(self):
        """Load the model for text-prompted segmentation."""
        logger.debug("Called load_model()")
        if self._model is not None:
            logger.debug("Model already loaded; skipping load_model")
            return

        try:
            # This will use the cached version if available for the current device
            self._model, self._processor = _load_sam3_model_and_processor(False, self._device)
        except Exception as e:
            logger.exception("Failed to load SAM3 model/processor: %s", e)
            raise

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Text-prompted segmentation using SAM3 image model."""
        logger.debug(
            "predict() called with image shape=%s, dtype=%s",
            getattr(image, "shape", None),
            getattr(image, "dtype", None),
        )
        if image.ndim not in (2, 3):
            logger.error("Invalid image dimensionality: %s", image.shape)
            raise ValueError(f"image must be 2D or 3D (H,W[,C]); got {image.shape}")

        if self._model is None:
            logger.debug("Model not loaded in predict(); calling load_model()")
            self.load_model()

        assert self._model is not None
        assert self._processor is not None

        # Ensure the image is uint8
        if image.dtype != np.uint8:
            try:
                img_min = float(image.min())
                img_max = float(image.max())
                logger.debug("Normalizing image with min=%s and max=%s", img_min, img_max)
                if img_max == img_min:
                    # Avoid division by zero; produce a zero image
                    logger.warning(
                        "Image has zero dynamic range (max == min == %s). Producing zeros before casting to uint8.",
                        img_max,
                    )
                    image = np.zeros_like(image, dtype=np.uint8)
                else:
                    image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            except Exception as e:
                logger.exception("Error normalizing image to uint8: %s", e)
                raise

        pil_image = Image.fromarray(image)
        logger.debug("Converted image to PIL with size=%s (width,height)", pil_image.size)

        # Get the text prompt based on the selected option
        text_prompt = self.TEXT_PROMPTS.get(self.text_prompt_option, self.TEXT_PROMPTS[0])
        logger.debug("Using text prompt option %s: '%s'", self.text_prompt_option, text_prompt)

        try:
            inputs = self._processor(images=pil_image, text=text_prompt, return_tensors="pt").to(
                self._device
            )
            logger.debug("Processor returned inputs keys=%s", list(inputs.keys()))
        except Exception as e:
            logger.exception("Processor failed to prepare inputs: %s", e)
            raise

        try:
            with torch.no_grad():
                t0 = time.time()
                outputs = self._model(**inputs)
                t_infer = time.time() - t0
                logger.debug("Model forward pass completed in %.3f seconds", t_infer)
        except Exception as e:
            logger.exception("Model inference failed: %s", e)
            raise

        # Post-process results
        try:
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
        except Exception as e:
            logger.exception("Post-processing failed: %s", e)
            raise

        # Get the masks, bounding boxes, and scores
        masks_output = results.get("masks", [])
        logger.debug("Post-processed results: found masks count=%d", len(masks_output))

        # Check if we got any masks
        if len(masks_output) == 0:
            h, w = pil_image.size[1], pil_image.size[0]  # PIL images are (width, height)
            logger.debug("No masks found; returning empty array with shape (0,%d,%d)", h, w)
            return np.empty((0, h, w), dtype=np.uint16)

        # Convert masks to instance masks
        masks = []
        for i, mask_data in enumerate(masks_output):
            try:
                # mask_data may be a torch tensor or numpy array
                mask = (
                    np.array(mask_data.cpu()) if hasattr(mask_data, "cpu") else np.array(mask_data)
                )
            except Exception as e:
                logger.exception("Failed to convert mask #%d to numpy array: %s", i, e)
                raise

            # Ensure mask is boolean
            if mask.dtype != bool:
                mask_bool = mask > 0
            else:
                mask_bool = mask

            logger.debug(
                "Mask %d shape=%s dtype=%s true_fraction=%.4f",
                i,
                mask_bool.shape,
                mask_bool.dtype,
                float(mask_bool.mean()),
            )
            masks.append(mask_bool)

        if len(masks) == 0:
            h, w = pil_image.size[1], pil_image.size[0]
            logger.debug("All masks filtered out; returning empty array with shape (0,%d,%d)", h, w)
            return np.empty((0, h, w), dtype=np.uint16)

        # Stack masks and create instance IDs
        masks = np.array(masks, dtype=bool)  # Shape: (N, H, W)
        num_masks = masks.shape[0]
        logger.debug("Stacked %d masks with shape %s", num_masks, masks.shape)
        instance_ids = np.arange(1, num_masks + 1, dtype=np.uint16)

        # Use broadcasting to multiply each boolean mask by its ID
        labeled_masks = masks * instance_ids[:, np.newaxis, np.newaxis]

        logger.debug(
            "Returning labeled_masks with dtype=%s shape=%s unique_ids_sample=%s",
            labeled_masks.dtype,
            labeled_masks.shape,
            np.unique(labeled_masks)[:10],
        )

        return labeled_masks.astype(np.uint16)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # configure basic logging for the demo run
    logging.basicConfig(level=logging.INFO)

    # Load an image
    img_folder = "data/SynMT/synthetic/full/images/"
    # take the first image in the folder
    import os

    # img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    # img_path = os.path.join(img_folder, img_files[0])
    img_path = os.path.join(
        img_folder,
        "series_5_9uMporcTub_300nMPfSPM1_1800nMPfTrxL1_repeat2__crop_6_rank_6_frame_0037.png",
    )
    image_rgb = Image.open(img_path)
    image = ImageOps.grayscale(image_rgb)

    def show_masks_on_image(image, masks, title="Segmentation Masks"):
        plt.figure(figsize=(10, 10))

        # Create a color map for the masks
        num_masks = len(masks)
        colors = plt.get_cmap("jet", num_masks + 1)

        plt.imshow(image)

        for i, mask in enumerate(masks):
            colored_mask = np.zeros((*mask.shape, 4))  # RGBA
            colored_mask[..., :3] = colors(i)[:3]  # RGB
            colored_mask[..., 3] = 0.99 * mask.astype(float)  # Alpha channel

            plt.imshow(colored_mask)

        plt.axis("off")
        plt.title(title)
        # export the figure to a file
        plt.savefig("sam3_segmentation_result.png", bbox_inches="tight", pad_inches=0)
        plt.show()

    image_np = np.array(image, dtype=np.uint8)

    preprocess_params = {
        "grayscale": True,
        "clip_to_percentiles": True,
        "rescale_using_percentiles": True,
        "invert": False,
        "histogram_normalization": False,
        "sharpen_radius": 0.38717865927737405,
        "smooth_radius": 3.410555347410367,
        "percentile_min": 3.506993161284297,
        "percentile_max": 99.46537030219284,
    }

    import synth_mt.utils.preprocessing as pre

    processed_image = pre.process_image(image_np, **preprocess_params)

    sam3text = SAM3Text(
        threshold=0.4018846604226014,
        mask_threshold=0.44189937830348586,
    )
    sam3text.load_model()
    masks = sam3text.predict(processed_image)

    # import synth_mt.utils.postprocessing as post
    # masks = post.filter_instance_masks(
    #     masks, 0, 10000, 0, 10000, -1
    # )

    show_masks_on_image(
        image_rgb, masks, title=f"SAM3 - Text-Prompted Segmentation ({len(masks)} masks)"
    )

    # sam3 = SAM3()
    # sam3.load_model()
    # masks = sam3.predict(processed_image)
    # show_masks_on_image(image_rgb, masks, title=f"SAM3 - Automatic Mask Generation ({len(masks)} masks)")

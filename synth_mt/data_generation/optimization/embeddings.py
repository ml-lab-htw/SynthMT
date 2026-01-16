import logging
import os
from glob import glob
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from cellpose import transforms
from cellpose.models import CellposeModel, normalize_default
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import (
    AutoModel,
    CLIPModel,
    CLIPImageProcessor,
    PreTrainedModel,
    FeatureExtractionMixin,
    AutoImageProcessor,
)

from synth_mt.config.synthetic_data import SyntheticDataConfig
from synth_mt.config.tuning import TuningConfig
from synth_mt.data_generation.video import generate_frames
from synth_mt.file_io import extract_frames

logger = logging.getLogger(__name__)


class ImageEmbeddingExtractor:
    """
    A class to extract and optionally reduce image embeddings using transformer models and PCA.

    This class encapsulates model loading, preprocessing, embedding computation, and
    dimensionality reduction. The PCA model is fitted on a set of reference embeddings
    and can then be applied to any subsequent embeddings.

    Attributes:
        config (TuningConfig): The configuration used to initialize the extractor.
        device (torch.device): The device the model is running on ('cuda', 'mps', or 'cpu').
        model (PreTrainedModel): The loaded transformer model.
        processor (FeatureExtractionMixin): The processor for preparing images for the model.
        pca_model (Optional[PCA]): The PCA model fitted on reference embeddings. None if PCA is not used.
    """

    def __init__(self, tuning_cfg: TuningConfig):
        """
        Initializes the ImageEmbeddingExtractor.

        Args:
            tuning_cfg (TuningConfig): Configuration object containing model name,
                                       cache directory, and optional PCA components.
        """
        logger.debug("Initializing ImageEmbeddingExtractor...")
        self.config = tuning_cfg

        # self.hf_login()

        self.device = self._get_best_device()
        logger.debug(f"Using device: {self.device}")

        try:
            self.model, self.processor = self._load_model_and_processor()

            logger.info(
                f"Model '{self.config.model_name}' loaded and set to evaluation mode on {self.device}."
            )
        except Exception as e:
            logger.critical(
                f"Failed to load or initialize model '{self.config.model_name}': {e}", exc_info=True
            )
            raise  # Re-raise to indicate a critical setup failure

        # Initialize PCA model, it will be fitted later
        self.pca_model: Optional[PCA] = None
        logger.debug("ImageEmbeddingExtractor initialized successfully.")

    def hf_login(self):
        """
        Attempts to log into the Hugging Face Hub using a token.

        This is a non-blocking login attempt. If a token is found, it will try
        to log in. If login fails or no token is found, it will log a warning
        and continue execution, assuming the model might be public.
        """
        logger.info("Attempting to log into Hugging Face Hub...")
        token = os.getenv("HUGGING_FACE_HUB_TOKEN")

        # Check if the token from env is valid, otherwise try to load from .env
        if not token or not token.startswith("hf_"):
            if token:
                logger.warning(
                    f"Environment variable HUGGING_FACE_HUB_TOKEN has a value that does not look like a token. "
                    f"Will attempt to load from .env file."
                )
            load_dotenv()
            token = os.getenv("HUGGING_FACE_HUB_TOKEN")

        if token and token.startswith("hf_"):
            try:
                login(token)
                logger.info("Successfully logged into Hugging Face Hub.")
            except Exception as e:
                logger.warning(
                    f"Hugging Face login failed: {e}. Continuing without authentication."
                )
        else:
            logger.warning("No valid Hugging Face token found. Continuing without authentication.")

    def _get_best_device(self) -> torch.device:
        """Selects the best available device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            logger.debug("CUDA is available. Using GPU.")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            logger.debug("MPS (Apple Silicon) is available. Using MPS.")
            return torch.device("mps")
        logger.debug("No GPU (CUDA/MPS) found. Falling back to CPU.")
        return torch.device("cpu")

    def _load_model_and_processor(self) -> Tuple[PreTrainedModel, FeatureExtractionMixin]:
        """Loads the model and processor from Hugging Face based on the config."""
        model_name = self.config.model_name
        cache_dir = self.config.hf_cache_dir

        logger.debug(f"Loading model and processor for '{model_name}' from Hugging Face...")
        try:

            if "clip" in model_name.lower():
                model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
                processor = CLIPImageProcessor.from_pretrained(
                    model_name, cache_dir=cache_dir, use_fast=True
                )
                logger.debug(f"Loaded CLIP model and processor: {model_name}.")

            elif "cellpose" in model_name.lower():
                model = CellposeModel(
                    model_type="cellpose_sam", gpu=torch.cuda.is_available(), device=self.device
                )
                logger.debug(f"Loaded Cellpose model: {model_name}.")

                encoder = model.net.encoder
                encoder.eval()

                return model, encoder
            else:
                model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
                processor = AutoImageProcessor.from_pretrained(
                    model_name, cache_dir=cache_dir, use_fast=True
                )
                logger.debug(f"Loaded AutoModel and AutoImageProcessor: {model_name}.")

            model.to(self.device)
            model.eval()  # Set model to evaluation mode

            return model, processor
        except Exception as e:
            logger.error(f"Failed to load model or processor '{model_name}': {e}", exc_info=True)
            raise

    def _compute_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Computes the raw (pre-PCA) embedding for a single RGB image.

        Args:
            image (np.ndarray): An image in RGB format (H, W, C).

        Returns:
            np.ndarray: A 1D numpy array representing the raw image embedding.
        """
        logger.debug(f"Computing embedding for image of shape {image.shape} (RGB).")

        try:
            if "cellpose" in self.config.model_name.lower():
                x = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
                x = transforms.convert_image(x, channel_axis=None, z_axis=None)
                x = x[np.newaxis, ...]

                normalize_params = normalize_default
                normalize_params["normalize"] = True
                normalize_params["invert"] = False

                x = transforms.normalize_img(x, **normalize_params)
                X = torch.from_numpy(x.transpose(0, 3, 1, 2)).to(
                    self.model.device, dtype=self.model.net.dtype
                )

                with torch.no_grad():
                    out = self.processor(X)

                return out.detach().squeeze().cpu().to(torch.float32).numpy().flatten()

            else:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    # Unified logic for models that support hidden states (like DINO, CLIP)
                    if isinstance(self.model, CLIPModel):
                        # For CLIP, we call the vision_model specifically to get hidden states
                        outputs = self.model.vision_model(**inputs, output_hidden_states=True)
                    else:
                        # For other models like DINOv2, the main model forward pass is used
                        outputs = self.model(**inputs, output_hidden_states=True)

                    if not hasattr(outputs, "hidden_states") or not outputs.hidden_states:
                        msg = "Model output does not contain 'hidden_states'. Cannot select a specific layer."
                        logger.error(msg)
                        raise ValueError(msg)

                    # hidden_states is a tuple of (batch_size, sequence_length, hidden_size)
                    # The first element is the input embeddings, subsequent are layer outputs.
                    hidden_states = outputs.hidden_states
                    layer_idx = self.config.embedding_layer

                    if layer_idx >= len(hidden_states):
                        logger.warning(
                            f"Configured embedding_layer {layer_idx} is out of bounds for model with "
                            f"{len(hidden_states)} layers. Falling back to the last layer ({len(hidden_states) - 1})."
                        )
                        layer_idx = -1  # Use the last layer

                    logger.debug(f"Extracting embedding from layer {layer_idx}.")
                    # Select the specified layer's hidden state.
                    embedding_tensor = hidden_states[layer_idx]
                    # Take the mean over the sequence dimension (patch tokens) to get a single vector.
                    embedding = embedding_tensor.mean(dim=1)

                return embedding.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(
                f"Error computing embedding for image of shape {image.shape}: {e}", exc_info=True
            )
            raise

    def extract_from_references(self) -> np.ndarray:
        """Loads reference video or images directory, computes embeddings, and fits PCA model."""
        logger.info(
            f"Starting reference embedding extraction (PCA components: {self.config.pca_components})."
        )
        embeddings = []

        # Determine if we're using video or image folder
        if self.config.reference_images_dir and os.path.isdir(self.config.reference_images_dir):
            # Process image folder
            logger.info(
                f"Processing reference images from directory: {self.config.reference_images_dir}"
            )
            image_files = sorted(
                glob(os.path.join(self.config.reference_images_dir, "*.png"))
                + glob(os.path.join(self.config.reference_images_dir, "*.jpg"))
                + glob(os.path.join(self.config.reference_images_dir, "*.tif"))
            )

            if not image_files:
                msg = f"No image files found in directory: {self.config.reference_images_dir}"
                logger.error(msg)
                raise ValueError(msg)

            # Limit number of frames if needed
            image_files = image_files[: self.config.num_compare_frames]

            for img_path in tqdm(image_files, desc="Processing reference images"):
                try:
                    frame = cv2.imread(img_path)
                    if frame is None:
                        logger.warning(f"Could not read image: {img_path}")
                        continue
                    frame_rgb = self.convert_frame(frame)
                    emb = self._compute_embedding(frame_rgb)
                    embeddings.append(emb)
                    logger.debug(
                        f"Processed image {os.path.basename(img_path)}. Embedding shape: {emb.shape}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing image {os.path.basename(img_path)}: {e}", exc_info=True
                    )
        else:
            # Process video file
            video_path = self.config.reference_video_path
            if not os.path.isfile(video_path):
                msg = f"Reference video file not found: {video_path}"
                logger.error(msg)
                raise ValueError(msg)

            logger.info(f"Processing reference video: {video_path}")
            try:
                frames, _ = extract_frames(video_path)
                if not frames:
                    msg = f"No frames extracted from reference video: {video_path}"
                    logger.error(msg)
                    raise ValueError(msg)

                # Use only a subset of frames if configured
                frames_to_process = frames[: self.config.num_compare_frames]
                logger.debug(f"Using {len(frames_to_process)} frames from reference video.")

                for frame in tqdm(frames_to_process, desc="Processing reference frames"):
                    frame_rgb = self.convert_frame(frame)
                    emb = self._compute_embedding(frame_rgb)
                    embeddings.append(emb)
            except Exception as e:
                logger.error(f"Error processing reference video {video_path}: {e}", exc_info=True)
                raise

        if not embeddings:
            msg = "No embeddings were extracted from reference data."
            logger.error(msg)
            raise ValueError(msg)

        raw_embeddings = np.stack(embeddings)
        logger.debug(
            f"Extracted {raw_embeddings.shape[0]} reference embeddings of dimension {raw_embeddings.shape[1]}."
        )

        # Fit and apply PCA if configured
        if self.config.pca_components is not None and self.config.pca_components > 0:
            n_components = min(
                self.config.pca_components, raw_embeddings.shape[0], raw_embeddings.shape[1]
            )
            logger.info(f"Fitting PCA with {n_components} components on reference embeddings.")
            self.pca_model = PCA(n_components=n_components)
            reduced_embeddings = self.pca_model.fit_transform(raw_embeddings)
            logger.debug(
                f"PCA reduced dimensions from {raw_embeddings.shape[1]} to {reduced_embeddings.shape[1]}."
            )
            return reduced_embeddings
        else:
            logger.debug("No PCA requested, returning original embeddings.")
            return raw_embeddings

    def _apply_pca_if_available(self, embeddings: np.ndarray) -> np.ndarray:
        """Applies the fitted PCA model to a new set of embeddings."""
        if self.pca_model:
            logger.debug(
                f"Applying fitted PCA to new embeddings (input shape: {embeddings.shape})."
            )
            try:
                transformed_embeddings = self.pca_model.transform(embeddings)
                logger.debug(
                    f"Embeddings transformed by PCA. New shape: {transformed_embeddings.shape}."
                )
                return transformed_embeddings
            except Exception as e:
                logger.error(f"Error applying PCA transformation: {e}", exc_info=True)
                # Depending on severity, you might return raw embeddings or re-raise
                raise
        else:
            logger.debug("PCA model not fitted or configured. Returning raw embeddings.")
            return embeddings

    def extract_from_synthetic_config(
        self, synthetic_cfg: SyntheticDataConfig, num_compare_frames: int = 1
    ) -> np.ndarray:
        logger.debug(
            f"Extracting embeddings from synthetic config (ID: {synthetic_cfg.id}). Comparing {num_compare_frames} frames."
        )
        raw_embeddings = []
        frame_generator = generate_frames(synthetic_cfg, num_compare_frames)

        for frame, *_ in frame_generator:

            # frame is already in RGB format, no need to convert
            # rgb_frame = self.convert_frame(frame)
            emb = self._compute_embedding(frame)
            raw_embeddings.append(emb)
            logger.debug(
                f"Generated and processed frame for synthetic config {synthetic_cfg.id}. Embedding shape: {emb.shape}"
            )

        if not raw_embeddings:
            logger.warning(f"No embeddings generated for synthetic config ID {synthetic_cfg.id}.")
            return np.array([])  # Return empty array if no embeddings could be generated

        raw_embeddings = np.stack(raw_embeddings)
        logger.debug(
            f"Extracted {raw_embeddings.shape[0]} raw embeddings from synthetic config {synthetic_cfg.id}."
        )
        return self._apply_pca_if_available(raw_embeddings)

    def extract_from_frames(
        self, frames: List[np.ndarray], num_compare_frames: int = 1
    ) -> np.ndarray:
        logger.debug(
            f"Extracting embeddings from a provided list of {len(frames)} frames. Comparing {num_compare_frames} frames."
        )
        raw_embeddings = []

        # Limit frames to num_compare_frames if necessary
        frames_to_process = frames[:num_compare_frames]
        if len(frames_to_process) < len(frames):
            logger.debug(
                f"Limiting processing to first {len(frames_to_process)} frames from the list."
            )

        try:
            for frame_idx, frame in enumerate(
                tqdm(frames_to_process, desc="Generating embeddings from frames")
            ):
                rgb_frame = self.convert_frame(frame)
                emb = self._compute_embedding(rgb_frame)
                raw_embeddings.append(emb)
                logger.debug(
                    f"Processed frame {frame_idx + 1} from list. Embedding shape: {emb.shape}"
                )
        except Exception as e:
            logger.error(
                f"Error during embedding extraction from provided frame list: {e}", exc_info=True
            )
            raise

        if not raw_embeddings:
            logger.warning("No embeddings generated from the provided frame list.")
            return np.array([])

        raw_embeddings = np.stack(raw_embeddings)
        logger.debug(f"Extracted {raw_embeddings.shape[0]} raw embeddings from provided frames.")
        return self._apply_pca_if_available(raw_embeddings)

    def convert_frame(self, frame: np.ndarray) -> np.ndarray:
        """Converts a frame to RGB format if necessary."""
        if frame.ndim == 2:
            logger.debug("Converting grayscale frame to RGB.")
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            # Assuming BGR input for cv2 operations, convert to RGB for models
            # Many image feature extractors expect RGB, while OpenCV defaults to BGR
            # This is a common point of error, so being explicit.
            if (
                frame.dtype == np.uint8
            ):  # Only convert if it's an 8-bit image for typical BGR -> RGB
                logger.debug("Converting BGR frame to RGB.")
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                logger.debug(
                    "Frame is already 3-channel. Assuming it's RGB or compatible. Skipping color conversion."
                )
                return frame
        else:
            logger.warning(
                f"Unexpected frame dimension/channels: {frame.shape}. Returning as is, may cause issues."
            )
            return frame

    @staticmethod
    def flatten_spatial_dims(embeddings: np.ndarray) -> np.ndarray:
        """Utility to ensure embeddings are 2D by flattening spatial dimensions."""
        if embeddings.ndim == 3:
            N, H, W = embeddings.shape
            logger.debug(f"Flattening spatial dimensions from {embeddings.shape} to {(N, H * W)}.")
            return embeddings.reshape(N, H * W)
        logger.debug(f"Embeddings are already 2D ({embeddings.shape}). No flattening needed.")
        return embeddings

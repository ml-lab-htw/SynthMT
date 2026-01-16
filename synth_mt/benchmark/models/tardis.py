import logging
import os
import time
import zipfile

import cv2
import numpy as np
import tifffile
from numpy import ndarray
from synth_mt.benchmark.models.anchor_point_model import AnchorPointModel

logger = logging.getLogger()
try:
    from tardis_em.utils.aws import get_all_version_aws
    from tardis_em.utils.predictor import GeneralPredictor
except ImportError as e:
    logger.warning(f"Failed to import tardis_em: {e}")


class TARDIS(AnchorPointModel):
    """
    Apart from the other models, TARDIS, as seen in its code `tardis_em/scripts/predict_mt_tirf.py`,
    needs all of the data given to the `GeneralPredictor` class at once. So the `load` and
    `predict` logic as in the other files won't quite work here.

    Also, all of the files have to stored as 2D grayscale TIFF images.

    Note: Sadly, there is no way to make TARDIS nicely use the `self._save_dir` parameter, as it
    offers no way to dynamically change its models cache dir (but will always use ~/.tardis_em).
    When using it on a cluster, we circumvent this by using a symbolik link (see `run_benchmark.yaml`).
    """

    def load_model(self):
        pass

    def __init__(
        self,
        model_name: str = "tardis_mt_tirf",
        cnn_threshold: float = 0.1,
        dist_threshold: float = 0.5,
        **kwargs,
    ):
        """
        TARDIS batch prediction model for microtubule segmentation.

        This class and its documentation are heavily inspired by tardis_em/scripts/predict_mt_tirf.py.

        Parameters
        ----------
        work_dir : Optional[str]
            Directory with images for prediction with CNN model. Used for temporary files and batch prediction images.
        cnn_threshold : float
            Threshold used for CNN prediction.
        dist_threshold : float
            Threshold used for instance prediction.

        """
        super().__init__(model_name, **kwargs)

        self.grayscale = True
        self._im_per_sec = -1
        self._rotate = True
        """
        rotate : bool
            If True, during CNN prediction image is rotated 4x by 90 degrees. This will increase prediction time 4x.
            However, may lead to cleaner output.
        """
        self.cnn_threshold = cnn_threshold
        self.dist_threshold = dist_threshold
        self._points_in_patch = 900
        # points_in_patch : int
        #     Size of the cropped point cloud, given as a max. number of points per crop. This will break the generated
        #     point cloud into smaller patches with overlap.

        self._network = "fnet_attn"
        self._subtype = "32"  # Could be parameterized if needed
        self._dataset = "microtubules_tirf"  # Could be parameterized if needed

        # Rename device for TARDIS...
        self._device = "gpu" if self._device == "cuda" else self._device

    def _get_latest_version(self) -> str | None:
        versions = get_all_version_aws(self._network, self._subtype, self._dataset)
        if not versions:
            return None
        vs = []
        for v in versions:
            try:
                num = int(v.split("_")[1])
                vs.append((num, v))
            except Exception:
                pass
        if vs:
            _, maxv = max(vs, key=lambda x: x[0])
            return maxv
        return None

    def _is_valid_checkpoint(self, path: str) -> bool:
        """
        Basic check: is file a zip archive (as PyTorch expects)?
        And is it not tiny (size threshold)?
        """
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        if size < 1_000:  # maybe too small to be legitimate
            print(f"Checkpoint file too small ({size} bytes): {path}")
            return False
        else:
            print(f"Checkpoint file size: {size} bytes")
        # Check whether it's a zip
        try:
            return zipfile.is_zipfile(path)
        except Exception:
            return False

    @staticmethod
    def _hash_image(image: np.ndarray) -> int:
        # Use a simple hash of the image bytes for lookup
        # maybe not needed, to be figured out
        return hash(image.tobytes())


    def predict_batch(self, images: list[np.ndarray]) -> list[list[ndarray]]:
        time_predict = self.load_and_predict_all(images)
        predictions = [self.predict(im) for im in images]
        logger.debug(
            f"TARDIS batch prediction: {len(images)} images in {time_predict:.2f} seconds "
            f"({self._im_per_sec:.2f} images/second)"
        )
        return predictions

    def load_and_predict_all(self, images: list[np.ndarray]) -> float:
        """
        Run the predictor on all images at once and store results for retrieval.

        Parameters
        ----------
        images : list of np.ndarray
            All images to be predicted in this batch.

        Returns
        -------
        float
            Total processing time in seconds for running the predictor on all images.
        """
        if self._work_dir is None:
            raise ValueError("work_dir must be specified for TARDIS batch prediction.")

        self._tiff_paths = {}

        # Save all images as grayscale 2D TIFFs in work_dir
        for idx, img in enumerate(images):
            img_hash = self._hash_image(img)
            # Convert to grayscale if needed
            if img.ndim == 3:
                if img.shape[2] == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    raise ValueError(f"Image at idx {idx} has unexpected shape: {img.shape}")
            else:
                img_gray = img
            tiff_path = os.path.join(self._work_dir, f"img_{idx}.tif")
            tifffile.imwrite(tiff_path, img_gray)

            self._tiff_paths[img_hash] = tiff_path

        # Setup predictor to process all images in work_dir
        self._predictor = GeneralPredictor(
            predict="Microtubule_tirf",
            dir_s=self._work_dir,
            binary_mask=False,
            correct_px=None,
            normalize_px=1.0,
            convolution_nn=self._network,
            checkpoint=[None, None],
            model_version=None,
            output_format="tif_csv",  # NOTE: Using `return_return` here could easily lead to memory
            # overflow, that's why we favor saving locally in temp working directory
            patch_size=128,
            cnn_threshold=str(self.cnn_threshold),
            dist_threshold=self.dist_threshold,
            points_in_patch=self._points_in_patch,
            predict_with_rotation=self._rotate,
            device_s=self._device,
            filter_by_length=None,
            connect_splines=0,
            connect_cylinder=None,
            instances=True,
            tardis_logo=False,
            debug=False,
            continue_b=False,
        )

        # Run prediction on all images
        start_time = time.time()
        self._predictor()
        return time.time() - start_time

    def _run_backend(self, image: np.ndarray) -> str | None:
        """Return the CSV file produced by TARDIS for this image."""
        if not hasattr(self, "_tiff_paths"):
            raise RuntimeError("Must call load_and_predict_all before predict().")

        img_hash = self._hash_image(image)
        if img_hash not in self._tiff_paths:
            raise KeyError("Prediction for this image not found in loaded batch.")

        grayscale_image_path = self._tiff_paths[img_hash]
        instance_prediction_result_path = os.path.join(
            os.path.dirname(grayscale_image_path),
            "Predictions",
            f"{os.path.splitext(os.path.basename(grayscale_image_path))[0]}_instances.csv",
        )

        if not os.path.isfile(instance_prediction_result_path):
            logger.debug(
                f"FileNotFoundError: Expected CSV not found at: {instance_prediction_result_path}. "
                "Probably because no instances found; returning None."
            )
            instance_prediction_result_path = None

        return instance_prediction_result_path

    def generate_batch_predictions(self, images: list):
        """
        Generates batch predictions for AnchorPointModel. AnchorPointModel needs to have a workdir where it saves
        temporary images and predictions. This function prepares the workdir, runs the batch prediction,
        and returns the average images per second.
        """
        logger.info(f"Generating TARDIS predictions for {len(images)} images")

        # Run the batch prediciton
        total_processing_time = self.load_and_predict_all(images)
        self._im_per_sec = len(images) / total_processing_time

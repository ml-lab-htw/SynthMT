import os
import shutil
from abc import abstractmethod

import numpy as np
import pandas as pd

from .base import BaseModel


class AnchorPointModel(BaseModel):
    """
    Base class for models that output microtubule anchor points as CSV files.
    Subclasses must implement `_run_backend(image)` that returns the CSV path.
    """

    def __init__(
        self,
        model_name: str,
        work_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self._work_dir = self.ensure_work_dir(work_dir)

    @abstractmethod
    def _run_backend(self, image) -> str:
        """
        Run the model backend and return the absolute path to the CSV file
        containing anchor points.
        """
        pass

    def ensure_work_dir(self, work_dir: str | None) -> str:

        if work_dir is not None:
            work_dir = os.path.join(work_dir, f"{self.model_name}_workdir")
            if os.path.exists(work_dir):
                # Delete all contents of the directory
                for filename in os.listdir(work_dir):
                    file_path = os.path.join(work_dir, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
            os.makedirs(work_dir, exist_ok=True)
        return work_dir

    def predict(self, image) -> list[np.ndarray]:
        """
        Run backend and parse the resulting CSV into per-instance anchor point arrays.
        """
        csv_path = self._run_backend(image)
        if csv_path is None:
            return []

        df = pd.read_csv(csv_path)

        anchor_points_instance_masks = [
            group[["X [A]", "Y [A]"]].to_numpy() for _, group in df.groupby("IDs")
        ]

        # plt.imshow(image.astype(np.uint8))
        # for pt in anchor_points_instance_masks:
        #     # plt.plot(pt[:, 0], pt[:, 1], "x", markersize=5)
        #     pt_h = plt.plot(pt[0, 0], pt[0, 1], ".", markersize=10)
        #     plt.plot(pt[-1, 0], pt[-1, 1], ".", markersize=10, color=pt_h[0].get_color())
        # plt.show()

        return anchor_points_instance_masks if anchor_points_instance_masks else []

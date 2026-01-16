# src/benchmark/models/drift.py
import numpy as np
from .base import BaseModel

class DRIFT(BaseModel):
    def __init__(self, **kwargs):
        super().__init__("DRIFT", **kwargs)

    def predict(self, image: np.ndarray) -> np.ndarray:
        # Placeholder: returns an empty array of masks
        print(f"Predicting with {self.model_name}...")
        return np.empty((0, *image.shape[:2]), dtype=np.uint16)

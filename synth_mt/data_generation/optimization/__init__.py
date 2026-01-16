# Expose main optimization API
from .eval import evaluate_tuning_cfg, evaluate_synthetic_data_cfg
from .embeddings import ImageEmbeddingExtractor
from .objective import objective
from .optimization import run_optimization
from .metrics import similarity, compute_cosine_score, compute_mahalanobis_score, compute_frechet_distance

__all__ = [
    "evaluate_tuning_cfg", "evaluate_synthetic_data_cfg",
    "ImageEmbeddingExtractor",
    "objective",
    "run_optimization",
    "similarity", "compute_cosine_score", "compute_mahalanobis_score", "compute_frechet_distance"
]

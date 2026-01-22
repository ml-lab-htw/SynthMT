import logging
from typing import Type, Dict, List, Optional

from synth_mt.benchmark.models.base import BaseModel

logger = logging.getLogger(__name__)


# Optional imports - these may fail if dependencies are not installed
def _safe_import(module_path: str, class_name: str) -> Optional[Type[BaseModel]]:
    """Safely import a model class, returning None if import fails."""
    try:
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, ModuleNotFoundError, RuntimeError) as e:
        logger.debug(f"Could not import {class_name} from {module_path}: {e}")
        return None


class ModelFactory:
    """A factory for creating model instances."""

    def __init__(self):
        self._models: Dict[str, Type[BaseModel]] = {}

    def register_model(self, model_class: Optional[Type[BaseModel]]):
        """
        Registers a model class with the factory.
        The model name is retrieved from the class's `get_model_name` method.
        """
        if model_class is not None:
            name = model_class.__name__.lower()
            if name in self._models:
                raise ValueError(f"Model '{name}' is already registered.")
            self._models[name] = model_class

    def create_model(self, name: str, **kwargs) -> BaseModel:
        """
        Creates an instance of a registered model by its name.

        Args:
            name: The name of the model to create.
            **kwargs: Additional keyword arguments to pass to the model's constructor.

        Returns:
            An instance of the specified model.
        """
        model_class = self._models.get(name.lower())
        if not model_class:
            raise ValueError(
                f"Model '{name}' not registered. Available models: {self.get_available_models()}"
            )
        return model_class(**kwargs)

    def get_available_models(self) -> List[str]:
        """Returns a list of all registered model names."""
        return sorted(list(self._models.keys()))


def setup_model_factory() -> ModelFactory:
    """Initializes and registers all models with the factory.

    Models with missing dependencies will be skipped and logged.
    """
    factory = ModelFactory()

    # Define models with their import paths
    model_imports = [
        ("synth_mt.benchmark.models.fiesta", "FIESTA"),
        ("synth_mt.benchmark.models.sam", "SAM"),
        ("synth_mt.benchmark.models.sam2", "SAM2"),
        ("synth_mt.benchmark.models.sam3", "SAM3"),
        ("synth_mt.benchmark.models.sam3", "SAM3Text"),
        ("synth_mt.benchmark.models.cellsam", "CellSAM"),
        ("synth_mt.benchmark.models.micro_sam", "MicroSAM"),
        ("synth_mt.benchmark.models.cellpose_sam", "CellposeSAM"),
        ("synth_mt.benchmark.models.stardist", "StarDist"),
        ("synth_mt.benchmark.models.tardis", "TARDIS"),
    ]

    for module_path, class_name in model_imports:
        model_class = _safe_import(module_path, class_name)
        if model_class is not None:
            factory.register_model(model_class)
        else:
            logger.warning(f"Model {class_name} not available (missing dependencies)")

    return factory

import copy
import json
import logging
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, get_type_hints, Union
from inspect import signature

import yaml

logger = logging.getLogger(__name__)


class BaseConfig(ABC):
    @classmethod
    def load(cls, config_path: Optional[str] = None, overrides: Optional[dict] = None):
        """
        Loads a configuration, recursively handling nested BaseConfig objects.

        CHANGED: After loading, it stores the source file path on the instance
        so it can be saved back to the same location later.
        """
        logger.debug(f"Attempting to load configuration for '{cls.__name__}'...")
        data = {}
        config_path_obj = None

        if config_path:
            config_path_obj = Path(config_path)  # Use pathlib for robustness
            logger.debug(f"Loading from file: {config_path_obj}")

            if not config_path_obj.is_file():
                logger.error(f"Config file not found: {config_path_obj}")
                raise FileNotFoundError(f"Config file not found: {config_path_obj}")

            try:
                with config_path_obj.open("r") as f:
                    if config_path_obj.suffix.lower() in (".yml", ".yaml"):
                        data = yaml.safe_load(f) or {}
                        logger.debug(f"Loaded YAML data from {config_path_obj}")
                    elif config_path_obj.suffix.lower() == ".json":
                        data = json.load(f) or {}
                        logger.debug(f"Loaded JSON data from {config_path_obj}")
                    else:
                        msg = f"Unsupported config file format: '{config_path_obj.suffix}'. Use .yml, .yaml, or .json"
                        logger.error(msg)
                        raise ValueError(msg)
            except Exception as e:
                logger.error(f"Error reading config file {config_path_obj}: {e}", exc_info=True)
                raise

        if overrides:
            logger.debug(f"Applying overrides: {overrides}")
            data.update(overrides)
            logger.debug(f"Data after overrides: {data}")

        # Ensure 'save_dir' is included in the configuration data
        if "save_dir" not in data:
            data["save_dir"] = ".models"  # Default value for save_dir

        # Create the instance from the loaded data
        instance = cls.from_dict(data)
        logger.debug(f"Created instance of '{cls.__name__}' from data.")

        # Store the source path on the instance for the new save() method
        if config_path_obj:
            instance._source_path = config_path_obj.resolve()
            logger.debug(
                f"Configuration for '{cls.__name__}' loaded successfully from {instance._source_path}."
            )
        else:
            instance._source_path = None
            logger.debug(
                f"Configuration for '{cls.__name__}' loaded/created in memory (no source file)."
            )

        return instance

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a config instance from a dictionary, recursively creating nested objects.
        This method handles the conversion of lists to tuples for fields that are typed as tuples.
        """
        data_copy = copy.deepcopy(data)
        type_hints = get_type_hints(cls)

        for key, field_type in type_hints.items():
            if key in data_copy:
                value = data_copy[key]
                origin = getattr(field_type, "__origin__", None)

                # Handle Optional[T] by getting the inner type
                if origin is Union and type(None) in getattr(field_type, "__args__", ()):
                    non_none_args = [
                        t for t in getattr(field_type, "__args__", ()) if t is not type(None)
                    ]
                    if len(non_none_args) == 1:
                        field_type = non_none_args[0]
                        origin = getattr(field_type, "__origin__", None)

                # Recursively create nested config objects
                if (
                    isinstance(value, dict)
                    and isinstance(field_type, type)
                    and issubclass(field_type, BaseConfig)
                ):
                    data_copy[key] = field_type.from_dict(value)
                # Convert lists to tuples if the type hint is a tuple
                elif isinstance(value, list) and origin is tuple:
                    data_copy[key] = tuple(value)

        # Filter out keys that are not in the class's __init__
        init_keys = {p.name for p in signature(cls).parameters.values()}
        filtered_data = {k: v for k, v in data_copy.items() if k in init_keys}

        return cls(**filtered_data)

    def save(self, path: Optional[str] = None):
        """
        NEW: Saves the current configuration state to a file.

        If a path is provided, it saves to that path. If no path is provided,
        it attempts to save back to the original file it was loaded from.

        Raises:
            ValueError: If no path is provided and the config was not loaded
                        from a file (i.e., it was created in memory).
        """
        logger.debug(f"Attempting to save configuration for '{self.__class__.__name__}'...")
        # 1. Determine the target path for saving
        if path:
            target_path = Path(path)  # Ensure target_path is a Path object
            logger.debug(f"Saving to user-provided path: {target_path}")
        elif hasattr(self, "_source_path") and self._source_path:
            target_path = Path(self._source_path)  # Convert to Path object
            logger.debug(f"Saving to original source path: {target_path}")
        else:
            msg = "Cannot save config: No path was provided and the config was not loaded from a file."
            logger.error(msg)
            raise ValueError(msg)

        # 2. Determine the format and call the appropriate writer method
        suffix = target_path.suffix.lower()
        try:
            if suffix == ".json":
                self.to_json(target_path)
                logger.debug(f"Configuration converted to JSON for saving.")
            elif suffix in [".yml", ".yaml"]:
                self.to_yml(target_path)
                logger.debug(f"Configuration converted to YAML for saving.")
            else:
                msg = f"Unsupported file extension: '{suffix}'. Please use .json, .yml, or .yaml."
                logger.error(msg)
                raise ValueError(msg)
        except Exception as e:
            logger.error(f"Error while saving config to {target_path}: {e}", exc_info=True)
            raise

        # 3. After a successful save, update the internal source path to the new location
        self._source_path = target_path.resolve()
        logger.debug(
            f"Configuration for '{self.__class__.__name__}' successfully saved to {self._source_path}"
        )

    def update(self, params: dict):
        """
        Updates the configuration, recursively handling nested objects.
        """
        logger.debug(
            f"Updating configuration for '{self.__class__.__name__}' with parameters: {list(params.keys())}"
        )
        type_hints = get_type_hints(self.__class__)
        for key, value in params.items():
            if not hasattr(self, key):
                msg = (
                    f"Invalid configuration key: '{key}' not found in '{self.__class__.__name__}'."
                )
                logger.error(msg)
                raise KeyError(msg)

            current_attr = getattr(self, key)
            if isinstance(value, dict) and isinstance(current_attr, BaseConfig):
                logger.debug(f"Recursively updating nested config for key '{key}'.")
                current_attr.update(value)
            else:
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.debug(f"Updated key '{key}' from '{old_value}' to '{value}'.")
        logger.debug(f"Configuration for '{self.__class__.__name__}' updated successfully.")

    def asdict(self):
        """
        Converts the config to a dictionary, recursively handling nested objects.
        Attributes starting with an underscore are ignored.
        """
        params_to_save = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") or callable(value):
                continue
            if isinstance(value, BaseConfig):
                params_to_save[key] = value.asdict()
            elif isinstance(value, Path):
                params_to_save[key] = str(value)
            else:
                params_to_save[key] = value
        return params_to_save

    def __eq__(self, other):
        # if not isinstance(other, BaseConfig):
        #     return NotImplemented
        return self.asdict() == other.asdict()

    def get(self, key):
        return getattr(self, key)

    def __str__(self):
        return yaml.dump(self.asdict(), sort_keys=False, indent=2)

    def to_yml(self, path: str):
        logger.debug(f"Writing YAML config to {path}...")
        path_obj = Path(path)
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with path_obj.open("w") as f:
                yaml.dump(self.asdict(), f, indent=2)
            logger.debug(f"YAML config written successfully to {path}.")
        except Exception as e:
            logger.error(f"Failed to write YAML config to {path}: {e}", exc_info=True)
            raise

    def to_json(self, path: str):
        logger.debug(f"Writing JSON config to {path}...")
        path_obj = Path(path)
        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            with path_obj.open("w") as f:
                json.dump(self.asdict(), f, indent=2)
            logger.debug(f"JSON config written successfully to {path}.")
        except Exception as e:
            logger.error(f"Failed to write JSON config to {path}: {e}", exc_info=True)
            raise

    def copy(self, deep: bool = True):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    @classmethod
    def from_yml(cls, path: str):
        logger.debug(f"Calling .load() to load '{cls.__name__}' from YAML path: {path}")
        return cls.load(config_path=path)

    @classmethod
    def from_json(cls, path: str):
        logger.debug(f"Calling .load() to load '{cls.__name__}' from JSON path: {path}")
        return cls.load(config_path=path)

    @abstractmethod
    def validate(self):
        """
        Optionally implemented by subclasses to validate configuration logic.
        """
        pass

    def __post_init__(self):
        pass

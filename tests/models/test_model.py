import json
import os
import sys
import unittest
from abc import abstractmethod, ABC
from pathlib import Path
from unittest.mock import patch

from synth_mt.benchmark.models.factory import setup_model_factory



class TestModelBase(unittest.TestCase, ABC):

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    @property
    @abstractmethod
    def model_name(self):
        pass

    @property
    @abstractmethod
    def config_path(self):
        pass

    def setUp(self):
        self.model_dir = ".models"
        self.temp_dir = ".temp"
        sys.path.insert(0, self.PROJECT_ROOT)

    def test_factory_load(self):
        with patch.object(Path, 'exists', return_value=True):
            with open(self.config_path, "r") as f:
                params = json.load(f)

            factory = setup_model_factory()
            params['save_dir'] = self.model_dir
            params['temp_dir'] = self.temp_dir

            self.model = factory.create_model(self.model_name, **params)

            # Patch the correct model class for each model type
            if self.model_name.lower() == "cellposesam":
                patch_target = 'cellpose.models.CellposeModel'
            elif self.model_name.lower() == "tardis":
                patch_target = 'synth_mt.benchmark.models.tardis.TARDIS'
            elif self.model_name.lower() == "stardist":
                patch_target = 'synth_mt.benchmark.models.stardist.StarDist'
            else:
                patch_target = None

            if patch_target:
                with patch(patch_target, autospec=True) as MockModel:
                    instance = MockModel.return_value
                    # Ensure the mock has a load_model method
                    instance.load_model = lambda *args, **kwargs: None
                    self.model.load_model()
            else:
                # If no patch needed, just call load_model
                self.model.load_model()

            self.assertEqual(self.model._save_dir, ".models")


    @abstractmethod
    def test_predict_batch(self):
        pass

    @abstractmethod
    def test_predict(self):
        pass

    @abstractmethod
    def test_load(self):
        pass

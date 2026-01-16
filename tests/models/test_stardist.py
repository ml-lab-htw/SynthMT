
import os
import unittest
from pathlib import Path

from tests.models.test_model import TestModelBase

class TestStardistModel(TestModelBase):
    model_name = "stardist"
    config_path = Path(os.path.join(TestModelBase.PROJECT_ROOT, "synth_mt/benchmark/models/stardist_default.json"))

    def test_predict_batch(self):
        pass

    def test_predict(self):
        pass

    def test_load(self):
        pass

    def test_factory_load(self):
        pass # TODO implement test

if __name__ == '__main__':
    unittest.main()



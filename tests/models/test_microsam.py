
import os
import unittest
from pathlib import Path

from tests.models.test_model import TestModelBase

class TestMicroSamModel(TestModelBase):
    model_name = "microsam"
    config_path = Path(os.path.join(TestModelBase.PROJECT_ROOT, "synth_mt/benchmark/models/microsam_default.json"))

    def test_predict_batch(self):
        pass

    def test_predict(self):
        pass

    def test_load(self):
        pass

    def test_factory_load(self):
        # TODO implement test
        pass

if __name__ == '__main__':
    unittest.main()



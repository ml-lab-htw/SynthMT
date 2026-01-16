import os
import unittest
from pathlib import Path

from tests.models.test_model import TestModelBase


class TestCellPoseSamModel(TestModelBase):
    model_name = "cellposesam"
    config_path = Path(os.path.join(TestModelBase.PROJECT_ROOT, "synth_mt/benchmark/models/cellposesam_default.json"))

    def test_predict_batch(self):
        pass

    def test_predict(self):
        pass

    def test_load(self):
        pass


if __name__ == '__main__':
    unittest.main()

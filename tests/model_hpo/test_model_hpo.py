import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import unittest
from unittest.mock import patch, MagicMock
import optuna
from synth_mt.model_hpo import model_hpo


@patch('synth_mt.utils.postprocessing.get_area_length_ranges', return_value=(0, 100, 0, 100))
class TestModelHPO(unittest.TestCase):
    def test_define_search_space(self, _mock_get_area_length_ranges):
        mock_trial = MagicMock(spec=optuna.Trial)

        # Test for 'cellposesam'
        params = model_hpo.define_search_space(mock_trial, 'cellposesam')
        self.assertIn('cellprob_threshold', params)

        # Test for 'microsam'
        params = model_hpo.define_search_space(mock_trial, 'microsam')
        self.assertIn('foreground_threshold', params)

        # Test for 'fiesta'
        params = model_hpo.define_search_space(mock_trial, 'fiesta')
        self.assertTrue(params['grayscale'])

        with self.assertRaises(ValueError):
            model_hpo.define_search_space(mock_trial, 'unknown_model')


if __name__ == '__main__':
    unittest.main()

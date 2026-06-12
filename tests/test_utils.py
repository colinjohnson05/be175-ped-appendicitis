import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import random

sys.path.append(str(Path.cwd().parent / 'code'))

from app_utils import compute_costs_over_thresholds

class TestUtils(unittest.TestCase):
    def test_compute_costs_over_threshold(self):
        # Given
        np.random.seed(42)
        N_test = 10
        N_thresholds = 20
        y_test = [random.randint(0, 1) for _ in range(N_test)]
        y_predicted = [random.random() for _ in range(N_test)]
        thresholds = np.linspace(0, 1, N_thresholds)

        fn = 3000
        fp = 2000
        fc = 100

        # When
        costs = compute_costs_over_thresholds(y_test, y_predicted, fn, fp, fc, thresholds)

        self.assertEqual(len(costs), N_thresholds)  # add assertion here


if __name__ == '__main__':
    unittest.main()

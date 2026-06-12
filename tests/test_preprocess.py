import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / 'code'))

from data_pp import appendicitis_pp


class TestPreprocess(unittest.TestCase):
    def test_numerical(self):
        # Given
        data_path = Path.cwd() / '..' / 'data' / 'app_data_test.xlsx'

        # When
        data, descriptors = appendicitis_pp(data_path)

        # Then
        non_numeric_cols = []

        for col in data.columns:
            non_nan_values = data[col].dropna()
            if not pd.api.types.is_numeric_dtype(non_nan_values):
                non_numeric_cols.append(col)

        self.assertEqual(len(non_numeric_cols),0)

    def test_size(self):
        # Given
        data_path = Path.cwd() / '..' / 'data' / 'app_data_test.xlsx'

        # When
        original_data = pd.read_excel(data_path)
        original_rows = original_data.shape[0]
        original_columns = original_data.shape[1]

        data, descriptors = appendicitis_pp(data_path)

        # Then
        self.assertLess(data.shape[0], original_rows)
        self.assertLess(data.shape[1], original_columns)


if __name__ == '__main__':
    unittest.main()

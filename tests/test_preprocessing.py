# tests/test_preprocessing.py

import unittest
from src.preprocessing import (your_preprocessing_function)  # Import your preprocessing functions
import pandas as pd
import numpy as np

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample data for preprocessing tests
        self.df = pd.DataFrame({
            'Price': [1, 2, np.nan, 4, 5],
            'Date': pd.date_range(start='1/1/2020', periods=5)
        })

    def test_handle_missing_values(self):
        # Assuming your preprocessing function is designed to handle missing values
        processed_df = your_preprocessing_function(self.df)
        # Check that NaN is handled correctly (e.g., replaced or dropped)
        self.assertFalse(processed_df['Price'].isnull().any())

    # Add more tests specific to your preprocessing functions

if __name__ == '__main__':
    unittest.main()

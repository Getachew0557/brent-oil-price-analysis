# tests/test_oilprice_eda.py

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from src.oilprice_eda import (plot_boxplot, plot_time_series, Checking_Stationarity, 
                               data_distrbution, seasonal_decomposition, 
                               plot_acf_pacf, fit_arima, fit_garch, moving_average, 
                               log_ploting)

class TestOilPriceEDA(unittest.TestCase):
    
    def setUp(self):
        # Create a simple DataFrame for testing
        dates = pd.date_range(start='1/1/2020', periods=100)
        prices = np.random.rand(100) * 100  # Random prices between 0 and 100
        self.df = pd.DataFrame({'Date': dates, 'Price': prices})
        self.df.set_index('Date', inplace=True)

    @patch('src.oilprice_eda.plt.show')
    def test_plot_boxplot(self, mock_show):
        plot_boxplot(self.df)
        mock_show.assert_called_once()

    @patch('src.oilprice_eda.plt.show')
    def test_plot_time_series(self, mock_show):
        plot_time_series(self.df)
        mock_show.assert_called_once()

    @patch('src.oilprice_eda.plt.show')
    def test_checking_stationarity(self, mock_show):
        Checking_Stationarity(self.df)
        mock_show.assert_called_once()

    @patch('src.oilprice_eda.plt.show')
    def test_data_distribution(self, mock_show):
        data_distrbution(self.df)
        mock_show.assert_called_once()

    @patch('src.oilprice_eda.plt.show')
    def test_seasonal_decomposition(self, mock_show):
        seasonal_decomposition(self.df)
        mock_show.assert_called_once()

    @patch('src.oilprice_eda.plt.show')
    def test_plot_acf_pacf(self, mock_show):
        plot_acf_pacf(self.df)
        mock_show.assert_called()

    @patch('src.oilprice_eda.plt.show')
    def test_fit_arima(self, mock_show):
        fit_arima(self.df)
        mock_show.assert_called()

    @patch('src.oilprice_eda.plt.show')
    def test_fit_garch(self, mock_show):
        fit_garch(self.df)
        mock_show.assert_called()

    @patch('src.oilprice_eda.plt.show')
    def test_moving_average(self, mock_show):
        moving_average(self.df)
        mock_show.assert_called()

    @patch('src.oilprice_eda.plt.show')
    def test_log_plotting(self, mock_show):
        log_ploting(self.df)
        mock_show.assert_called()

if __name__ == '__main__':
    unittest.main()

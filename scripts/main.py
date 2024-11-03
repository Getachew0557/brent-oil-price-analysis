import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from load_data import load_data
from preprocessing import preprocess_data,  data_summary
from oilprice_eda import (
    plot_boxplot,
    plot_time_series,
    Checking_Stationarity,
    data_distrbution,
    seasonal_decomposition,
    plot_acf_pacf,
    fit_arima,
    fit_garch,
    moving_average,
    log_ploting,
    price_change_event,
    critical_event
)

def main():
    # Load data
    file_path = '../data/BrentOilPrices.csv'  # Update the path as necessary
    df = load_data(file_path)

    # Preprocess data
    df = preprocess_data(df)
    data_summary(df)

    # EDA and modeling
    plot_boxplot(df)
    plot_time_series(df)
    Checking_Stationarity(df)
    data_distrbution(df)
    seasonal_decomposition(df)
    plot_acf_pacf(df)
    fit_arima(df)
    fit_garch(df)
    moving_average(df)
    log_ploting(df)
    price_change_event(df)
    critical_event(df)

if __name__ == "__main__":
    main()
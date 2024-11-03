# src/oilprice_eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot

def plot_boxplot(df):
    """Plot a boxplot of the oil prices."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(y=df['Price'])
    plt.title('Boxplot of Oil Prices')
    plt.show()

def plot_time_series(df):
    """Plot the time series of Brent Oil Prices."""
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Price'], label='Brent Oil Price', color='blue')
    plt.title('Brent Oil Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per Barrel)')
    plt.legend()
    plt.grid()
    plt.show()

## checking stationarity
def Checking_Stationarity(df):
    # Checking for Stationarity
    rolling_mean = df['Price'].rolling(window=12).mean()
    rolling_std = df['Price'].rolling(window=12).std()

    plt.figure(figsize=(14, 7))
    plt.plot(df['Price'], label='Original', color='blue')
    plt.plot(rolling_mean, label='Rolling Mean', color='red')
    plt.plot(rolling_std, label='Rolling Std', color='black')
    plt.title('Rolling Mean and Standard Deviation')
    plt.legend()
    plt.show()

## Histogram of prices distrbution
def data_distrbution(df):
    # Histogram of Prices
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Price'], bins=30, kde=True)
    plt.title('Distribution of Brent Oil Prices')
    plt.xlabel('Price (USD per Barrel)')
    plt.ylabel('Frequency')
    plt.show()

#seasonal decomposition
def seasonal_decomposition(df):
    # Decompose the time series
    result = seasonal_decompose(df['Price'], model='multiplicative', period=12)

    # Plot the decomposition
    result.plot()
    plt.show()

def plot_acf_pacf(df):
    """Plot ACF and PACF."""
    plt.figure(figsize=(12, 6))
    plot_acf(df['Price'], lags=30)
    plt.title('Autocorrelation Function (ACF)')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plot_pacf(df['Price'], lags=30)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.show()

def fit_arima(df):
    """Fit an ARIMA model and plot the results."""
    price_data = df['Price']
    arima_model = ARIMA(price_data, order=(1, 1, 1))
    arima_result = arima_model.fit()
    
    # Forecast the trend
    arima_forecast = arima_result.get_forecast(steps=30)
    arima_forecast_ci = arima_forecast.conf_int()

    # Plot ARIMA results
    plt.figure(figsize=(12, 6))
    plt.plot(price_data, label='Actual Prices')
    plt.plot(arima_result.fittedvalues, color='orange', label='ARIMA Fitted')
    plt.plot(arima_forecast.predicted_mean, color='green', label='ARIMA Forecast')
    plt.fill_between(arima_forecast_ci.index, arima_forecast_ci.iloc[:, 0], arima_forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.title('ARIMA Model - Brent Oil Prices')
    plt.legend()
    plt.show()

def fit_garch(df):
    """Fit a GARCH model to the residuals of the ARIMA model."""
    price_data = df['Price']
    arima_model = ARIMA(price_data, order=(1, 1, 1))
    arima_result = arima_model.fit()
    residuals = arima_result.resid
    
    # Fit GARCH model
    garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
    garch_result = garch_model.fit()
    
    # Plot GARCH model volatility forecast
    garch_volatility = garch_result.conditional_volatility
    plt.figure(figsize=(12, 6))
    plt.plot(garch_volatility, label='Conditional Volatility (GARCH)', color='purple')
    plt.title('GARCH Model - Brent Oil Price Volatility')
    plt.legend()
    plt.show()
    
    # Summary of the GARCH model
    print(garch_result.summary())
def moving_average(df):
    # Calculate the Simple Moving Average (SMA) with different window sizes

    df['SMA_3'] = df['Price'].rolling(window=3).mean()  # 3-month moving average
    df['SMA_6'] = df['Price'].rolling(window=6).mean()  # 6-month moving average
    df['SMA_12'] = df['Price'].rolling(window=12).mean()  # 12-month moving average

    # Plot the original data and the moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Price'], label='Original Data', color='blue')
    plt.plot(df.index, df['SMA_3'], label='3-Month Moving Average', color='orange')
    plt.plot(df.index, df['SMA_6'], label='6-Month Moving Average', color='green')
    plt.plot(df.index, df['SMA_12'], label='12-Month Moving Average', color='red')

    # Add titles and labels
    plt.title('Moving Averages (1987-2024)')
    plt.xlabel('Year')
    plt.ylabel('Import Value')
    plt.legend()
    plt.grid(False)
    plt.show()

def log_ploting(df):
    # Plot lag plot
    plt.figure(figsize=(8, 8))
    lag_plot(df['Price'], lag=1)
    plt.title('Lag Plot (Lag=1)')
    plt.xlabel('Price(t)')
    plt.ylabel('Price(t-1)')
    plt.grid()
    plt.show()

events = {
        '1987-05-20': 'Iran-Iraq War continues', 
        '1987-07-24': 'Economic concerns, falling stock market', 
        '1987-09-07': 'Oil price crash begins',
        '1987-12-03': 'OPEC announces production cuts', 
        '1988-04-18': 'Increased production from non-OPEC sources',
        '1988-07-10': 'U.S. and Soviet Union relations improve', 
        '1988-08-19': 'Gulf War tensions begin',
        '1988-10-15': 'OPEC meetings lead to production agreements', 
        '1990-08-03': 'Gulf War begins',
        '1990-10-11': 'U.N. sanctions on Iraq',
        '1991-03-05': 'Gulf War ends',
        '1991-04-01': 'OPEC agrees to increase production',
        '1992-01-24': 'Economic recession in the U.S.',
        '1992-10-15': 'OPEC cuts production again',
        '1992-12-03': 'Concerns over global oversupply',
        '1993-01-14': 'Recovery from recession',
        '1993-08-25': 'Changes in refinery capacity',
        '1994-04-14': 'Rising production in the U.S.',
        '1994-11-25': 'OPEC announces new cuts to stabilize prices',
        '1994-12-16': 'Asian financial crisis',
        '1995-05-19': 'Economic growth signals in the U.S.',
        '1995-10-20': 'Increased production from Russia',
        '1995-05-19': 'Economic growth signals in the U.S.',
        '1995-10-20': 'Increased production from Russia',
        '1996-01-02': 'OPEC production cuts lead to price stabilization',
        '1996-02-15': 'U.S. economic growth increases demand',
        '1996-05-03': "OPEC's output increases after cuts",
        '1996-08-15': 'Asian financial crisis impacts global demand',
        '1996-10-18': 'Winter demand expected to rise',
        '1996-12-01': 'OPEC decides to cut production again',
        '1997-03-14': 'Asian economic crisis continues',
        '1997-06-15': "OPEC's production adjustments",
        '1997-10-30': 'Financial instability in Asia impacts demand',
        '1998-03-15': 'Economic concerns lead to reduced consumption',
        '1998-05-24': 'Global oversupply issues persist',
        '1998-08-01': 'Continued low demand and high production',
        '1998-10-08': 'OPEC agrees to cut production',
        '1999-01-01': 'Global economic recovery begins',
        '1999-04-15': 'Increased demand from recovering Asian economies',
        '1999-10-01': 'OPEC production cuts take effect',
        '2000-03-20': 'Increased demand in the U.S. and Europe',
        '2000-07-10': 'Geopolitical tensions in the Middle East',
        '2000-11-01': 'OPEC cuts production to support prices',
        '2001-01-20': 'Economic slowdown concerns affect demand',
        '2001-04-08': 'OPEC cuts production again to stabilize prices',
        '2001-06-15': 'U.S. recession concerns influence global demand',
        '2002-01-01': 'Economic recovery in the U.S. increases demand',
        '2002-03-15': 'Tensions in the Middle East heighten',
        '2002-08-01': 'U.S. military actions in Iraq affect supply',
        '2002-10-10': "OPEC's production cuts lead to price stabilization",
        '2003-01-20': 'Iraq War begins, causing supply fears',
        '2003-04-15': 'Initial phases of the Iraq War disrupt supply',
        '2003-07-01': 'Hurricane season in the Gulf of Mexico begins',
        '2003-10-30': 'OPEC agrees to increase production',
        '2004-01-01': 'Global economic growth leads to higher demand',
        '2004-04-10': 'Geopolitical tensions and instability in Nigeria',
        '2004-07-01': 'Hurricane Ivan impacts U.S. production',
        '2004-10-01': 'Supply disruptions and increasing demand',
        '2005-01-01': 'Rising demand from emerging markets',
        '2005-03-15': 'Geopolitical tensions in Iran rise',
        '2005-08-01': 'Hurricane Katrina causes severe supply disruptions',
        '2005-10-15': 'Recovery from Hurricane Katrina',
        '2006-01-01': 'Global economic growth drives demand',
        '2006-04-01': 'Escalating tensions in the Middle East',
        '2006-07-15': 'Conflict in Lebanon raises supply concerns',
        '2007-01-01': 'Market stabilization after previous highs',
        '2007-05-01': "OPEC’s output policies influence prices",
        '2007-10-01': 'Rising global demand continues',
        '2008-01-01': 'Market speculation and geopolitical instability',
        '2008-04-01': 'Global financial crisis begins affecting demand',
        '2008-10-01': 'Economic downturn leads to reduced demand',
        '2009-01-01': 'Severe global recession impacts consumption',
        '2009-04-01': 'Early signs of economic recovery',
        '2009-10-01': 'Recovery in the U.S. economy and rising demand',
        '2010-01-01': 'Continued recovery in global economies',
        '2011-01-01': 'Political unrest in the Middle East (Arab Spring)',
        '2011-03-01': 'Libyan Civil War disrupts oil supply',
        '2011-05-01': 'Global economic recovery raises demand',
        '2011-07-01': 'Debt crisis in the Eurozone impacts markets',
        '2011-10-01': 'OPEC maintains production levels',
        '2012-01-01': 'Sanctions on Iran influence supply',
        '2012-03-01': 'Tensions with Iran over nuclear program',
        '2012-05-01': 'Economic slowdown in China raises concerns',
        '2012-07-01': 'Global economic uncertainty impacts demand',
        '2012-10-01': 'Hurricane Sandy disrupts U.S. oil production',
        '2013-01-01': 'Continued recovery in U.S. economy',
        '2013-03-01': 'OPEC continues to maintain output',
        '2013-05-01': 'U.S. shale oil production increases supply',
        '2013-07-01': 'Middle East tensions and geopolitical risks',
        '2013-10-01': 'U.S. government shutdown affects markets',
        '2014-01-01': 'Global supply concerns from unrest in Iraq',
        '2014-03-01': 'Russia-Ukraine conflict raises supply fears',
        '2014-07-01': 'Continued conflict in Iraq and geopolitical risks',
        '2014-10-01': 'OPEC decides not to cut production',
        '2015-01-01': 'Oil price crash due to oversupply',
        '2015-04-01': 'Iran nuclear deal leads to fears of increased supply',
        '2015-07-01': 'U.S. production continues to rise',
        '2015-10-01': 'Global economic concerns affect demand',
        '2016-01-01': 'Continued oversupply and weak demand',
        '2016-04-01': 'Stabilization efforts by OPEC',
        '2016-07-01': 'Recovery in demand from emerging markets',
        '2016-10-01': 'OPEC agreement to cut production',
        '2017-01-01': 'Implementation of OPEC cuts',
        '2017-04-01': 'U.S. shale production increases',
        '2017-07-01': 'Increased U.S. inventories impact prices',
        '2017-10-01': 'Geopolitical tensions in the Middle East',
        '2018-01-01': 'Strong global economic growth boosts demand',
        '2018-04-01': 'U.S. sanctions on Iran announced',
        '2018-06-01': 'OPEC and allies agree to increase output',
        '2018-09-01': 'U.S.-China trade tensions impact markets',
        '2018-11-01': 'Increased U.S. production leads to oversupply',
        '2019-01-01': 'OPEC cuts production in response to oversupply',
        '2019-04-01': 'U.S. sanctions on Venezuela impact supply',
        '2019-07-01': 'Concerns over global economic slowdown',
        '2019-10-01': 'Saudi Arabia oil facility attack raises concerns',
        '2020-01-01': 'Escalating U.S.-Iran tensions',
        '2020-04-01': 'COVID-19 pandemic leads to historic price drop',
        '2020-07-01': 'Recovery in demand as economies reopen',
        '2020-10-01': 'Continued recovery but uncertainty remains',
        '2021-01-01': 'Vaccine rollout boosts global economic outlook',
        '2021-04-01': 'Rising demand from U.S. summer driving season',
        '2021-07-01': 'OPEC+ struggles to agree on output levels',
        '2021-10-01': 'Global supply chain issues impact production',
        '2022-01-01': 'Geopolitical tensions over Ukraine',
        '2022-04-01': 'Russia\'s invasion of Ukraine leads to sanctions',
        '2022-07-01': 'Global inflation concerns impact oil demand',
        '2022-10-01': 'OPEC+ cuts production to stabilize prices',
        '2023-01-01': 'China easing COVID restrictions boosts demand',
        '2023-04-01': 'Banking crisis in the U.S. raises recession fears',
        '2023-07-01': 'OPEC+ extends output cuts to support prices',
        '2023-10-01': 'Geopolitical tensions in the Middle East',
        '2024-01-01': 'Global economic recovery influences demand'
}
def price_change_event(df):
    # Define major events with their dates and descriptions

    # Plot the time series of Brent Oil Prices
    plt.figure(figsize=(20, 20))
    plt.plot(df.index, df['Price'], label='Brent Oil Price', color='blue')
    plt.title('Brent Oil Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per Barrel)')
    plt.legend()
    plt.grid()

    # Annotate each major event on the plot
    for date, event in events.items():
        event_date = pd.to_datetime(date)
        if event_date in df.index:
            price = df.loc[event_date, 'Price']
            plt.axvline(event_date, color='red', linestyle='--', alpha=0.6)
            plt.text(event_date, price + 1, event, rotation=90, verticalalignment='bottom', fontsize=8, color='darkred')

    plt.show()


def critical_event(df):
    # Plot the time series of Brent Oil Prices
    plt.figure(figsize=(20, 10))
    plt.plot(df.index, df['Price'], label='Brent Oil Price', color='blue')
    plt.title('Brent Oil Prices Over Time with Major Events')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per Barrel)')
    plt.legend()
    plt.grid()

    # Annotate events with significant price impacts
    significant_changes = {}
    for date, event in events.items():
        event_date = pd.to_datetime(date)
        if event_date in df.index:
            # Calculate percent change over 7 days to check impact
            price_before = df['Price'].get(event_date - pd.Timedelta(days=7))
            price_after = df['Price'].get(event_date + pd.Timedelta(days=7))
            
            if price_before and price_after:
                percent_change = ((price_after - price_before) / price_before) * 100
                if abs(percent_change) >= 5:  # Setting 5% as the threshold
                    significant_changes[date] = (event, percent_change)
                    # Plot vertical line and annotate significant events
                    plt.axvline(event_date, color='red', linestyle='--', alpha=0.6)
                    plt.text(event_date, df.loc[event_date, 'Price'], 
                            f"{event}\n({percent_change:.2f}%)", 
                            rotation=90, verticalalignment='bottom', fontsize=8, color='darkred')

    plt.show()

    # Display significant changes with events
    for date, (event, change) in significant_changes.items():
        print(f"Date: {date}, Event: {event}, Impact: {change:.2f}%")

    
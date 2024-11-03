# src/preprocessing.py
import pandas as pd
import numpy as np
def data_summary(df):
    print("Size of Dataset:", df.shape)
    print("Statistics of data:", df.describe())
    print("Data types of data", df.dtypes)

def preprocess_data(df):
    """Preprocess the DataFrame: handle missing values, convert date, and remove outliers."""
    #identifying missing values
    missing_values = df.isnull().sum()
    print(f"Missing Values:\n{missing_values}")

    #Duplicated values
    print(f"the number of duplicated value:\n{df.duplicated().sum()}")

    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Set 'Date' as the index
    df.set_index('Date', inplace=True)
    
    # Ensure 'Price' is a float
    df['Price'] = df['Price'].astype(float)

    # Remove outliers using IQR method
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Price'] < (Q1 - 1.5 * IQR)) | (df['Price'] > (Q3 + 1.5 * IQR)))]

    # Example of creating new features from the Date index
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day

    return df

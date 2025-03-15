import numpy as np
import pandas as pd
import streamlit as st
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
#selected_ticker = "BLZE" 
import api_functions as api
import joblib
import re

def prepare_company_data(df, ticker):
    """
    Prepare model data for a specific company based on its ticker.
    """
    df_company = df[df['ticker'] == ticker].copy()  # Filter for selected company
    
    # Ensure data is sorted correctly
    df_company = df_company.sort_values(by='date')

    # Creating lag features (last 3 days closing prices)
    df_company['close_t-1'] = df_company['close'].shift(1)
    df_company['close_t-2'] = df_company['close'].shift(2)
    df_company['close_t-3'] = df_company['close'].shift(3)

    # Target Variable: Next day's price movement (1 if up, 0 if down)
    df_company['target'] = (df_company['close'].shift(-1) > df_company['close']).astype(int)

    # Drop NaN values caused by shifting
    df_company.dropna(inplace=True)

    return df_company[['date', 'close_t-3', 'close_t-2', 'close_t-1', 'target']]


def transform_dataframe(df,ticker):
    
    # Step 1: Rename existing columns to match target schema
    rename_dict = {
        'Date': 'date',
        'Dividend Paid': 'dividend',
        'Common Shares Outstanding': 'shares_outstanding',
        'Last Closing Price': 'close',
        'Adjusted Closing Price': 'adj._close',
        'Highest Price': 'high',
        'Lowest Price': 'low',
        'Opening Price': 'open',
        'Trading Volume': 'volume'
    }
    df = df.rename(columns=rename_dict)

    # Step 2: Add missing columns with default values
    print(df.columns)
    if 'ticker' not in df.columns:
        df['ticker'] = ticker  # or assign a default value if known
    
    # Step 3: Reorder columns to match target structure
    target_columns = [
        'ticker','date', 'open', 'high', 'low', 'close',
        'adj._close', 'volume', 'dividend', 'shares_outstanding'
    ]
    df = df[target_columns]

    return df
def get_latest_stock_data(ticker):
    
    simfin = api.SimFinAPI(ticker)

    # Fetch company info and stock data (defaults to last two weeks)
    simfin.fetch_company_info()
    simfin.fetch_stock_data()  # Uses last two weeks by default

    # Get and display DataFrames
    #df_company = simfin.get_company_dataframe()
    #df_stock = simfin.get_stock_dataframe()

    latest_data = simfin.get_stock_dataframe()

    # Example usage:
    latest_data = transform_dataframe(latest_data,ticker)


    latest_data = latest_data.drop(columns=["dividend"], errors="ignore")

    # Apply the function
    latest_data = prepare_company_data(latest_data, ticker)
    # Check the processed data
    #print(f"✅ Data Prepared for {selected_ticker}. Shape:", latest_data.shape)
    #print("🔍 Sample Data:\n", latest_data.head())

    return latest_data
def predict_next_day_xgboost_api(model, ticker):
    """
    Fetch latest stock data from SimFin and predict whether the stock will go up or down.
    """
    # Fetch latest stock data from SimFin API
    latest_data = get_latest_stock_data(ticker)

    if latest_data is None:
        print("❌ Could not retrieve latest stock data. Aborting prediction.")
        return

    # Ensure feature alignment: Keep only the features used in training
    model_features = model.feature_names_in_
    latest_data = latest_data[model_features]  # Select only relevant columns

    # Make Prediction
    prediction = model.predict(latest_data)
    prediction_label = "📈 Up" if prediction[0] == 1 else "📉 Down"

    return prediction_label

# Function to extract available companies from model filenames
def get_available_companies(model_dir):
        companies = []
        if os.path.exists(model_dir):  # Check if folder exists
            for filename in os.listdir(model_dir):
                match = re.match(r"xgb_model_final_(.+)\.pkl", filename)  # Extract ticker from filename
                if match:
                    companies.append(match.group(1))  # Add ticker to list
        return sorted(companies)  # Return sorted company tickers


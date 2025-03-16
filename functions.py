import pandas as pd
import streamlit as st
import os
import api_functions as api
import re
from lightweight_charts.widgets import StreamlitChart

def get_df_api(ticker):

    simfin = api.SimFinAPI(ticker)

    # Fetch company info and stock data (defaults to last 4 weeks)
    simfin.fetch_company_info()
    simfin.fetch_stock_data() 

    df_company = simfin.get_company_dataframe()
    df_stock = simfin.get_stock_dataframe()
    
    return df_company,df_stock

def show_prediction(prediction_label):

    prediction_label_clean = prediction_label.split()[1].lower()

    # Define colors and messages based on prediction
    if prediction_label_clean == "up":
        bg_color = "#1a1a1a"
        text_color = "#00ff7f"  # Green text
        border_color = "#00ff7f"  # Green border
        shadow_color = "rgba(0, 255, 127, 0.4)"  # Green glow
        icon = "📈"
        action_text = "BUY ✅"
    elif prediction_label_clean == "down":
        bg_color = "#1a1a1a"
        text_color = "#ff4c4c"  # Red text
        border_color = "#ff4c4c"  # Red border
        shadow_color = "rgba(255, 76, 76, 0.4)"  # Red glow
        icon = "📉"
        action_text = "SELL ❌"
    else:
        bg_color = "#1a1a1a"
        text_color = "#ffffff"
        border_color = "#cccccc"
        shadow_color = "rgba(200, 200, 200, 0.4)"
        icon = "❔"
        action_text = "NO SIGNAL ⚠️"

    # Force CSS changes to apply
    st.markdown(
        f"""
        <p style="
            background-color: {bg_color};
            color: {text_color};
            font-size: 22px;
            font-weight: bold;
            text-align: center;
            border: 2px solid {border_color};
            box-shadow: 0px 4px 12px {shadow_color};
            padding: 15px;
            border-radius: 10px;
            display: inline-block;
            width: 100%;
        ">
            Prediction for Tomorrow: <br>
            <span style="font-size: 26px; font-weight: bold;">{prediction_label}</span> <br>
            Recommended Action: <span style="font-size: 26px; font-weight: bold;">{action_text}</span>
        </p>
        """,
        unsafe_allow_html=True
    )

def show_graph(df_stock,df_company,chart_col, data_col,ticker,period ):
    # Ensure 'Date' column is in datetime format
    stock_data = df_stock.copy()
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

            # Add 'ticker' column to df_stock by assigning the correct ticker
    stock_data['ticker'] = df_company['ticker'].iloc[0]  # Assign first available ticker

            # Merge stock data with df_company to ensure correct mapping
    stock_data = stock_data.merge(df_company[['ticker', 'name']], on='ticker', how='left')

            # Filter the selected time period
    latest_date = stock_data['Date'].max()
    stock_data = stock_data[(stock_data['ticker'] == ticker) & 
                                    (stock_data['Date'] >= latest_date - pd.DateOffset(days=period))]

            # Drop unnecessary columns
    stock_data = stock_data.drop(columns=['Dividend Paid', 'Common Shares Outstanding', 
                                                  'Adjusted Closing Price', 'ticker', 'name'], errors='ignore')

            # Ensure numeric columns are properly formatted
    numeric_cols = ['Last Closing Price', 'Opening Price', 'Highest Price', 'Lowest Price', 'Trading Volume']
    for col in numeric_cols:
        if col in stock_data.columns:
            stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')

    with chart_col:
        with st.container():
            chart = StreamlitChart(height=500, width=950) 

            # Enable better grid visibility
            chart.grid(vert_enabled=True, horz_enabled=True)

            # Layout settings for improved visibility
            chart.layout(
                background_color='#131722', 
                font_family='Trebuchet MS', 
                font_size=16
            )

            # Enable auto price scaling (Fixes `price_scale_mode` error)
            chart.price_scale(auto_scale=True)

            # Ensure candlesticks fill the width of the chart
            chart.time_scale(min_bar_spacing=20) # Adjust for wider candles

            # Styling for candlesticks
            chart.candle_style(
                up_color='#2962ff', down_color='#e91e63',
                border_up_color='#2962ff', border_down_color='#e91e63',
                wick_up_color='#2962ff', wick_down_color='#e91e63'
            )

            # Improve volume bars
            chart.volume_config(
                up_color='#2962ff', down_color='#e91e63'
            )

            # Enable OHLC data + percentage change in legend
            chart.legend(visible=True, font_family='Trebuchet MS', ohlc=True, percent=True)

            # Ensure correct Date format before renaming columns
            stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.strftime('%Y-%m-%d')

            # Rename columns for compatibility with the chart
            hist_df = stock_data.rename(columns={
                'Date': 'time',
                'Last Closing Price': 'close',
                'Opening Price': 'open',
                'Highest Price': 'high',
                'Lowest Price': 'low',
                'Trading Volume': 'volume'
            })

            # Load data into the chart
            chart.set(hist_df)
            chart.load()



                    
    with data_col:
        st.dataframe(hist_df, height=500)  # ✅ Table height matches chart

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

def get_latest_stock_data(ticker,df_stock):
    
    latest_data = transform_dataframe(df_stock,ticker)

    latest_data = latest_data.drop(columns=["dividend"], errors="ignore")

    # Apply the function
    latest_data = prepare_company_data(latest_data, ticker)

    return latest_data

def predict_next_day_xgboost_api(model, ticker,df_stock):
    """
    Predict whether the stock will go up or down.
    """
    latest_data = get_latest_stock_data(ticker,df_stock)
    print(latest_data)
    if latest_data is None:
        print("❌ Could not retrieve latest stock data. Aborting prediction.")
        return

    # Ensure feature alignment: Keep only the features used in training
    model_features = model.feature_names_in_
    latest_data = latest_data[model_features]  # Select only relevant columns

    # Make Prediction
    prediction = model.predict(latest_data)
    prediction_label = "📈 Up" if prediction[0] == 1 else "📉 Down"

    return prediction,prediction_label

def get_available_companies(model_dir):
        companies = []
        if os.path.exists(model_dir):  # Check if folder exists
            for filename in os.listdir(model_dir):
                match = re.match(r"xgb_model_final_(.+)\.pkl", filename)  # Extract ticker from filename
                if match:
                    companies.append(match.group(1))  # Add ticker to list
        return sorted(companies)  # Return sorted company tickers


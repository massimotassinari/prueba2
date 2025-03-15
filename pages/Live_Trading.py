# ---------------------------
# LIVE TRADING PAGE
# ---------------------------

##HEADER

import streamlit as st
import pandas as pd
import api_functions as api
from lightweight_charts.widgets import StreamlitChart
import functions as function
import joblib
import os

def show_graph(df_stock,df_company):
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

            # ✅ Ensure numeric columns are properly formatted
    numeric_cols = ['Last Closing Price', 'Opening Price', 'Highest Price', 'Lowest Price', 'Trading Volume']
    for col in numeric_cols:
        if col in stock_data.columns:
            stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')

    with chart_col:
        with st.container():
            chart = StreamlitChart(height=500, width=950)  # ✅ Fixed width

            # Enable better grid visibility
            chart.grid(vert_enabled=True, horz_enabled=True)

            # Layout settings for improved visibility
            chart.layout(
                background_color='#131722', 
                font_family='Trebuchet MS', 
                font_size=16
            )

            # ✅ Enable auto price scaling (Fixes `price_scale_mode` error)
            chart.price_scale(auto_scale=True)

            # ✅ Ensure candlesticks fill the width of the chart
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

            # ✅ Ensure correct Date format before renaming columns
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
def get_df_api(ticker):
    simfin = api.SimFinAPI(ticker)

    # Fetch company info and stock data (defaults to last two weeks)
    simfin.fetch_company_info()
    simfin.fetch_stock_data()  # Uses last two weeks by default

    df_company = simfin.get_company_dataframe()
    df_stock = simfin.get_stock_dataframe()
    


    return df_company,df_stock

if "symbols_list" not in st.session_state:
    st.session_state.symbols_list = None
    
st.set_page_config(
    layout = 'wide',
    page_title = 'Automated Trading System'
)

st.markdown(
    """
    <style>
        footer {display: none}
        [data-testid="stHeader"] {display: none}
    </style>
    """, unsafe_allow_html = True
)
# Define the folder path
model_dir = "trained_models"


# Get list of available companies
allowed_tickers = function.get_available_companies(model_dir)

#Falta un try except
ticker = allowed_tickers[0]

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)



title_col, emp_col, company_col, open_col, high_col, low_col, close_col = st.columns([1,0.5,1,1,1,1,1])


## BODY
# Define Streamlit columns (Chart & Table now equal)
params_col, chart_col, data_col = st.columns([0.5, 1, 1])  # ✅ Chart & Table same width



# Ensure df_company contains allowed tickers
#if 'ticker' in df_company.columns:
#    available_tickers = df_company[df_company['ticker'].isin(allowed_tickers)]['ticker'].unique()
#else:
#    available_tickers = allowed_tickers  # Fallback in case df_company is missing tickers
#'''
with params_col:
    with st.form(key='params_form'):
        st.markdown(f'<p class="params_text">CHART DATA PARAMETERS', unsafe_allow_html=True)
        st.divider()
        
        # Dropdown for selecting ticker
        ticker = st.selectbox('Select Stock Ticker', allowed_tickers, key='ticker_selectbox',index=4)
        #######
        df_company,df_stock = get_df_api(ticker)
        # Saving chache
        st.session_state.ticker = ticker
        st.session_state.df_stock = df_stock.copy()
        st.session_state.df_company = df_company.copy()
        
        ######   
        # Select data range (kept only this option)
        period_selection = st.selectbox("Select Data Range", 
                                        ["1 Day", "3 Days", "5 Days", "1 Week", "1 Month"], 
                                        key='period_selectbox',index=4)
        
        # Map period selection to numerical days
        period_map = {
            "1 Day": 1,
            "3 Days": 3,
            "5 Days": 5,
            "1 Week": 7,
            "1 Month": 30
        }
        period = period_map[period_selection]

        st.markdown('')
        update_chart = st.form_submit_button('Update chart')
        st.markdown('')

        
        if update_chart or len(allowed_tickers)>0:
            show_graph(df_stock,df_company)


with title_col:
    st.markdown('<p class="dashboard_title">Automated <br>Trading <br>System</p>', unsafe_allow_html = True)

    st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

with company_col:
    with st.container():
        ticker = df_company.iloc[-1]['ticker']
        st.markdown(f'<p class="company_text">Ticker<br></p><p class="stock_details">{ticker}</p>', unsafe_allow_html = True)

with open_col:
    with st.container():
        opening_price = df_stock.iloc[-1]['Opening Price']
        st.markdown(f'<p class="open_text">Open<br></p><p class="stock_details">{opening_price}</p>', unsafe_allow_html = True)

with high_col:
    with st.container():
        high_price = df_stock.iloc[-1]['Highest Price']
        st.markdown(f'<p class="high_text">High<br></p><p class="stock_details">{high_price}</p>', unsafe_allow_html = True)

with low_col:
    with st.container():
        low_price = df_stock.iloc[-1]['Lowest Price']
        st.markdown(f'<p class="low_text">Low<br></p><p class="stock_details">{low_price}</p>', unsafe_allow_html = True)

with close_col:
    with st.container():
        close_price = df_stock.iloc[-1]['Last Closing Price']
        st.markdown(f'<p class="close_text">Close<br></p><p class="stock_details">{close_price}</p>', unsafe_allow_html = True)
    

# Construct the model file path dynamically
model_path = os.path.join(model_dir, f"xgb_model_final_{ticker}.pkl")

    # Load the model
xgb_model_final = joblib.load(model_path)

prediction_df,prediction_label = function.predict_next_day_xgboost_api(xgb_model_final, ticker,df_stock)
    # Button to confirm selection
st.session_state.prediction_df = prediction_df.copy()

st.success(f"The prediction for tomorrows is that the stock goes: {prediction_label}")

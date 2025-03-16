import streamlit as st
from lightweight_charts.widgets import StreamlitChart
import functions as function
import joblib
import os


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

#ticker = allowed_tickers[0]

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)


st.markdown('<p class="dashboard_title"> Automated Trading System</p>', unsafe_allow_html = True)

st.markdown("<div style='margin-bottom: 50px'></div>", unsafe_allow_html=True)

space_col, title_col, emp_col, company_col, open_col, high_col, low_col, close_col = st.columns([0.3,1,0.1,1,1,1,1,1])

## BODY
# Define Streamlit columns (Chart & Table now equal)
params_col, chart_col, data_col = st.columns([0.5, 1, 1])  # ✅ Chart & Table same width


with params_col:
    with st.form(key='params_form'):
        st.markdown(f'<p class="params_text">CHART DATA PARAMETERS', unsafe_allow_html=True)
        st.divider()
        
        # Dropdown for selecting ticker
        ticker = st.selectbox('Select Stock Ticker', allowed_tickers, key='ticker_selectbox',index=4)
        
        df_company,df_stock = function.get_df_api(ticker)

        # Saving chache
        st.session_state.ticker = ticker
        st.session_state.df_stock = df_stock.copy()
        st.session_state.df_company = df_company.copy()
        # End saving cache

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
            function.show_graph(df_stock,df_company,chart_col, data_col,ticker,period  )


with title_col:
    # Load the logo from the `logos/` folder
    logo_path = f"logos/{ticker}.png"
    # Check if logo exists, otherwise use placeholder
    if os.path.exists(logo_path):
        logo_url = logo_path
    else:
        logo_url = "logos/nf.png"  # Default placeholder if logo is missing

    
    st.image(logo_url, width=100)

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

# Get prediction
prediction_df,prediction_label = function.predict_next_day_xgboost_api(xgb_model_final, ticker,df_stock)

# Button to confirm selection
st.session_state.prediction_df = prediction_df.copy()

# SHow the prediction and recomended action
function.show_prediction(prediction_label)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

st.set_page_config(
    layout="wide",
    page_title="Trading Strategy Analysis"
)

st.title("Trading Strategy Performance")

# General Explanation of How AI Trading Works
st.subheader("How Does AI Trading Work?")
st.write("""
This AI trading strategy predicts whether a stock's price will go **up** or **down** the next day:
- **Prediction `1`** → AI expects the price to increase → **Buys stock**.
- **Prediction `0`** → AI expects the price to decrease → **Sells stock** (or holds cash).
The goal is to **maximize profit** by making better buy/sell decisions than a simple Buy & Hold strategy.
""")

# Ensure stored data exists
if "prediction_df" in st.session_state and "df_stock" in st.session_state and "df_company" in st.session_state:

    # Ensure `prediction_df` is a DataFrame
    if isinstance(st.session_state.prediction_df, np.ndarray):
        st.session_state.prediction_df = pd.DataFrame(st.session_state.prediction_df, columns=["value"])

    # Handle missing last 3 days
    missing_predictions = len(st.session_state.df_stock) - len(st.session_state.prediction_df)
    if missing_predictions > 0:
        trading_df = st.session_state.df_stock.iloc[:-missing_predictions].copy()
    else:
        trading_df = st.session_state.df_stock.copy()

    # Ensure predictions exist
    if "value" not in st.session_state.prediction_df.columns:
        st.warning("The 'value' column is missing from prediction_df.")
    else:
        trading_df["Prediction"] = st.session_state.prediction_df["value"].values

        if "Last Closing Price" not in trading_df.columns:
            st.warning("Closing prices missing from stock data.")
        else:
            # Initialize variables
            initial_balance = 1000
            balance = initial_balance
            shares = 0
            portfolio_values = []
            buy_sell_signals = []

            # Simulate trading
            for index, row in trading_df.iterrows():
                closing_price = row["Last Closing Price"]
                prediction = row["Prediction"]

                if prediction == 1 and balance > 0:  # Buy signal
                    shares = balance / closing_price
                    balance = 0
                    buy_sell_signals.append(("Buy", index, closing_price))

                elif prediction == 0 and shares > 0:  # Sell signal
                    balance = shares * closing_price
                    shares = 0
                    buy_sell_signals.append(("Sell", index, closing_price))

                # Store portfolio value
                portfolio_values.append(balance + (shares * closing_price))

            # Final portfolio value
            final_value = balance + (shares * trading_df.iloc[-1]["Last Closing Price"])

            # Buy & Hold Strategy
            initial_shares = initial_balance / trading_df.iloc[0]["Last Closing Price"]
            hold_final_value = initial_shares * trading_df.iloc[-1]["Last Closing Price"]

            # Display Company Info (Logo, Name, Last Price)
            st.subheader("Company Overview")

            col1, col2, col3 = st.columns([1, 3, 2])

            # Get company name & ticker
            company_name = st.session_state.df_company.iloc[-1]["name"]
            last_closing_price = trading_df.iloc[-1]["Last Closing Price"]
            ticker = st.session_state.df_company.iloc[-1]["ticker"] 

            # Load the logo from the `logos/` folder
            logo_path = f"logos/{ticker}.png"

            # Check if logo exists, otherwise use placeholder
            if os.path.exists(logo_path):
                logo_url = logo_path
            else:
                logo_url = "logos/nf.png"  # Default placeholder if logo is missing
            
          
            with col1:
                st.image(logo_url, width=100)

            with col2:
                st.write(f"# {company_name}")

            with col3:
                st.metric(label="Last Closing Price", value=f"${last_closing_price:.2f}")


            st.subheader("Final Portfolio Performance")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="AI Trading Strategy", value=f"${final_value:.2f}")
            with col2:
                st.metric(label="Buy & Hold Strategy", value=f"${hold_final_value:.2f}")

            # Plot Portfolio Growth Over Time
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(trading_df.index, portfolio_values, label="AI Trading Strategy", color="blue")
            ax.axhline(y=hold_final_value, color="gray", linestyle="dashed", label="Buy & Hold Final Value")
            ax.set_title("Portfolio Value Over Time")
            ax.set_xlabel("Days")
            ax.set_ylabel("Portfolio Value ($)")
            ax.legend()
            st.pyplot(fig)

            # Plot Buy & Sell Signals on Stock Price Chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(trading_df.index, trading_df["Last Closing Price"], label="Stock Price", color="black")

            # Add buy/sell markers
            for action, idx, price in buy_sell_signals:
                if action == "Buy":
                    ax.scatter(idx, price, color="green", marker="^", label="Buy Signal" if "Buy Signal" not in ax.get_legend_handles_labels()[1] else "")
                elif action == "Sell":
                    ax.scatter(idx, price, color="red", marker="v", label="Sell Signal" if "Sell Signal" not in ax.get_legend_handles_labels()[1] else "")

            ax.set_title("Stock Price with Buy & Sell Signals")
            ax.set_xlabel("Days")
            ax.set_ylabel("Stock Price ($)")
            ax.legend()
            st.pyplot(fig)

else:
    st.warning("No prediction_df data available.")


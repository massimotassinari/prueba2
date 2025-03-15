import requests
import pandas as pd
from datetime import datetime, timedelta

# SimFin API Key
API_KEY = "344dd533-861f-4bef-9f52-be02f0276014"

class SimFinAPI:
    """
    A class to fetch company and stock data from SimFin API and return them as Pandas DataFrames.
    """
    BASE_URL_GENERAL = "https://backend.simfin.com/api/v3/companies/general/compact"
    BASE_URL_PRICES = "https://backend.simfin.com/api/v3/companies/prices/compact"

    def __init__(self, ticker):
        """
        Initialize the SimFinAPI object with a stock ticker.
        """
        self.ticker = ticker
        self.company_data = None  # Raw JSON for company info
        self.stock_data = None  # Raw JSON for stock prices
        self.df_company = None  # Processed company DataFrame
        self.df_stock = None  # Processed stock DataFrame

    def fetch_company_info(self):
        """
        Fetch general company information from SimFin API.
        """
        url = f"{self.BASE_URL_GENERAL}?ticker={self.ticker}"
        headers = {
            "accept": "application/json",
            "Authorization": f"api-key {API_KEY}"
        }

        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                self.company_data = response.json()
                print(f"✅ Company Info Retrieved for {self.ticker}")
                self.process_company_info()
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"❌ Error Fetching Company Info: {e}")

    def process_company_info(self):
        """
        Convert the JSON response into a structured Pandas DataFrame.
        """
        if not self.company_data:
            print("❌ No company data available. Fetch data first.")
            return None

        try:
            # Extract columns and data
            columns = self.company_data["columns"]
            records = self.company_data["data"]

            # Convert to DataFrame
            self.df_company = pd.DataFrame(records, columns=columns)

            print(f"✅ Company Data Processed for {self.ticker}")

        except Exception as e:
            print(f"❌ Error Processing Company Data: {e}")

    def fetch_stock_data(self, start_date=None):
        """
        Fetch daily stock price data from SimFin API.
        If no start_date is provided, fetches the last two weeks of data by default.
        """
        if start_date is None:
            # Calculate the default start date (two weeks ago)
            start_date = (datetime.today() - timedelta(weeks=2)).strftime("%Y-%m-%d")

        url = f"{self.BASE_URL_PRICES}?ticker={self.ticker}&start={start_date}"
        headers = {
            "accept": "application/json",
            "Authorization": f"api-key {API_KEY}"
        }

        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                self.stock_data = response.json()
                print(f"✅ Stock Price Data Retrieved for {self.ticker} (from {start_date})")
                self.process_stock_data()
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"❌ Error Fetching Stock Data: {e}")

    def process_stock_data(self):
        """
        Convert stock data JSON response into a Pandas DataFrame.
        """
        if not self.stock_data:
            print("❌ No stock data available. Fetch data first.")
            return None

        try:
            # Extract columns and data
            columns = self.stock_data[0]["columns"]
            records = self.stock_data[0]["data"]

            # Convert to DataFrame
            self.df_stock = pd.DataFrame(records, columns=columns)

            # Convert Date column to datetime format
            self.df_stock["Date"] = pd.to_datetime(self.df_stock["Date"])

            print(f"✅ Stock Data Processed for {self.ticker}")

        except Exception as e:
            print(f"❌ Error Processing Stock Data: {e}")

    def get_company_dataframe(self):
        """
        Return the processed company data as a Pandas DataFrame.
        """
        if self.df_company is None:
            print("❌ No processed company data available. Run fetch_company_info() first.")
            return None
        return self.df_company

    def get_stock_dataframe(self):
        """
        Return the processed stock data as a Pandas DataFrame.
        """
        if self.df_stock is None:
            print("❌ No processed stock data available. Run fetch_stock_data() first.")
            return None
        return self.df_stock



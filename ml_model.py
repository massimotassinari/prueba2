import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import os
import joblib

# File paths - Update these with your actual file locations
SHARE_PRICES_FILE = "data/simfin_data/us-shareprices-daily.csv"
COMPANY_INFO_FILE = "data/simfin_data/us-companies.csv"

# Load Share Prices Data
df_share_prices_global = pd.read_csv(SHARE_PRICES_FILE, delimiter=';', parse_dates=['Date'])
print("✅ Share Prices Data Loaded. Shape:", df_share_prices_global.shape)

# Load Company Info Data
df_company_global = pd.read_csv(COMPANY_INFO_FILE, delimiter=';')
print("✅ Company Info Data Loaded. Shape:", df_company_global.shape)


selected_ticker = "BLZE" 

# Selecting only the company to be evaluated in the model
df_share_prices = df_share_prices_global[df_share_prices_global['Ticker'] == selected_ticker]
df_company = df_company_global[df_company_global['Ticker'] == selected_ticker]

print(df_share_prices.head())
print(df_company.head())

# 2. Data transformation
## 2.1 Handiling columns name convention

# Example transformations:
df_share_prices.columns = [col.strip().lower().replace(" ", "_") for col in df_share_prices.columns]  # Normalize column names
# Example transformations:
df_company.columns = [col.strip().lower().replace(" ", "_") for col in df_company.columns]  # Normalize column names

## 2.2 Data types change

df_share_prices['date'] = pd.to_datetime(df_share_prices['date'])  # Convert date columns

print(df_share_prices.info())
print(df_company.info())

## 2.3 Checking for null values
### 2.2.1 Share prices

# Check for missing values
print(df_share_prices.isnull().sum())  # Count of nulls per column


plt.figure(figsize=(10,5))
sns.heatmap(df_share_prices.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()


# Calculate percentage of missing values per column
null_percentage = (df_share_prices.isnull().sum() / len(df_share_prices)) * 100

# Display the result
print(null_percentage)

#### 2.2.1.1 Handle missing values

# Cleaning Share Prices Data
def clean_share_prices(df):
    df = df.copy()  # Avoid chained assignment warnings
    df = df.drop(columns=['dividend'], errors='ignore')  # Drop the dividend column
    df = df.assign(
        shares_outstanding=df['shares_outstanding'].ffill()  # Forward fill missing shares outstanding
    )
    df = df.sort_values(by=['ticker', 'date'])
    return df


# Apply Cleaning Functions
df_share_prices_cleaned = clean_share_prices(df_share_prices)


# Check Results
print("✅ Share Prices Cleaned. Shape:", df_share_prices_cleaned.shape)

print("🔍 Missing Values in Share Prices:\n", df_share_prices_cleaned.isnull().sum())

print(df_share_prices_cleaned)

# 4. Preparing the Model Table Data

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

# Specify a company ticker
#selected_ticker = "AAPL"  # Change this to test different companies

# Apply the function
df_company_model = prepare_company_data(df_share_prices_cleaned, selected_ticker)

# Check the processed data
print(f"✅ Data Prepared for {selected_ticker}. Shape:", df_company_model.shape)
print("🔍 Sample Data:\n", df_company_model.head())

print(df_company_model)

# 5. Traing the model

def train_xgboost_final(df_company_model):
    """
    Train an optimized XGBoost model using RandomizedSearchCV for better accuracy.
    """
    # Ensure 'target' column exists
    if 'target' not in df_company_model.columns:
        print("❌ Error: 'target' column is missing. Recalculating it now.")
        df_company_model['target'] = (df_company_model['close'].shift(-1) > df_company_model['close']).astype(int)

    # Splitting Features (X) and Target (y)
    cols_to_drop = ['date', 'close', 'target']
    cols_to_drop = [col for col in cols_to_drop if col in df_company_model.columns]  # Drop only if they exist

    print(df_company_model)
    
    X = df_company_model.drop(columns=cols_to_drop)
    y = df_company_model['target']

    # Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define XGBoost Model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Final Optimized Hyperparameters
    param_dist = {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 5],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.6, 0.8],
        'scale_pos_weight': [1.2, 1.5]  # Adjusting for class imbalance
    }

    # Perform Randomized Search
    rand_search = RandomizedSearchCV(
        model, param_distributions=param_dist, n_iter=15,
        scoring='accuracy', cv=3, n_jobs=-1, random_state=42
    )

    rand_search.fit(X_train, y_train)

    # Best Model
    best_model = rand_search.best_estimator_

    # Make Predictions
    y_pred = best_model.predict(X_test)

    # Evaluate Performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("✅ Final XGBoost Model Training Complete!")
    print(f"📊 Best Hyperparameters: {rand_search.best_params_}")
    print(f"📊 Final Optimized Accuracy: {accuracy:.4f}")
    print("\n🔍 Classification Report:\n", report)

    return best_model

# Train the final optimized model
xgb_model_final = train_xgboost_final(df_company_model)

# Define the folder path
model_dir = "trained_models"

# Create the folder if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Save the model in the folder
model_path = os.path.join(model_dir, f"xgb_model_final_{selected_ticker}.pkl")
joblib.dump(xgb_model_final, model_path)

print(f"✅ Model saved at: {model_path}")

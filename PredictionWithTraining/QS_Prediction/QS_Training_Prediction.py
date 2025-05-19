import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders import TargetEncoder
from datetime import datetime
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')
#####################################################

# ---------------Summary of the Script------------------------
# Data Loading 
# Drops unnecessary or highly correlated columns.
# Removes duplicates
# Feature Encoding
# Handling High Correlation:
# Drops columns that have high correlation or are redundant to reduce feature redundancy.
# Generating Realistic Sale Dates:
# Creates sale dates between 2015 and 2020 using weighted probabilities to mimic realistic seasonal patterns:
# Generates sale dates using a weighted random sampling method to reflect realistic business trends.

#################################################
print("----> Starting the script of QS Prediction...")
# Load the data
dfPurchases = pd.read_csv("./DataSalesPurchaseFeatured.csv")
df_raw = dfPurchases.copy() # To backup data of columns with its real values

print("----> Data loaded successfully!")
# Common columns to drop
columns_to_drop = [
    "SalesName", "ProductName", "PurchaseName", 
    "VendorTaxGroup", "VendorCurrency", "CustomerPaymentTermID", "PurchaseOrderAccount",
    "VendorGroupAvgPurchase", "CustomerGroupAvgSales", 
    "DeliveryMonthSin", "DeliveryMonthCos", "PriceRatio", "SalesPriceMarkupRatio", "SoldToPurchasedRatio"
]
dfPurchases.drop(columns=[col for col in columns_to_drop if col in dfPurchases.columns], inplace=True)

# Sort and remove duplicates
dfPurchases = dfPurchases.sort_values(by=["PurchaseYear", "PurchaseMonth"], ascending=True)
dfPurchases = dfPurchases.drop_duplicates() 

# Define categorical column groups
nominal_cols = [
    "CustomerID", "CustomerGroup", "DeliveryMode", 
    "CustomerTaxGroup", "ProductID", 
    "VendorID", "VendorGroup", "PurchaseStatus"
]
ordinal_cols = [
    "CustomerCreditRating", 
    "ProductProfitabilityCategory"
]
binary_cols = [
    "HighCreditRisk", 
    "DeliveryIsWeekend", 
    "PurchaseIsWeekend",
    "IsEndOfFiscalQuarter",
    "IsHighTurnoverProduct"
]

# 1. Nominal Categorical Encoding (Target Encoding)
target_encoder = TargetEncoder(cols=nominal_cols)
dfPurchases[nominal_cols] = target_encoder.fit_transform(
    dfPurchases[nominal_cols], dfPurchases['QuantitySold']
)

# 2. Ordinal Encoding (Label Encoding)
label_encoder = LabelEncoder()
for col in ordinal_cols:
    if col in dfPurchases.columns:
        dfPurchases[col] = label_encoder.fit_transform(dfPurchases[col].astype(str))

# 3. Binary Encoding (0/1)
for col in binary_cols:
    if col in dfPurchases.columns:
        dfPurchases[col] = dfPurchases[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
print("----> Data encoded successfully!")


# Drop high-correlation and redundant columns
drop_corr = [
    'VendorGroupAvgPurchase','CustomerGroupAvgSales','DeliveryMonthSin','DeliveryMonthCos',
    'CostOfSalesRatio','SoldToPurchasedRatio','SalesPriceMarkupRatio','PriceRatio',
    'GrossMarginPercentage','DeliveryQuarter','DeliveryIsWeekend','PurchaseIsWeekend',
    'PurchaseQuarter','HighCreditRisk','CreditRatingScore','GrossMargin',
    'VendorTaxGroup','VendorCurrency','PurchaseOrderAccount','PurchaseName',
    'ProductName','SalesName','ProductProfitabilityCategory','DeliveryYear',
    'TotalPurchaseAmount','CustomerTaxGroup','VendorGroup','CustomerCreditMax',
    'PurchasePrice','PurchaseStatus','IsHighTurnoverProduct'
]
cols_to_drop = [col for col in drop_corr if col in dfPurchases.columns]
dfSales_cleaned = dfPurchases.drop(columns=cols_to_drop)
print("----> Data cleaned and columns dropped successfully!")


# Rename columns for sales terminology
rename_mapping = {
    "PurchaseYear": "SalesYear",
    "PurchaseMonth": "SalesMonth",
    "PurchaseDayOfWeek": "SalesDayOfWeek"
}
dfSales_cleaned.rename(columns=rename_mapping, inplace=True)

# Generate sale dates based on weighted probabilities
start_date = pd.to_datetime("2015-01-01")
end_date = pd.to_datetime("2020-12-31")
date_range = pd.date_range(start=start_date, end=end_date, freq="D")

# Base weight definitions
base_weekday_weights = {i: 1 for i in range(7)}
base_weekday_weights.update({0: 1.52, 3: 1.68, 5: 0.82, 6: 0.51})
basic_month_weights = {i: 1 for i in range(1, 13)}
favored_months = [1, 3, 6, 9, 11]
month_favor = {m: (1.5 if m in favored_months else 1.0) for m in range(1, 13)}
year_weights = {2015: 1.27, 2016: 1.54, 2017: 1.35, 2018: 1.28, 2019: 1.43, 2020: 0.65}
years = list(range(2015, 2021))

year_weekday_weights = {}
year_month_weights = {}
year_week_weights_by_month = {}
for year in years:
    year_weekday_weights[year] = {d: base_weekday_weights[d] * np.random.uniform(0.75, 1.25) for d in range(7)}
    year_month_weights[year] = {m: basic_month_weights[m] * np.random.uniform(0.75, 1.55) for m in range(1, 13)}
    np.random.seed(123 + year)
    month_week_weights = {}
    for month in range(1, 13):
        base_weights = np.ones(5)
        num_favored = np.random.choice([1, 2])
        favored_weeks = np.random.choice(5, size=num_favored, replace=False)
        for wk in favored_weeks:
            base_weights[wk] *= np.random.uniform(1.5, 2.5)
        base_weights /= base_weights.sum()
        month_week_weights[month] = base_weights
    year_week_weights_by_month[year] = month_week_weights

# Helper to compute week of month
def get_week_of_month(date):
    first_day = date.replace(day=1)
    return ((date.day + first_day.weekday()) - 1) // 7 + 1

# Assemble weights arrays
year_weights_arr = np.array([year_weights[d.year] for d in date_range])
weekday_weights_arr = np.array([year_weekday_weights[d.year][d.weekday()] for d in date_range])
month_weights_arr = np.array([year_month_weights[d.year][d.month] * month_favor[d.month] for d in date_range])
week_of_month_weights = np.array([year_week_weights_by_month[d.year][d.month][min(get_week_of_month(d),5)-1] for d in date_range])
weights = weekday_weights_arr * month_weights_arr * year_weights_arr * week_of_month_weights

# Apply promotional and special date multipliers
promo_mask = np.zeros(len(date_range), dtype=bool)
midyear_mask = np.zeros(len(date_range), dtype=bool)
special_mask = np.zeros(len(date_range), dtype=bool)
for year in years:
    promo_mask |= ((date_range >= datetime(year,11,25)) & (date_range <= datetime(year,11,30)))
    midyear_mask |= ((date_range >= datetime(year,6,15)) & (date_range <= datetime(year,6,20)))
    nov1 = datetime(year,11,1)
    offset = (3 - nov1.weekday()) % 7
    bf = nov1 + pd.Timedelta(days=offset+22)
    idx = np.where(date_range==bf)[0]
    if len(idx): special_mask[idx[0]] = True
for date in date_range:
    if date.weekday()==0 and 1<=date.day<=7:
        idx = np.where(date_range==date)[0]
        if len(idx): special_mask[idx[0]] = True
weights[promo_mask] *= 2.0
weights[midyear_mask] *= 1.8
weights = np.where(special_mask, weights*1.85, weights)
weights /= weights.sum()

# Generate sale_date and update features
dfSales_generated = dfSales_cleaned.copy()
idxs = np.random.choice(np.arange(len(date_range)), size=len(dfSales_generated), p=weights)
dfSales_generated['sale_date'] = pd.Series(date_range[idxs])

def update_date_features(df):
    df['SaleYear'] = df['sale_date'].dt.year
    df['SaleMonth'] = df['sale_date'].dt.month
    df['SaleQuarter'] = df['sale_date'].dt.quarter
    df['SaleDayOfWeek'] = df['sale_date'].dt.dayofweek
    df['SaleIsWeekend'] = (df['SaleDayOfWeek'] >= 5).astype(int)
    df['SaleWeekOfMonth'] = df['sale_date'].apply(get_week_of_month)
    return df

dfSales_generated = update_date_features(dfSales_generated)
print("----> Date features added !")
# Split final test set (2020)
dfFinalTest = dfSales_generated[dfSales_generated['SaleYear']==2020].copy()
dfSales_generated = dfSales_generated[dfSales_generated['SaleYear']!=2020].copy()

dfSales_generated.drop(columns=['SalesYear',	'SalesMonth',	'SalesDayOfWeek'], inplace=True)
dfFinalTest.drop(columns=['SalesYear',	'SalesMonth',	'SalesDayOfWeek'], inplace=True)

# First, sort the data by sale_date
dfSales_generated = dfSales_generated.sort_values(by="sale_date")
dfFinalTest = dfFinalTest.sort_values(by="sale_date")

##################################################
# Data Loading and Splitting:
# Splits data into training (70%), validation (20%), and testing (10%) sets.
# Scales numerical features using StandardScaler.
# LightGBM Model Training:
# Model Evaluation
# Predicts QuantitySold for both validation and test sets.
# Computes metrics
# Model Saving
#################################################

# Compute sizes
n_total = len(dfSales_generated)
n_train = int(0.7 * n_total)
n_val = int(0.2 * n_total)

# Split the data
train_df = dfSales_generated.iloc[:n_train]
val_df = dfSales_generated.iloc[n_train:n_train + n_val]
test_df = dfSales_generated.iloc[n_train + n_val:]

print("Training data date range:", train_df['sale_date'].min(), "to", train_df['sale_date'].max())
print("Validation data date range:", val_df['sale_date'].min(), "to", val_df['sale_date'].max())
print("Test data date range:", test_df['sale_date'].min(), "to", test_df['sale_date'].max())

dfSales_generated.drop(columns=['sale_date'], inplace=True)
dfFinalTest.drop(columns=['sale_date'], inplace=True)

# Define target column
target_column = "QuantitySold"

# Identify numerical features (already encoded)
numerical_features = [
    "SalesPrice", "TotalSalesAmount", "QuantityPurchased", "InventoryCostPosted",
    "InventoryQuantityChange", "OrderToDeliveryDays", 
    "CreditUtilizationRatio", "InventoryEfficiency",
]

# Time-based features
time_features = ["SaleYear", "SaleMonth", "SaleQuarter", "SaleDayOfWeek", "SaleWeekOfMonth"]

# Categorical features to be added
categorical_features = [
    "CustomerID", "CustomerCreditRating", "CustomerGroup", "DeliveryMode",
    "ProductID", "VendorID", "FiscalQuarter", "IsEndOfFiscalQuarter", "SaleIsWeekend"
]

# Final feature list (including both numerical and categorical features)
feature_columns = numerical_features + time_features + categorical_features

# Ensure all selected features exist in the dataset
feature_columns = [col for col in feature_columns if col in train_df.columns]

# Ensure all selected features exist in the dataset
feature_columns = [col for col in feature_columns if col in dfFinalTest.columns]

# Extract features from the final test set
X_final_test = dfFinalTest[feature_columns].copy()

# Print selected features
print(f"Features used for modeling: {feature_columns}")

# Extract features (X) and target (y) for each split
# X_train, y_train = train_df[feature_columns], train_df[target_column]
# X_val, y_val = val_df[feature_columns], val_df[target_column]
# X_test, y_test = test_df[feature_columns], test_df[target_column]
# ===> this code raised an error ! 

# same but with explicit copy to pass error
X_train      = train_df[feature_columns].copy()
y_train      = train_df[target_column].copy()
X_val        =   val_df[feature_columns].copy()
y_val        =   val_df[target_column].copy()
X_test       =  test_df[feature_columns].copy()
y_test       =  test_df[target_column].copy()

# Scaling numerical features
scaler = StandardScaler()

# X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
# X_val[numerical_features] = scaler.transform(X_val[numerical_features])
# X_test[numerical_features] = scaler.transform(X_test[numerical_features])
# ===> this code raised same error as below !

# same but with using .loc on a true copy
X_train.loc[:, numerical_features]      = scaler.fit_transform(  X_train[numerical_features] )

joblib.dump(scaler, "./Meta/QS_Prediction/QS_Scaler.pkl")
print("----> Scaler saved successfully!")

X_val.loc[:,   numerical_features]      = scaler.transform(      X_val[numerical_features]   )
X_test.loc[:,  numerical_features]      = scaler.transform(      X_test[numerical_features]  )

# Standardize numerical features
X_final_test[numerical_features] = scaler.transform(X_final_test[numerical_features])

# Print dataset shapes
print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Extract features (same ones used in training)
X_final_test = dfFinalTest[feature_columns]
# This line is changed to the next script QSPredection.py
# X_final_test.loc[:, numerical_features] = scaler.transform(X_final_test[numerical_features])
print("----> Scaling complete. Data ready for predictive modeling !")

# --- Data Preparation ---
# Create LightGBM datasets for training and validation
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

evals_result = {}

# --- Model Parameters ---
# Set the parameters for LightGBM model
params = {
    'boosting_type': 'gbdt',          # Gradient Boosting Decision Tree method
    'objective': 'regression',        # Set the objective as regression
    'metric': 'rmse',                 # Set the evaluation metric to RMSE (Root Mean Squared Error)
    'num_leaves': 18,                 # Number of leaves in one tree
    'learning_rate': 0.04,            # Step size at each iteration
    'feature_fraction': 0.9,          # Fraction of features to be used for training
    'bagging_fraction': 0.8,          # Fraction of data to be used for training
    'bagging_freq': 5,                # Frequency of bagging
    'verbose': -1                     # Suppress the LightGBM logs
}

# --- Model Training ---
# Train the model using the LightGBM dataset
print("----> Training LightGBM Model...")
lgb_model = lgb.train(
    params,                          # Model parameters
    lgb_train,                       # Training dataset
    num_boost_round=100,             # Number of boosting rounds (iterations)
    valid_sets=[lgb_train, lgb_val], # Validation datasets (train and validation)
    valid_names=['train', 'valid'],  # Names of the validation sets
    callbacks=[                      # Early stopping callback to prevent overfitting
        lgb.early_stopping(stopping_rounds=100),
        lgb.record_evaluation(evals_result)
    ]
)

# --- Model Predictions ---
# Predict on the validation and test datasets
y_val_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
y_test_pred_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
print("----> Model training complete !")

# --- Metrics Calculation ---
# Calculate R², RMSE, and MAE for the validation and test predictions
lgb_val_r2 = r2_score(y_val, y_val_pred_lgb)
lgb_test_r2 = r2_score(y_test, y_test_pred_lgb)

# Calculate Adjusted R²
n_val = X_val.shape[0]  # number of validation samples
p_val = X_val.shape[1]  # number of features
lgb_val_r2_adj = 1 - (1 - lgb_val_r2) * (n_val - 1) / (n_val - p_val - 1)
n_test = X_test.shape[0]  # number of test samples
p_test = X_test.shape[1]  # same number of features
lgb_test_r2_adj = 1 - (1 - lgb_test_r2) * (n_test - 1) / (n_test - p_test - 1)

lgb_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred_lgb))
lgb_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_lgb))
lgb_val_mae = mean_absolute_error(y_val, y_val_pred_lgb)
lgb_test_mae = mean_absolute_error(y_test, y_test_pred_lgb)

print(f"Validation R²={lgb_val_r2:.4f}, Adjusted R²={lgb_val_r2_adj:.4f}, RMSE={lgb_val_rmse:.2f}, MAE={lgb_val_mae:.2f}")
print(f"Test R²={lgb_test_r2:.4f}, Adjusted R²={lgb_test_r2_adj:.4f}, RMSE={lgb_test_rmse:.2f}, MAE={lgb_test_mae:.2f}")

# --- Enregistrer le modèle LightGBM ---
model_filename = "./Meta/QS_Prediction/QS_lightgbm_model.pkl"
joblib.dump(lgb_model, model_filename)
print(f"----> Modèle enregistré sous : {model_filename}")

# Predict the target variable
predictions = lgb_model.predict(X_final_test)
# Display the first 5 predictions
print("----> Prédictions QS completed !")

# Check if target column is in the final test set
if target_column in dfFinalTest:
    y_final_actual = dfFinalTest[target_column]

    # Compute evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_final_actual, predictions))
    mae = mean_absolute_error(y_final_actual, predictions)
    r2 = r2_score(y_final_actual, predictions)

    print(f"- Final Test Data Evaluation:")
    print(f"- RMSE: {rmse:.4f},  MAE: {mae:.4f}, R² Score: {r2:.4f}")

# --- Build a results DataFrame keyed by row_id ---
df_results = pd.DataFrame({
    'row_id':    dfFinalTest['row_id'],
    'PredictedQuantitySold': predictions
})

# --- Optional: attach actuals ---
if target_column in dfFinalTest.columns:
    df_results['ActualQuantitySold'] = dfFinalTest[target_column]

# --- Merge back all original raw columns from df_raw ---
# df_raw was loaded from DataSalesPurchaseFeatured.csv and still contains every pre-encoded/scaled column
df_final = df_results.merge(df_raw, on='row_id', how='left')

# --- Save the fully-joined table ---
output_filename = './Results/QS_Prediction/QS_Predictions.csv'
df_final.to_csv(output_filename, index=False)
print(f"---->> Saved predictions plus original raw columns to {output_filename}")
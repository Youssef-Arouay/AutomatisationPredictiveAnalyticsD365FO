import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from datetime import datetime
import sys
sys.stdout.reconfigure(encoding='utf-8')
#####################################################

print("----> Starting prediction of QS...")

# Load the data
print("----> Loading data...")
dfPurchases = pd.read_csv("./DataSalesPurchaseFeatured.csv")
df_raw = dfPurchases.copy()

# Common columns to drop
columns_to_drop = [
    "SalesName", "ProductName", "PurchaseName", 
    "VendorTaxGroup", "VendorCurrency", "CustomerPaymentTermID", "PurchaseOrderAccount",
    "VendorGroupAvgPurchase", "CustomerGroupAvgSales", 
    "DeliveryMonthSin", "DeliveryMonthCos", "PriceRatio", "SalesPriceMarkupRatio", "SoldToPurchasedRatio"
]
dfPurchases.drop(columns=[col for col in columns_to_drop if col in dfPurchases.columns], inplace=True)
dfPurchases = dfPurchases.drop_duplicates() 

# Sort and remove duplicates
dfPurchases = dfPurchases.sort_values(by=["PurchaseYear", "PurchaseMonth"], ascending=True)

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

print("----> Starting data encoding...")
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
print("----> Data encoded successfully !")


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
print("----> Data cleaned and columns dropped successfully !")


print("----> Generating date...")
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

dfSales_generated.drop(columns=['SalesYear',	'SalesMonth',	'SalesDayOfWeek'], inplace=True)



# First, sort the data by sale_date
dfSales_generated = dfSales_generated.sort_values(by="sale_date")
dfSales_generated.drop(columns=['sale_date'], inplace=True)
dfSales_generated = dfSales_generated.drop_duplicates() 

#############################################


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
try:
    # Filter feature columns to keep only existing ones
    feature_columns = [col for col in feature_columns if col in dfSales_generated.columns]
    if not feature_columns:
        raise ValueError("None of the specified feature columns were found in dfSales_generated.")
except Exception as e:
    print(f"Error: {e}")

# Extract features (X) and target (y) for each split
X_dfSales_generated      = dfSales_generated[feature_columns].copy()
y_dfSales_generated      = dfSales_generated[target_column].copy()

# Scaling numerical features
# Load the saved scaler
scaler_filename = "./Meta/QS_Prediction/QS_Scaler.pkl"
print(f"----> Loading Scaler from : {scaler_filename}")
scaler = joblib.load(scaler_filename)

# same but with using .loc on a true copy
X_dfSales_generated.loc[:, numerical_features]      = scaler.transform(  X_dfSales_generated[numerical_features] )
print("----> Scaling completed !")

# Load the trained LightGBM model
model_filename = "./Meta/QS_Prediction/QS_lightgbm_model.pkl"
print(f"----> Loading model from : {model_filename}")
lgb_model = joblib.load(model_filename)

# Predict the target variable
print("----> Predicting target variable...")
y_dfSales_generated_predicted = lgb_model.predict(X_dfSales_generated)
print("----> Prédictions completed !")


target_column = 'QuantitySold'
# --- Build a results DataFrame keyed by row_id ---
df_results = pd.DataFrame({
    'row_id':    dfSales_generated['row_id'],
    'PredictedQuantitySold': y_dfSales_generated_predicted
})

# Check if target column is in the final test set
if target_column in dfSales_generated:
    # Compute evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_dfSales_generated, y_dfSales_generated_predicted))
    mae = mean_absolute_error(y_dfSales_generated, y_dfSales_generated_predicted)
    r2 = r2_score(y_dfSales_generated, y_dfSales_generated_predicted)
    print(f"----> Model Evaluation:")
    print(f"- RMSE: {rmse:.4f} |  MAE: {mae:.4f} | R² Score: {r2:.4f}")

    df_results['ActualQuantitySold'] = dfSales_generated[target_column]

# --- Merge back all original raw columns from df_raw ---
# df_raw was loaded from DataSalesPurchaseFeatured.csv and still contains every pre-encoded/scaled column
df_final = df_results.merge(df_raw, on='row_id', how='left')

# --- Save the fully-joined table ---
output_filename = './Results/QS_Prediction/QS_Predictions.csv'
df_final.to_csv(output_filename, index=False)
print(f"----> Saved predictions to {output_filename}")
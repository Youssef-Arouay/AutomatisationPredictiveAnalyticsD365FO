import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the trained LightGBM model
model_filename = "./QS_Predection/QS_lightgbm_model.pkl"
lgb_model = joblib.load(model_filename)
print(f"Modèle chargé depuis : {model_filename}")


# Load the saved scaler
scaler_filename = "./QS_Predection/QS_Scaler.pkl"
scaler = joblib.load(scaler_filename)
print(f"Scaler chargé depuis : {scaler_filename}")


# Load the test data
dfFinalTest = pd.read_csv("./QS_Predection/DataSalesPurchaseFeaturedTraited2020.csv")
# Load the original raw data
df_raw   = pd.read_csv("./DataSalesPurchaseFeatured.csv")
print(f"DataFrame chargé depuis : ./QS_Predection/DataSalesPurchaseFeaturedTraited2020.csv")

# Define the features used for prediction
numerical_features = [
    "SalesPrice", "TotalSalesAmount", "QuantityPurchased", "InventoryCostPosted",
    "InventoryQuantityChange", "OrderToDeliveryDays", 
    "CreditUtilizationRatio", "InventoryEfficiency"
]

time_features = ["SaleYear", "SaleMonth", "SaleQuarter", "SaleDayOfWeek", "SaleWeekOfMonth"]

categorical_features = [
    "CustomerID", "CustomerCreditRating", "CustomerGroup", "DeliveryMode",
    "ProductID", "VendorID", "FiscalQuarter", "IsEndOfFiscalQuarter", "SaleIsWeekend"
]

# Combine features
feature_columns = numerical_features + time_features + categorical_features

# Ensure all selected features exist in the dataset
feature_columns = [col for col in feature_columns if col in dfFinalTest.columns]

# Extract features from the final test set
X_final_test = dfFinalTest[feature_columns].copy()

# Standardize numerical features
X_final_test[numerical_features] = scaler.transform(X_final_test[numerical_features])

# Predict the target variable
predictions = lgb_model.predict(X_final_test)

# Display the first 5 predictions
print("Prédictions completed !")
target_column = 'QuantitySold'

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
output_filename = './QS_Predictions.csv'
df_final.to_csv(output_filename, index=False)
print(f"Saved predictions plus original raw columns to {output_filename}")

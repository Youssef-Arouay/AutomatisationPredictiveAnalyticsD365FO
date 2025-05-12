import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

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


# Read the CSV file into a DataFrame
dfSales_generated = pd.read_csv("./QS_Predection/DataSalesPurchaseFeaturedTraited2015-2019.csv")
dfFinalTest = pd.read_csv("./QS_Predection/DataSalesPurchaseFeaturedTraited2020.csv")


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
    # "InventoryTurnoverRate", "InventoryCostPerUnit","MonthlySalesSeasonality", 
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

joblib.dump(scaler, "./QS_Predection/QS_Scaler.pkl")
print("Scaler saved successfully!")

X_val.loc[:,   numerical_features]      = scaler.transform(      X_val[numerical_features]   )
X_test.loc[:,  numerical_features]      = scaler.transform(      X_test[numerical_features]  )


# Print dataset shapes
print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Extract features (same ones used in training)
X_final_test = dfFinalTest[feature_columns]
# This line is changed to the next script QSPredection.py
# X_final_test.loc[:, numerical_features] = scaler.transform(X_final_test[numerical_features])
print("\nScaling complete. Data ready for predictive modeling.")


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
print("Training LightGBM Model...")
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
print("Model training complete.")
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
model_filename = "./QS_Predection/QS_lightgbm_model.pkl"
joblib.dump(lgb_model, model_filename)
print(f"Modèle enregistré sous : {model_filename}")
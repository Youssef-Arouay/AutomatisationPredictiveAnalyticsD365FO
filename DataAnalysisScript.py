import pandas as pd
import numpy as np

# Step 1: Load and rename columns
column_names = [
    "SalesOrderID", "SalesName", "SalesOrderDate", "ExpectedDeliveryDate", "DeliveryName",
    "CustomerID", "CustomerCreditMax", "CustomerCreditRating", "CustomerGroup", "DeliveryMode",
    "PaymentMode", "CustomerTaxGroup", "CustomerPaymentTermID",
    "ProductID", "QuantitySold", "SalesPrice", "TotalSalesAmount", "PurchaseOrderID", "PurchaseOrderAccount", "PurchaseName", "PurchaseOrderDate",
    "PurchaseDeliveryName", "PurchaseStatus", "VendorID", "VendorGroup", "VendorTaxGroup", "VendorPaymentTermID", "VendorCurrency",
    "QuantityPurchased", "PurchasePrice", "PurchasePriceUnit", "TotalPurchaseAmount",
    "InventoryTransactionDate", "InventoryCostPosted", "PhysicalTransactionDate",
    "InventoryQuantityChange", "InventoryIssueStatus", "InventoryReceiptStatus"
]

data = pd.read_csv("DataExported.csv", header=None, names=column_names)
print("[Step 1] Data loaded and columns renamed successfully!")

# Step 2: Drop unneeded columns (Group 1)
cols_to_drop1 = [
    'PurchasePriceUnit', 'PurchaseDeliveryName', 'DeliveryName',
    'InventoryIssueStatus', 'InventoryReceiptStatus',
    'InventoryTransactionDate', 'PhysicalTransactionDate',
    'SalesOrderDate', 'PaymentMode', 'VendorPaymentTermID','SalesOrderID', 'PurchaseOrderID',
]
data.drop(columns=cols_to_drop1, inplace=True, errors='ignore')
data.drop_duplicates(inplace=True)  # Drop duplicates after dropping columns
print("[Step 2] Group 1 columns dropped!")

# Step 3: Handle missing values
categorical_cols = [
    'CustomerCreditMax', 'CustomerCreditRating', 'DeliveryMode',
    'CustomerTaxGroup', 'VendorTaxGroup', 'VendorCurrency'
]
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode())

numerical_cols = [
    'TotalPurchaseAmount', 'QuantityPurchased', 'SalesPrice',
    'TotalSalesAmount', 'PurchasePrice', 'InventoryCostPosted'
]
for col in numerical_cols:
    data[col] = data[col].fillna(data[col].median())


print("[Step 3] Missing values handled!")

# Step 5: Remove outliers
numeric_columns = data.select_dtypes(include=[np.number]).columns

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[((df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR)))]


for col in numeric_columns:
    data = remove_outliers(data, col)
print("[Step 4] Outliers removed!")


# Step 6: Drop duplicates
data.drop_duplicates(inplace=True)
print("[Step 5] Duplicates dropped!")



# Step 7: Feature engineering function
def feature_engineering(df):
    df_processed = df.copy()
    # Convert date columns
    for col in ['ExpectedDeliveryDate', 'PurchaseOrderDate']:
        df_processed[col] = pd.to_datetime(
            df_processed[col],
            format="%m/%d/%Y %H:%M",
            errors="coerce",
            exact=False
        )
    # Time components
    for prefix, col in [('Delivery', 'ExpectedDeliveryDate'), ('Purchase', 'PurchaseOrderDate')]:
        df_processed[f'{prefix}Year'] = df_processed[col].dt.year
        df_processed[f'{prefix}Month'] = df_processed[col].dt.month
        df_processed[f'{prefix}Quarter'] = df_processed[col].dt.quarter
        df_processed[f'{prefix}DayOfWeek'] = df_processed[col].dt.dayofweek
        df_processed[f'{prefix}IsWeekend'] = (df_processed[f'{prefix}DayOfWeek'] >= 5).astype(int)
    # Delay
    df_processed['OrderToDeliveryDays'] = (
        df_processed['ExpectedDeliveryDate'] - df_processed['PurchaseOrderDate']
    ).dt.days
    # Financial metrics
    df_processed['GrossMargin'] = df_processed['TotalSalesAmount'] - df_processed['TotalPurchaseAmount']
    df_processed['GrossMarginPercentage'] = np.divide(
        df_processed['GrossMargin'], df_processed['TotalSalesAmount'],
        out=np.zeros_like(df_processed['GrossMargin']),
        where=df_processed['TotalSalesAmount'] != 0
    ) * 100
    df_processed['CostOfSalesRatio'] = np.divide(
        df_processed['TotalPurchaseAmount'], df_processed['TotalSalesAmount'],
        out=np.zeros_like(df_processed['TotalPurchaseAmount']),
        where=df_processed['TotalSalesAmount'] != 0
    )
    # Inventory metrics
    df_processed['InventoryEfficiency'] = np.divide(
        df_processed['QuantitySold'], df_processed['QuantityPurchased'],
        out=np.zeros_like(df_processed['QuantitySold']),
        where=df_processed['QuantityPurchased'] != 0
    )
    # Credit metrics
    df_processed['CreditUtilizationRatio'] = np.divide(
        df_processed['TotalSalesAmount'], df_processed['CustomerCreditMax'],
        out=np.zeros_like(df_processed['TotalSalesAmount']),
        where=df_processed['CustomerCreditMax'] != 0
    )
    credit_rating_map = {'Excellent': 5, 'Good': 4, 'Fair': 3, 'Poor': 2, 'Bad': 1}
    df_processed['CreditRatingScore'] = df_processed['CustomerCreditRating'].replace(credit_rating_map).infer_objects()
    df_processed['HighCreditRisk'] = ((
        df_processed['CreditUtilizationRatio'] > 0.8
    ) | (df_processed['CreditRatingScore'] < 3)).astype(int)
    return df_processed

# Step 8: Apply feature engineering
data_fe = feature_engineering(data)
print("[Step 6] Feature engineering applied!")

# Drop intermediate date columns
data_fe.drop(columns=['ExpectedDeliveryDate', 'PurchaseOrderDate'], inplace=True, errors='ignore')
data_fe.drop_duplicates(inplace=True)  # Drop duplicates after feature engineering

data_fe['row_id'] = np.arange(len(data_fe))


# Step 9: Save final DataFrame
output_path = 'DataSalesPurchaseFeatured.csv'
data_fe.to_csv(output_path, index=False)
print(f"[Step 7] Processed data saved to {output_path}!")

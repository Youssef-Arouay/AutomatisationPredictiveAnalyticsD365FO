import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from datetime import datetime

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

# Load the data
dfPurchases = pd.read_csv("DataSalesPurchaseFeatured.csv")

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

# Preserve unencoded columns
df_unencodedColumns = dfPurchases[["CustomerID", "ProductID"]].copy()
df_unencodedColumns.set_index(dfPurchases.index, inplace=True)

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
print("1/5: Data encoded successfully!")


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
print("2/5: Data cleaned and columns dropped successfully!")


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
print("3/5: Date features added !")
# Split final test set (2020)
dfFinalTest = dfSales_generated[dfSales_generated['SaleYear']==2020].copy()
dfSales_generated = dfSales_generated[dfSales_generated['SaleYear']!=2020].copy()

dfSales_generated.drop(columns=['SalesYear',	'SalesMonth',	'SalesDayOfWeek'], inplace=True)
dfFinalTest.drop(columns=['SalesYear',	'SalesMonth',	'SalesDayOfWeek'], inplace=True)



# First, sort the data by sale_date
dfSales_generated = dfSales_generated.sort_values(by="sale_date")
dfSales_generated.to_csv("./QS_Predection/DataSalesPurchaseFeaturedTraited2015-2019.csv", index=False)
print("4/5: Data saved to DataSalesPurchaseFeaturedTraited2015-2019.csv")

dfFinalTest.to_csv("./QS_Predection/DataSalesPurchaseFeaturedTraited2020.csv", index=False)
print("5/5: Data saved to DataSalesPurchaseFeaturedTraited2020.csv")


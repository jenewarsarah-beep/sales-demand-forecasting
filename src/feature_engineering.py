
import pandas as pd

# Load cleaned sales data
df = pd.read_csv("data/cleaned_sales_data.csv")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Convert individual orders into weekly product demand
weekly = (
    df.groupby([
        pd.Grouper(key="Order_Date", freq="W-MON"),
        "Product",
        "Region"
    ])["Quantity"]
    .sum()
    .reset_index()
)

weekly = weekly.sort_values(
    ["Product", "Region", "Order_Date"]
).reset_index(drop=True)

# Create date-based features
weekly["Month"] = weekly["Order_Date"].dt.month
weekly["Day_of_Week"] = weekly["Order_Date"].dt.dayofweek
weekly["Week_of_Year"] = weekly["Order_Date"].dt.isocalendar().week.astype(int)

# Create historical demand features
grouped = weekly.groupby(["Product", "Region"])["Quantity"]

weekly["Previous_Week_Demand"] = grouped.shift(1)
weekly["Previous_2_Week_Demand"] = grouped.shift(2)
weekly["Average_Past_4_Weeks"] = grouped.transform(
    lambda x: x.shift(1).rolling(4, min_periods=4).mean()
)

# Remove rows without sufficient historical data
weekly = weekly.dropna().reset_index(drop=True)

# Save features
weekly.to_csv("data/features.csv", index=False)

print("Feature engineering completed!")
print("\nFirst five rows:")
print(weekly.head())

print("\nDataset shape:", weekly.shape)
print("\nColumns:")
print(weekly.columns.tolist())

print("\nSaved to data/features.csv")

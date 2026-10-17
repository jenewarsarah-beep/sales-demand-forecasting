
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data/sales_data.csv")

print("----- ORIGINAL DATA -----")
print(df.head())
print("\nDataset shape:", df.shape)

# Convert Order_Date to datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Check missing values
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()

# Create month column
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# Total revenue
total_revenue = df["Sales"].sum()
print("\nTotal Revenue: Rs.", total_revenue)

# Total quantity sold
total_quantity = df["Quantity"].sum()
print("Total Quantity Sold:", total_quantity)

# Best-selling products by quantity
product_sales = df.groupby("Product")["Quantity"].sum()
product_sales = product_sales.sort_values(ascending=False)

print("\n----- PRODUCT SALES -----")
print(product_sales)

# Revenue by region
region_sales = df.groupby("Region")["Sales"].sum()
region_sales = region_sales.sort_values(ascending=False)

print("\n----- REVENUE BY REGION -----")
print(region_sales)

# Monthly revenue
monthly_sales = df.groupby("Month")["Sales"].sum()

print("\n----- MONTHLY REVENUE -----")
print(monthly_sales)

# Plot product sales
product_sales.plot(kind="bar", title="Quantity Sold by Product")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")
plt.tight_layout()
plt.savefig("data/product_sales.png")
plt.show()

# Plot monthly revenue
monthly_sales.plot(kind="line", marker="o", title="Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (Rs.)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/monthly_sales.png")
plt.show()

# Save cleaned data
df.to_csv("data/cleaned_sales_data.csv", index=False)

print("\nAnalysis completed successfully!")
print("Charts saved in the data folder.")

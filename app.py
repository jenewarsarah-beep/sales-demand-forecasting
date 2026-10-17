
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Sales & Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.title("Sales & Demand Forecasting System")
st.caption("Sales analytics and machine-learning prediction dashboard")

sales_path = DATA_DIR / "cleaned_sales_data.csv"
predictions_path = DATA_DIR / "predictions.csv"

if not sales_path.exists():
    st.error("Cleaned dataset not found. Run src/analysis.py first.")
    st.stop()

df = pd.read_csv(sales_path)
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Sidebar filters
st.sidebar.header("Dashboard Filters")

products = ["All"] + sorted(df["Product"].dropna().unique().tolist())
regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())

selected_product = st.sidebar.selectbox("Select Product", products)
selected_region = st.sidebar.selectbox("Select Region", regions)

filtered = df.copy()

if selected_product != "All":
    filtered = filtered[filtered["Product"] == selected_product]

if selected_region != "All":
    filtered = filtered[filtered["Region"] == selected_region]

# Key performance indicators
total_revenue = filtered["Sales"].sum()
total_units = filtered["Quantity"].sum()
total_orders = len(filtered)
average_order_value = (
    total_revenue / total_orders if total_orders else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
c2.metric("Units Sold", f"{total_units:,}")
c3.metric("Number of Orders", f"{total_orders:,}")
c4.metric("Average Order Value", f"₹{average_order_value:,.2f}")

st.divider()

# Monthly revenue chart
st.subheader("Monthly Revenue Trend")

filtered = filtered.copy()
filtered["Month"] = filtered["Order_Date"].dt.to_period("M").astype(str)

monthly = filtered.groupby("Month")["Sales"].sum()

if not monthly.empty:
    st.line_chart(monthly)

# Product performance
left, right = st.columns(2)

with left:
    st.subheader("Units Sold by Product")
    product_units = filtered.groupby("Product")["Quantity"].sum()
    st.bar_chart(product_units)

with right:
    st.subheader("Revenue by Region")
    region_revenue = filtered.groupby("Region")["Sales"].sum()
    st.bar_chart(region_revenue)

# Prediction results
st.divider()
st.subheader("Demand Prediction Results")

if predictions_path.exists():
    predictions = pd.read_csv(predictions_path)

    if "Order_Date" in predictions.columns:
        predictions["Order_Date"] = pd.to_datetime(
            predictions["Order_Date"]
        )

    if selected_product != "All":
        predictions = predictions[
            predictions["Product"] == selected_product
        ]

    if selected_region != "All":
        predictions = predictions[
            predictions["Region"] == selected_region
        ]

    if not predictions.empty:
        st.dataframe(predictions,width="stretch")

        if {
            "Quantity", "Predicted_Quantity"
        }.issubset(predictions.columns):
            comparison = predictions[
                ["Quantity", "Predicted_Quantity"]
            ].reset_index(drop=True)

            st.write("Actual versus Predicted Demand")
            st.line_chart(comparison)
    else:
        st.info("No prediction records match these filters.")
else:
    st.warning("Run src/model.py to generate predictions.")

# Download filtered data
st.divider()

st.download_button(
    "Download Filtered Sales Data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_sales_data.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully!")

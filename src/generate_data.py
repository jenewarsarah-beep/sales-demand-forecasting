
import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

products = {
    "Laptop": 55000,
    "Mouse": 800,
    "Keyboard": 1500,
    "Monitor": 12000,
    "Headphones": 2500
}

categories = {
    "Laptop": "Electronics",
    "Mouse": "Accessories",
    "Keyboard": "Accessories",
    "Monitor": "Electronics",
    "Headphones": "Accessories"
}

regions = ["Chennai", "Bangalore", "Hyderabad", "Mumbai", "Delhi"]

data = []
start_date = datetime(2025, 1, 1)

for i in range(1000):
    product = random.choice(list(products.keys()))
    quantity = random.randint(1, 10)
    unit_price = products[product]
    order_date = start_date + timedelta(days=random.randint(0, 364))

    data.append({
        "Order_ID": f"ORD{i + 1:04d}",
        "Order_Date": order_date,
        "Product": product,
        "Category": categories[product],
        "Region": random.choice(regions),
        "Quantity": quantity,
        "Unit_Price": unit_price,
        "Sales": quantity * unit_price
    })

df = pd.DataFrame(data)

df.to_csv("data/sales_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nDataset shape:", df.shape)

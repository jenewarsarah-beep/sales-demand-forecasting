
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load engineered features
df = pd.read_csv("data/features.csv")

# Select input features and prediction target
features = [
    "Month",
    "Day_of_Week",
    "Week_of_Year",
    "Previous_Week_Demand",
    "Previous_2_Week_Demand",
    "Average_Past_4_Weeks"
]

X = df[features]
y = df["Quantity"]

# Split data chronologically: earlier observations for training
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    min_samples_leaf=2
)

model.fit(X_train, y_train)

# Predict demand on the test period
predictions = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("----- MODEL EVALUATION -----")
print(f"Mean Absolute Error (MAE): {mae:.2f} units")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} units")
print(f"R2 Score: {r2:.3f}")

# Save the trained model
with open("data/demand_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save test predictions for inspection
results = df.iloc[split_index:][
    ["Order_Date", "Product", "Region", "Quantity"]
].copy()

results["Predicted_Quantity"] = predictions.round(2)
results.to_csv("data/predictions.csv", index=False)

print("\nModel saved to data/demand_model.pkl")
print("Predictions saved to data/predictions.csv")

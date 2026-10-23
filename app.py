import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
df = pd.read_csv(
    "data/Sample - Superstore.csv",
    encoding="latin1"
)

# Convert Order Date to datetime
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%m/%d/%Y"
)

# Sort data by date
df = df.sort_values("Order Date")

# Create monthly sales
monthly_sales = (
    df.set_index("Order Date")
      .resample("ME")["Sales"]
      .sum()
      .reset_index()
)

print("Monthly Sales:")
print(monthly_sales.head(12))

print("\nNumber of months:", len(monthly_sales))

# Plot monthly sales trend
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()

# Prepare data for forecasting
monthly_sales["Month_Number"] = np.arange(len(monthly_sales))

X = monthly_sales[["Month_Number"]]
y = monthly_sales["Sales"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Forecast next 6 months
future_month_numbers = np.arange(
    len(monthly_sales),
    len(monthly_sales) + 6
).reshape(-1, 1)

future_sales = model.predict(future_month_numbers)

# Create future dates
future_dates = pd.date_range(
    start=monthly_sales["Order Date"].max(),
    periods=7,
    freq="ME"
)[1:]

forecast = pd.DataFrame({
    "Order Date": future_dates,
    "Forecast Sales": future_sales
})

print("\nNext 6 Months Sales Forecast:")
print(forecast)
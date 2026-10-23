import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error,mean_squared_error

from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="AI Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# Load dataset
df = pd.read_csv(
    "data/Sample - Superstore.csv",
    encoding="latin1"
)

# Convert date
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%m/%d/%Y"
)

# Dashboard title
st.title("📈 AI Sales Forecasting Dashboard")
st.write("Interactive analysis of Superstore sales performance and trends.")

# -----------------------
# Sidebar filters
# -----------------------

st.sidebar.header("Filters")

# Date range filter
min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

# Filter data

if len(date_range) == 2:
    start_date, end_date = date_range

    filtered_df = df[
        (df["Region"].isin(region)) &
        (df["Category"].isin(category)) &
        (df["Order Date"].dt.date >= start_date) &
        (df["Order Date"].dt.date <= end_date)
    ]
else:
    filtered_df = df[
        (df["Region"].isin(region)) &
        (df["Category"].isin(category))
    ]
# -----------------------
# KPI cards
# -----------------------

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order ID"].nunique()
total_customers = filtered_df["Customer ID"].nunique()

profit_margin = (
    (total_profit / total_sales) * 100
    if total_sales != 0
    else 0
)

col1, col2, col3, col4 ,col5= st.columns(5)

col1.metric("Total Sales", f"${total_sales:,.0f}")
col2.metric("Total Profit", f"${total_profit:,.0f}")
col3.metric("Total Orders", f"{total_orders:,}")
col4.metric("Total Customers", f"{total_customers:,}")
col5.metric("Profit Margin", f"{profit_margin:.1f}%")
st.divider()

# -----------------------
# Monthly sales trend
# -----------------------

monthly_sales = (
    filtered_df
    .set_index("Order Date")
    .resample("ME")["Sales"]
    .sum()
    .reset_index()
)

st.subheader("Monthly Sales Trend")

fig_sales = px.line(
    monthly_sales,
    x="Order Date",
    y="Sales",
    markers=True
)

st.plotly_chart(fig_sales, use_container_width=True)

# -----------------------
# Sales by category
# -----------------------

category_sales = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

st.subheader("Sales by Category")

fig_category = px.bar(
    category_sales,
    x="Category",
    y="Sales"
)

st.plotly_chart(fig_category, use_container_width=True)

# -----------------------
# Data preview
# -----------------------

st.subheader("Sales Data")

st.dataframe(
    filtered_df,
    use_container_width=True
)
# -----------------------
# Top 10 Products
# -----------------------

st.subheader("Top 10 Products by Sales")

top_products = (
    filtered_df
    .groupby("Product Name")["Sales"]
    .sum()
    .nlargest(10)
    .reset_index()
)

fig_products = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    title="Top 10 Best-Selling Products"
)

fig_products.update_layout(
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(
    fig_products,
    use_container_width=True
)
# -----------------------
# Sales Forecast
# -----------------------

st.subheader("6-Month Sales Forecast")

# Prepare monthly sales for machine learning
forecast_data = monthly_sales.copy()

forecast_data["Month_Number"] = np.arange(len(forecast_data))

X = forecast_data[["Month_Number"]]
y = forecast_data["Sales"]

# -----------------------
# Train/Test Evaluation
# -----------------------

# Keep the last 6 months for testing
split_index = len(forecast_data) - 6

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train evaluation model
evaluation_model = LinearRegression()
evaluation_model.fit(X_train, y_train)

# Predict unseen test months
test_predictions = evaluation_model.predict(X_test)

# Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train on historical training data
rf_model.fit(X_train, y_train)

# Predict test period
rf_predictions = rf_model.predict(X_test)

# Calculate Random Forest errors
rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = mean_squared_error(
    y_test,
    rf_predictions
) ** 0.5

# Calculate evaluation metrics
mae = mean_absolute_error(
    y_test,
    test_predictions
)

rmse = mean_squared_error(
    y_test,
    test_predictions
) ** 0.5

# Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = mean_squared_error(
    y_test,
    rf_predictions
) ** 0.5

# Create test results dataframe
test_results = pd.DataFrame({
    "Order Date": forecast_data["Order Date"].iloc[split_index:],
    "Actual Sales": y_test.values,
    "Predicted Sales": test_predictions
})

st.subheader("Model Evaluation: Actual vs Predicted Sales")

fig_evaluation = px.line(
    test_results,
    x="Order Date",
    y=["Actual Sales", "Predicted Sales"],
    markers=True,
    title="Actual vs Predicted Sales - Test Period"
)

st.plotly_chart(
    fig_evaluation,
    use_container_width=True
)

# Train final model on all available data
model = LinearRegression()
model.fit(X, y)




st.write("### Model Performance")

st.write("### Model Comparison")

comparison_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        mae,
        rf_mae
    ],
    "RMSE": [
        rmse,
        rf_rmse
    ]
})

st.dataframe(
    comparison_df,
    use_container_width=True
)

metric1, metric2 = st.columns(2)

metric1.metric(
    "Mean Absolute Error (MAE)",
    f"${mae:,.2f}"
)

metric2.metric(
    "Root Mean Squared Error (RMSE)",
    f"${rmse:,.2f}"
)

# Predict next 6 months
future_month_numbers = np.arange(
    len(forecast_data),
    len(forecast_data) + 6
).reshape(-1, 1)

future_sales = model.predict(future_month_numbers)

# Create future dates
future_dates = pd.date_range(
    start=forecast_data["Order Date"].max(),
    periods=7,
    freq="ME"
)[1:]

forecast_df = pd.DataFrame({
    "Order Date": future_dates,
    "Forecast Sales": future_sales
})

# Display forecast values
st.dataframe(
    forecast_df,
    use_container_width=True
)

# Create forecast chart
historical = monthly_sales[["Order Date", "Sales"]].copy()
historical["Type"] = "Historical"
historical = historical.rename(columns={"Sales": "Amount"})

future = forecast_df.rename(
    columns={"Forecast Sales": "Amount"}
)
future["Type"] = "Forecast"

chart_data = pd.concat(
    [historical, future],
    ignore_index=True
)

fig_forecast = px.line(
    chart_data,
    x="Order Date",
    y="Amount",
    color="Type",
    markers=True,
    title="Historical Sales vs Forecast"
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)
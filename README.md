# AI Sales Forecasting Dashboard

An interactive sales analytics and forecasting dashboard built using Python, Streamlit, Plotly, and Machine Learning.

## Project Overview

This project analyzes historical Superstore sales data and provides an interactive dashboard for exploring business performance.

It also uses a Linear Regression model to generate a 6-month baseline sales forecast based on historical monthly sales.

## Features

- Interactive sales dashboard
- Region-based filtering
- Category-based filtering
- Total Sales KPI
- Total Profit KPI
- Total Orders KPI
- Total Customers KPI
- Monthly sales trend analysis
- Sales by category visualization
- 6-month sales forecasting
- Historical vs forecast comparison
- Interactive sales data table

## Technologies Used

- Python
- Pandas
- NumPy
- Streamlit
- Plotly
- Scikit-learn
- Matplotlib

## Machine Learning

The project uses a Linear Regression model as a baseline forecasting approach.

Historical sales transactions are aggregated into monthly sales totals. A time index is then used as the input feature to predict sales for the next six months.

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

## Project Structure

ai-sales-forecasting-dashboard/
│
├── data/
│   └── Sample - Superstore.csv
├── app.py
├── dashboard.py
├── requirements.txt
└── README.md

## Installation

Clone the repository and install the required Python libraries:

pip install -r requirements.txt

## Run the Dashboard

Run the following command:

streamlit run dashboard.py

Then open the local Streamlit URL in your browser.

## Dataset

The project uses the Sample Superstore dataset containing order, customer, product, sales, discount, and profit information.

## Future Improvements

- Add more advanced time-series forecasting models
- Evaluate forecasting accuracy using MAE/RMSE
- Add date-range filters
- Add product and sub-category analysis
- Add regional profit analysis
- Deploy the dashboard online

## Author

Dakshayini M
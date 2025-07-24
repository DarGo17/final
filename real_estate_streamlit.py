import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import joblib
import requests
from datetime import datetime

# Load Model and Data
# Load Model and Data
model = joblib.load('prophet_model.pkl')
combined_df = joblib.load('combined_df.pkl')  # Includes Date, price, MORTGAGE30US
historical = combined_df.copy()  # define historical for clarity
forecast = model.predict(model.make_future_dataframe(periods=120, freq='M'))


# RentCast API Setup
API_KEY = "bc87829e1d2d4ee68dcbb775c90b598a"
VALUE_URL = "https://api.rentcast.io/v1/avm/value"
HEADERS = {"X-Api-Key": API_KEY}

def fetch_property_value(address):
    params = {"address": address}
    response = requests.get(VALUE_URL, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Value API Error {response.status_code}: {response.text}")
        return None

# Streamlit Page Setup
st.set_page_config(page_title="Forecast and Value Comparison", layout="wide")
st.title("Address-Based Home Value Forecast and Analysis")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["Forecast Explorer", "Property Insights", "Mortgage Calculator"])

# Tab 1 - Forecast Explorer
conflict_periods = {
    "Great Recession (2008-2009)": ("2008-01-01", "2009-12-31"),
    "War on Terror (2008-2011)": ("2008-01-01", "2011-12-31"),
    "Arab Spring (2011-2014)": ("2011-01-01", "2014-12-31"),
    "Crimea Annexation (2014)": ("2014-01-01", "2014-12-31"),
    "US-China Trade War (2018-2019)": ("2018-07-01", "2019-12-31"),
    "COVID-19 (2020-2022)": ("2020-03-01", "2022-06-30"),
    "Russia-Ukraine War (2022-2025)": ("2022-02-01", "2025-01-01"),
    "Israel–Hamas Escalation (2023-2025)": ("2023-10-01", "2025-01-01")
}

st.title("Fayetteville Home Prices Forecast with Conflict Period Insights")

# ========================================== Inputs =======================================
months_to_predict = st.slider("Select months into the future to forecast", 1, 120, 12)
conflict_selected = st.selectbox("Select a Conflict Period to Highlight", list(conflict_periods.keys()))
show_conflict = st.checkbox("Highlight Conflict Period on Graph", value=True)
show_trend = st.checkbox("Show Trend Line", value=True)
show_uncertainty = st.checkbox("Show Uncertainty Interval", value=True)
show_historical = st.checkbox("Show Historical Prices", value=True)
show_interest = st.checkbox("Show Average Mortgage Interest Rate Line", value=False)

# Date range selector for zooming
min_date = historical['Date'].min()
max_date = historical['Date'].max() + pd.DateOffset(months=months_to_predict)
date_range = st.date_input("Select Date Range to Display", [min_date, max_date])

# --- Forecast ---
future = model.make_future_dataframe(periods=months_to_predict, freq='M')
forecast = model.predict(future)

# Filter forecast and historical data by date range
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
forecast_filtered = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]
historical_filtered = historical[(historical['Date'] >= start_date) & (historical['Date'] <= end_date)]

# ====================================== Plot =========================================
fig, ax = plt.subplots(figsize=(12,6))

# Plot historical data
if show_historical:
    ax.plot(historical_filtered['Date'], historical_filtered['price'], 'k.', label='Historical Prices')

# Plot predicted yhat
ax.plot(forecast_filtered['ds'], forecast_filtered['yhat'], 'b-', label='Predicted')

# Plot trend if toggled
if show_trend:
    ax.plot(forecast_filtered['ds'], forecast_filtered['trend'], 'g--', label='Trend')

# Plot uncertainty intervals if toggled
if show_uncertainty:
    ax.fill_between(forecast_filtered['ds'], forecast_filtered['yhat_lower'], forecast_filtered['yhat_upper'], color='blue', alpha=0.2, label='Uncertainty Interval')

#  conflict period highlight
if show_conflict:
    conflict_start, conflict_end = pd.to_datetime(conflict_periods[conflict_selected])
    ax.axvspan(conflict_start, conflict_end, color='orange', alpha=0.3, label='Selected Conflict Period')

    # Conflict stats
    mask = (historical['Date'] >= conflict_start) & (historical['Date'] <= conflict_end)
    conflict_prices = historical.loc[mask, 'price']
    conflict_rates = historical.loc[mask, 'MORTGAGE30US']

    if not conflict_prices.empty:
        price_change_pct = 100 * (conflict_prices.iloc[-1] - conflict_prices.iloc[0]) / conflict_prices.iloc[0]
        avg_interest_rate = conflict_rates.mean()
        st.write(f"### Conflict Period: {conflict_selected}")
        st.write(f"- Home Price Change: {price_change_pct:.2f}%")
        st.write(f"- Average Interest Rate: {avg_interest_rate:.2f}%")

        # price change on graph
        ax.annotate(f'{price_change_pct:.2f}% Price Change', xy=(conflict_end, conflict_prices.iloc[-1]),
                    xytext=(conflict_end, conflict_prices.max()),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10)

# mortgage interest rate line
if show_interest:
    ax2 = ax.twinx()
    ax2.plot(historical_filtered['Date'], historical_filtered['MORTGAGE30US'], 'r-', label='Mortgage Interest Rate')
    ax2.set_ylabel('Mortgage Interest Rate (%)', color='r')
    ax2.tick_params(axis='y', colors='r')
    ax2.legend(loc='upper right')

ax.set_title("Fayetteville Home Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Home Price ($)")
ax.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Tab 2 - Address Comparison with RentCast
with tab2:
    st.subheader("Compare Property Value with Forecast on Selected Date")

    address = st.text_input("Enter Property Address", "8018 Christmas Ct, Charlotte, NC 28316")
    forecast_min = forecast['ds'].min()
    forecast_max = forecast['ds'].max()
    date_input = st.date_input("Select Forecast Date", forecast_max, min_value=forecast_min, max_value=forecast_max)

    forecast_start_val = pd.DataFrame()

    if address and date_input:
        value_data = fetch_property_value(address)

        if value_data:
            rentcast_price = value_data.get("price")
            st.metric("RentCast Estimated Value", f"${rentcast_price:,.0f}")

            selected_date = pd.to_datetime(date_input)
            closest_row = forecast.iloc[(forecast['ds'] - selected_date).abs().argsort()[:1]]
            model_price = closest_row['yhat'].values[0]
            model_date = closest_row['ds'].values[0]

            forecast_start_val = closest_row.copy()  # Store for use in Tab 3

            st.metric("Forecasted Value on Selected Date", f"${model_price:,.0f}")
            diff_pct = 100 * (rentcast_price - model_price) / model_price

            if diff_pct > 0:
                st.write(f"The property appears overvalued by {diff_pct:.2f}% compared to the model.")
            else:
                st.write(f"The property appears undervalued by {abs(diff_pct):.2f}% compared to the model.")

            filtered_forecast = forecast[forecast['ds'] >= selected_date]

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(filtered_forecast['ds'], filtered_forecast['yhat'], label="Forecast")
            ax2.fill_between(filtered_forecast['ds'], filtered_forecast['yhat_lower'], filtered_forecast['yhat_upper'], color='blue', alpha=0.2)
            ax2.axhline(y=rentcast_price, color='r', linestyle='--', label="RentCast Value")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price ($)")
            ax2.set_title("Forecast vs RentCast Value")
            ax2.legend()
            st.pyplot(fig2)

# Tab 3 - Mortgage Calculator
with tab3:
    st.subheader("Mortgage Calculator")

    if not forecast_start_val.empty:
        predicted_price = float(forecast_start_val.iloc[0]['yhat'])
    else:
        predicted_price = 250000  # Default if no date selected in tab 2

    st.write(f"Model Forecasted Price: ${predicted_price:,.0f}")

    down_payment_pct = st.slider("Down Payment (%)", 0, 100, 20)
    interest_rate = st.slider("Interest Rate (%)", 0.0, 10.0, 6.5, 0.1)
    loan_term_years = st.selectbox("Loan Term (Years)", [15, 20, 30], index=2)

    down_payment_amount = predicted_price * (down_payment_pct / 100)
    loan_amount = predicted_price - down_payment_amount
    monthly_rate = interest_rate / 100 / 12
    num_payments = loan_term_years * 12

    if monthly_rate > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments) / \
                          ((1 + monthly_rate) ** num_payments - 1)
    else:
        monthly_payment = loan_amount / num_payments

    st.subheader("Results")
    st.metric("Estimated Monthly Payment", f"${monthly_payment:,.2f}")
    st.write(f"Purchase Price: ${predicted_price:,.0f}")
    st.write(f"Down Payment: ${down_payment_amount:,.0f} ({down_payment_pct}%)")
    st.write(f"Loan Amount: ${loan_amount:,.0f}")
    st.write(f"Interest Rate: {interest_rate:.2f}%")
    st.write(f"Term: {loan_term_years} years")

# Optional CSV Download
st.markdown("---")
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
st.download_button(label="Download Full Forecast CSV", data=csv, file_name='forecast.csv', mime='text/csv')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import joblib
import requests
from datetime import datetime

# ========== Load Prophet Model and Forecast Data ==========
model = joblib.load('prophet_model.pkl')
combined_df = joblib.load('combined_df.pkl')  # optional mortgage + price data
forecast = model.predict(model.make_future_dataframe(periods=120, freq='M'))
forecast['ds'] = pd.to_datetime(forecast['ds'])

# ========== RentCast API Setup ==========
API_KEY = "d0cfdc1b86b44b9c8a51ad678456ee30"
HEADERS = {"X-Api-Key": API_KEY}
VALUE_URL = "https://api.rentcast.io/v1/avm/value"
RENT_URL = "https://api.rentcast.io/v1/avm/rent/long-term"

def fetch_property_value(address):
    response = requests.get(VALUE_URL, headers=HEADERS, params={"address": address})
    if response.status_code == 200:
        return response.json()
    st.error(f"Value API Error {response.status_code}: {response.text}")
    return None

def fetch_rent_estimate(address):
    response = requests.get(RENT_URL, headers=HEADERS, params={"address": address})
    if response.status_code == 200:
        return response.json()
    st.error(f"Rent API Error {response.status_code}: {response.text}")
    return None

# ========== Streamlit Layout ==========
st.set_page_config(page_title="Real Estate Forecasting", layout="wide")
st.title("Real Estate Forecasting & Mortgage Tools")

tab1, tab2, tab3 = st.tabs(["Forecast Explorer", "Address Comparison", "Mortgage Calculator"])

# ========== TAB 1: Forecast Explorer ==========
with tab1:
    st.header("Home Price Forecast")
    months_to_predict = st.slider("Forecast Months Ahead", 1, 120, 60)
    date_range = st.date_input("Select Date Range", [forecast['ds'].min(), forecast['ds'].max()])
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    forecast_filtered = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(forecast_filtered['ds'], forecast_filtered['yhat'], label="Forecast")
    ax1.fill_between(forecast_filtered['ds'], forecast_filtered['yhat_lower'], forecast_filtered['yhat_upper'], alpha=0.2)
    ax1.set_title("Forecasted Home Values")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    st.pyplot(fig1)

# ========== TAB 2: Address Comparison ==========
with tab2:
    st.header("Compare Property Value with Forecast")
    address = st.text_input("Enter Property Address", "8018 Christmas Ct, Charlotte, NC 28316")
    date_input = st.date_input("Select Forecast Start Date", forecast['ds'].max(),
                               min_value=forecast['ds'].min(), max_value=forecast['ds'].max())

    if address and date_input:
        value_data = fetch_property_value(address)
        rent_data = fetch_rent_estimate(address)

        if value_data and rent_data:
            rentcast_price = value_data.get("price", 0)
            est_rent = rent_data.get("rent", 0)
            st.metric("RentCast Estimated Value", f"${rentcast_price:,.0f}")
            st.metric("RentCast Estimated Rent", f"${est_rent:,.0f}")

            selected_date = pd.to_datetime(date_input)
            closest_row = forecast.iloc[(forecast['ds'] - selected_date).abs().argsort()[:1]]
            model_price = closest_row['yhat'].values[0]
            st.metric("Forecasted Value on Date", f"${model_price:,.0f}")

            diff_pct = 100 * (rentcast_price - model_price) / model_price
            if diff_pct > 0:
                st.success(f"The property appears overvalued by {diff_pct:.2f}%")
            else:
                st.info(f"The property appears undervalued by {abs(diff_pct):.2f}%")

            st.markdown("### Forecast from Selected Date")
            filtered_forecast = forecast[forecast['ds'] >= selected_date]
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(filtered_forecast['ds'], filtered_forecast['yhat'], label="Forecast")
            ax2.fill_between(filtered_forecast['ds'], filtered_forecast['yhat_lower'], filtered_forecast['yhat_upper'], alpha=0.2)
            ax2.axhline(y=rentcast_price, color='red', linestyle='--', label="RentCast Value")
            ax2.legend()
            st.pyplot(fig2)

# ========== TAB 3: Mortgage Calculator ==========
with tab3:
    st.header("Mortgage Calculator")
    col1, col2 = st.columns(2)

    with col1:
        home_price = st.number_input("Home Price ($)", value=model_price, step=1000.0)
        down_payment = st.number_input("Down Payment ($)", value=(model_price*.3), step=1000.0)
        interest_rate = st.number_input("Annual Interest Rate (%)", value=6.5, step=0.1)

    with col2:
        loan_term = st.selectbox("Loan Term (Years)", [15, 20, 30], index=2)
        property_tax = st.number_input("Annual Property Tax ($)", value=3000, step=100)
        insurance = st.number_input("Annual Insurance ($)", value=1500, step=100)

    loan_amount = home_price - down_payment
    monthly_rate = interest_rate / 100 / 12
    n_payments = loan_term * 12

    if monthly_rate > 0:
        monthly_pi = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    else:
        monthly_pi = loan_amount / n_payments

    monthly_tax = property_tax / 12
    monthly_insurance = insurance / 12
    monthly_total = monthly_pi + monthly_tax + monthly_insurance

    st.subheader("Monthly Payment Summary")
    st.write(f"**Principal & Interest:** ${monthly_pi:,.2f}")
    st.write(f"**Tax:** ${monthly_tax:,.2f}")
    st.write(f"**Insurance:** ${monthly_insurance:,.2f}")
    st.write(f"**Total Estimated Payment:** ${monthly_total:,.2f}")

    # Optional: Amortization chart
    show_amortization = st.checkbox("Show Amortization Schedule & Balance Curve")
    if show_amortization:
        balance = loan_amount
        balances = []
        for i in range(n_payments):
            interest = balance * monthly_rate
            principal = monthly_pi - interest
            balance -= principal
            balances.append(balance if balance > 0 else 0)

        st.line_chart(pd.DataFrame(balances, columns=["Remaining Balance"]))

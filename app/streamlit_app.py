from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.title("Customer Churn Prediction")
st.write("Enter customer information to predict whether the customer may churn.")

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=5)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox(
    "Multiple Lines", ["Yes", "No", "No phone service"]
)
internet_service = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)
online_security = st.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
online_backup = st.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)
device_protection = st.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
tech_support = st.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
streaming_tv = st.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
streaming_movies = st.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=200.0)

if st.button("Predict Churn"):
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        response = requests.post(
            API_URL,
            json=customer,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

        prediction = result["prediction"]
        probability = result["churn_probability"]

        if prediction == 1:
            st.warning("⚠️ Customer is likely to churn.")
        else:
            st.success("✅ Customer is likely to stay.")

        st.write(f"Churn Probability: {probability * 100:.2f}%")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to API: {e}")

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load trained model and scaler
model = load_model("model.h5")
scaler = pickle.load(open("ann_model.pkl", "rb"))

st.set_page_config(page_title="Bank Term Deposit Prediction", layout="centered")

st.title("🏦 Bank Term Deposit Prediction App")
st.write("Predict whether a customer will subscribe to a term deposit")

st.divider()

# ---- User Inputs ----
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job (Encoded)", list(range(0, 12)))
marital = st.selectbox("Marital Status (Encoded)", [0, 1, 2])
education = st.selectbox("Education Level (Encoded)", [0, 1, 2, 3])
balance = st.number_input("Account Balance", value=1000)
housing = st.selectbox("Housing Loan", [0, 1])
loan = st.selectbox("Personal Loan", [0, 1])
duration = st.number_input("Last Contact Duration (seconds)", value=100)
campaign = st.number_input("Number of Contacts in Campaign", value=1)

st.divider()

# ---- Prediction ----
if st.button("🔍 Predict"):
    input_data = np.array([[age, job, marital, education, balance,
                            housing, loan, duration, campaign]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]

    if prediction >= 0.5:
        st.success("✅ Customer WILL subscribe to Term Deposit")
    else:
        st.error("❌ Customer will NOT subscribe to Term Deposit")

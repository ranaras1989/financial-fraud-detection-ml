import streamlit as st
import pandas as pd
import joblib



# Load saved files


model = joblib.load("tuned_xgboost_fraud_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
final_threshold = joblib.load("final_threshold.pkl")



# Page configuration


st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="💳",
    layout="centered"
)



# App title


st.title("💳 Financial Fraud Detection App")

st.write("""
This app predicts whether a financial transaction is potentially fraudulent.
It uses a tuned XGBoost model trained on the PaySim financial transaction dataset.
""")



# User input section


st.header("Enter Transaction Details")

step = st.number_input(
    "Transaction Step",
    min_value=1,
    value=1
)

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

oldbalanceOrg = st.number_input(
    "Sender Old Balance",
    min_value=0.0,
    value=5000.0
)

newbalanceOrig = st.number_input(
    "Sender New Balance",
    min_value=0.0,
    value=4000.0
)

oldbalanceDest = st.number_input(
    "Receiver Old Balance",
    min_value=0.0,
    value=0.0
)

newbalanceDest = st.number_input(
    "Receiver New Balance",
    min_value=0.0,
    value=1000.0
)

isFlaggedFraud = st.selectbox(
    "Is Transaction Flagged by Existing Rule?",
    [0, 1]
)



# Create input dataframe


input_data = pd.DataFrame({
    "step": [step],
    "type": [transaction_type],
    "amount": [amount],
    "oldbalanceOrg": [oldbalanceOrg],
    "newbalanceOrig": [newbalanceOrig],
    "oldbalanceDest": [oldbalanceDest],
    "newbalanceDest": [newbalanceDest],
    "isFlaggedFraud": [isFlaggedFraud]
})



# Preprocess input


input_encoded = pd.get_dummies(
    input_data,
    columns=["type"],
    drop_first=True
)

input_encoded = input_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)



# Prediction


if st.button("Predict Fraud Risk"):

    fraud_probability = model.predict_proba(input_encoded)[:, 1][0]

    prediction = 1 if fraud_probability >= final_threshold else 0

    st.subheader("Prediction Result")

    st.write(f"Fraud Probability: **{fraud_probability:.4f}**")
    st.write(f"Selected Threshold: **{final_threshold}**")

    if prediction == 1:
        st.error("⚠️ High Risk: This transaction is predicted as Fraudulent.")
    else:
        st.success("✅ Low Risk: This transaction is predicted as Not Fraudulent.")

    st.subheader("Input Transaction Summary")
    st.dataframe(input_data)
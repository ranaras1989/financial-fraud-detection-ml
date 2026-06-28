from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tuned_xgboost_fraud_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"
THRESHOLD_PATH = BASE_DIR / "final_threshold.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    final_threshold = joblib.load(THRESHOLD_PATH)
    return model, feature_columns, final_threshold


st.set_page_config(
    page_title="Financial Fraud Detection App",
    layout="centered",
)

st.title("Financial Fraud Detection App")

st.write(
    """
    This app predicts whether a financial transaction is potentially fraudulent.
    It uses a tuned XGBoost model trained on the PaySim financial transaction dataset.
    """
)

try:
    model, feature_columns, final_threshold = load_artifacts()
except FileNotFoundError as error:
    st.error(f"Required model file is missing: {error.filename}")
    st.stop()

st.header("Enter Transaction Details")

step = st.number_input("Transaction Step", min_value=1, value=1)

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
)

amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)

oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=5000.0)

newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=4000.0)

oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)

newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=1000.0)

isFlaggedFraud = st.selectbox("Is Transaction Flagged by Existing Rule?", [0, 1])

input_data = pd.DataFrame(
    {
        "step": [step],
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest],
        "isFlaggedFraud": [isFlaggedFraud],
    }
)

input_encoded = pd.get_dummies(input_data, columns=["type"], drop_first=True)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

if st.button("Predict Fraud Risk"):
    fraud_probability = model.predict_proba(input_encoded)[:, 1][0]
    prediction = int(fraud_probability >= final_threshold)

    st.subheader("Prediction Result")
    st.write(f"Fraud Probability: **{fraud_probability:.4f}**")
    st.write(f"Selected Threshold: **{final_threshold:.2f}**")

    if prediction == 1:
        st.error("High Risk: This transaction is predicted as fraudulent.")
    else:
        st.success("Low Risk: This transaction is predicted as not fraudulent.")

    st.subheader("Input Transaction Summary")
    st.dataframe(input_data)

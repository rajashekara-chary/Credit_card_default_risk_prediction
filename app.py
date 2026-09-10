import streamlit as st
import pandas as pd
import pickle

# lodel binary code
with open("credit_model.pkl","rb")as file:
    model=pickle.load(file)

st.title("Credit Card Default Risk Prediction System")


# 💳 Credit Limit
st.subheader("💳 Credit Limit")

LIMIT_BAL = st.number_input("LIMIT_BAL", min_value=0)


# 📅 Repayment Status - Past 6 Months
st.subheader("📅 Repayment Status - Past 6 Months")
st.caption("Enter the repayment status for each period.")

PAY_0 = st.number_input("Most Recent Month → PAY_0")
PAY_2 = st.number_input("2 Months Ago → PAY_2")
PAY_3 = st.number_input("3 Months Ago → PAY_3")
PAY_4 = st.number_input("4 Months Ago → PAY_4")
PAY_5 = st.number_input("5 Months Ago → PAY_5")
PAY_6 = st.number_input("6 Months Ago → PAY_6")


# 🧾 Bill Statement Amounts - Past 6 Months
st.subheader("🧾 Bill Statement Amounts - Past 6 Months")
st.caption("Enter the bill amount for each period.")

BILL_AMT1 = st.number_input("Most Recent Month → BILL_AMT1")
BILL_AMT2 = st.number_input("2 Months Ago → BILL_AMT2")
BILL_AMT3 = st.number_input("3 Months Ago → BILL_AMT3")
BILL_AMT4 = st.number_input("4 Months Ago → BILL_AMT4")
BILL_AMT5 = st.number_input("5 Months Ago → BILL_AMT5")
BILL_AMT6 = st.number_input("6 Months Ago → BILL_AMT6")


# 💰 Payment Amounts - Past 6 Months
st.subheader("💰 Payment Amounts - Past 6 Months")
st.caption("Enter the payment amount made during each period.")

PAY_AMT1 = st.number_input("Most Recent Month → PAY_AMT1")
PAY_AMT2 = st.number_input("2 Months Ago → PAY_AMT2")
PAY_AMT3 = st.number_input("3 Months Ago → PAY_AMT3")
PAY_AMT4 = st.number_input("4 Months Ago → PAY_AMT4")
PAY_AMT5 = st.number_input("5 Months Ago → PAY_AMT5")
PAY_AMT6 = st.number_input("6 Months Ago → PAY_AMT6")




import pandas as pd

input_data = pd.DataFrame({
    'LIMIT_BAL': [LIMIT_BAL],

    # Repayment Status
    'PAY_0': [PAY_0],
    'PAY_2': [PAY_2],
    'PAY_3': [PAY_3],
    'PAY_4': [PAY_4],
    'PAY_5': [PAY_5],
    'PAY_6': [PAY_6],

    # Bill Statement Amounts
    'BILL_AMT1': [BILL_AMT1],
    'BILL_AMT2': [BILL_AMT2],
    'BILL_AMT3': [BILL_AMT3],
    'BILL_AMT4': [BILL_AMT4],
    'BILL_AMT5': [BILL_AMT5],
    'BILL_AMT6': [BILL_AMT6],

    # Previous Payments
    'PAY_AMT1': [PAY_AMT1],
    'PAY_AMT2': [PAY_AMT2],
    'PAY_AMT3': [PAY_AMT3],
    'PAY_AMT4': [PAY_AMT4],
    'PAY_AMT5': [PAY_AMT5],
    'PAY_AMT6': [PAY_AMT6]
})




if st.button('Predict'):
    prediction=model.predict(input_data)

    if prediction[0] in [1,"Y"]:
        st.write("Customer is NOT paying properly  ❌")
    else:
        st.write('Customer is paying properly ✅')

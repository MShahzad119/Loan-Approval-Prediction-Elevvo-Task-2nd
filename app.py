# This is the updated Streamlit app for Loan Approval Prediction
# Save this as app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# Load the trained model using pickle
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# App title
st.title("💰 Loan Approval Prediction App")
st.markdown("""
Predict whether a loan application will be **approved or rejected** based on applicant details and financial information.
""")

# Sidebar for input
st.sidebar.header("Applicant Information")

def user_input_features():
    no_of_dependents = st.sidebar.number_input('Number of Dependents', 0, 10, 0)
    education = st.sidebar.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.sidebar.selectbox('Self Employed', ['Yes', 'No'])
    income_annum = st.sidebar.number_input('Annual Income ($)', 0, 2000000, 50000)
    loan_amount = st.sidebar.number_input('Loan Amount ($)', 0, 2000000, 100000)
    loan_term = st.sidebar.number_input('Loan Term (Months)', 12, 360, 120)
    cibil_score = st.sidebar.number_input('CIBIL Score', 300, 900, 700)
    residential_assets_value = st.sidebar.number_input('Residential Assets Value ($)', 0, 2000000, 50000)
    commercial_assets_value = st.sidebar.number_input('Commercial Assets Value ($)', 0, 2000000, 50000)
    luxury_assets_value = st.sidebar.number_input('Luxury Assets Value ($)', 0, 2000000, 50000)
    bank_asset_value = st.sidebar.number_input('Bank Asset Value ($)', 0, 2000000, 50000)
    
    # Encode categorical variables
    education_enc = 1 if education.strip() == 'Graduate' else 0
    self_employed_enc = 1 if self_employed.strip() == 'Yes' else 0

    data = np.array([[no_of_dependents, education_enc, self_employed_enc, income_annum,
                      loan_amount, loan_term, cibil_score, residential_assets_value,
                      commercial_assets_value, luxury_assets_value, bank_asset_value]])
    return data

input_data = user_input_features()

st.subheader("Applicant Input Details")
st.write("Below are the details entered by the applicant:")
st.write(pd.DataFrame(input_data, columns=[
    'Dependents', 'Education', 'Self Employed', 'Income', 'Loan Amount', 'Loan Term',
    'CIBIL Score', 'Residential Assets', 'Commercial Assets', 'Luxury Assets', 'Bank Assets'
]))

# Predict button
if st.button('Predict Loan Status'):
    with st.spinner('Predicting...'):
        time.sleep(1)  # small delay for effect
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]*100
    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {probability:.2f}%)")
        st.balloons()  # fun effect
    else:
        st.error(f"❌ Loan Rejected! (Confidence: {probability:.2f}%)")

# Footer
st.markdown("---")
st.markdown("Developed by **Shahzad** | Data Analytics & ML Enthusiast")

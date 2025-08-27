# Loan Approval Prediction

This project predicts whether a loan application will be approved based on applicant information and financial details.

## Dataset
The dataset contains the following columns:
- `loan_id`
- `no_of_dependents`
- `education`
- `self_employed`
- `income_annum`
- `loan_amount`
- `loan_term`
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`
- `loan_status` (Target: Approved/Rejected)

## Project Overview
- Handle missing values
- Encode categorical features
- Train classification models (Logistic Regression & Decision Tree)
- Evaluate model performance (Precision, Recall, F1-score)
- Handle imbalanced data using SMOTE

## Files
- `loan_approval.ipynb`: Jupyter Notebook for EDA, preprocessing, and model training
- `loan_model.pkl`: Saved trained model
- `app.py`: Streamlit app for deployment
- `loan.csv`: Dataset file
- `requirements.txt`: Required Python packages

## Deployment
Run the Streamlit app:

```bash
streamlit run app.py

import streamlit as st
import pandas as pd
import pickle
from PhishingPredictor import EmailPreProcessor
from sklearn.preprocessing import FunctionTransformer

def combine_subject_body(x):
    return x['subject'].fillna('') + ' ' + x['body'].fillna('')

# load pickle
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Phishing Email Detector")

# User input fields
subject = st.text_input("Email Subject")
body = st.text_area("Email Body")
sender = st.text_input("Sender Email")
date = st.date_input("Date Received")

# Predict button
if st.button("Check for Phishing"):
    # Format input as DataFrame for your model
    input_data = pd.DataFrame([{
        "subject": subject,
        "body": body,
        "sender": sender,
        "date": pd.to_datetime(date)
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.error("This email is probably a phishing attempt ask for help")
    else:
        st.success("This email is probably safe")

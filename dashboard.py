import streamlit as st
import requests
import json

# Streamlit App
st.title("Customer Segmentation Model")
st.write("Enter customer details below to predict their cluster.")

# Input Fields
gender = st.number_input("Gender", min_value=0, max_value=1, value=0)
age = st.number_input("Age", min_value=18, max_value=100, value=25)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=150, value=50)
spending = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Predict Button
if st.button('Predict Cluster'):
    # Prepare the input data
    input_data = {"features": [gender, age, income, spending]}
    
    # Make the POST request to Flask API
    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(input_data)
        )
        result = response.json()

        if "cluster" in result:
            st.success(f"Predicted Cluster: {result['cluster']}")
        else:
            st.error(f"Error: {result['error']}")
    except Exception as e:
        st.error(f"Failed to connect to API. Error: {str(e)}")
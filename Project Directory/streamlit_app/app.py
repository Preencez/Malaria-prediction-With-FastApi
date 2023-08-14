import streamlit as st
import requests

st.title("Malaria Prediction App")

# User input for symptoms
symptoms = st.multiselect("Select symptoms:", ["Fever", "Headache", "Fatigue", "Nausea", "Muscle Pain"])

# Button to make prediction
if st.button("Predict"):
    data = {"symptoms": symptoms}
    response = requests.post("http://localhost:8000/predict", json=data)
    result = response.json()

    # Display prediction result
    st.write("Prediction:", result["prediction"])

import streamlit as st
import joblib
import numpy as np

# Load the categorical imputer, label encoder, and Random Forest model
categorical_imputer_filepath = "D:/Projects/Malaria prediction With FastApi/Project Directory/Ml components/categorical_imputer.joblib"
label_encoder_filepath = "D:/Projects/Malaria prediction With FastApi/Project Directory/Ml components/label_encoder.joblib"
best_rf_model_filepath = "D:/Projects/Malaria prediction With FastApi/Project Directory/Ml components/best_rf_model.joblib"

categorical_imputer = joblib.load(categorical_imputer_filepath)
label_encoder = joblib.load(label_encoder_filepath)
model = joblib.load(best_rf_model_filepath)

# Streamlit app
def main():
    st.title("Malaria Prediction App")
    st.write("This app predicts whether a person has malaria based on input features.")

    # Input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    fever = st.checkbox("Fever")
    fatigue = st.checkbox("Fatigue")
    cough = st.checkbox("Cough")

    # Process input data
    input_data = {
        "gender": np.array([gender]),
        "fever": np.array([fever]),
        "fatigue": np.array([fatigue]),
        "cough": np.array([cough])
    }

    # Encode categorical features
    input_data["gender"] = label_encoder.transform(input_data["gender"])
    
    # Make prediction
    prediction = model.predict_proba(input_data)

    # Display prediction
    st.subheader("Prediction")
    if prediction[0][1] > 0.5:
        st.write("Likely to have malaria.")
    else:
        st.write("Unlikely to have malaria.")

if __name__ == "__main__":
    main()

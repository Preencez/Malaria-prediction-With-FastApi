from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the model and other components
categorical_imputer_filepath = "Ml components/categorical_imputer.joblib"
label_encoder_filepath = "Ml components/label_encoder.joblib"
best_rf_model_filepath = "Ml components/best_rf_model.joblib"
scaler_filepath = "Ml components/scaler.joblib"

categorical_imputer = joblib.load(categorical_imputer_filepath)
label_encoder = joblib.load(label_encoder_filepath)
best_rf_model = joblib.load(best_rf_model_filepath)
scaler = joblib.load(scaler_filepath)

class PredictionInput(BaseModel):
    gender: str
    fever: bool
    fatigue: bool
    cough: bool

@app.post("/predict/")
async def predict(data: PredictionInput):
    # Process input data
    input_data = {
        "gender": np.array([data.gender]),
        "fever": np.array([data.fever]),
        "fatigue": np.array([data.fatigue]),
        "cough": np.array([data.cough])
    }

    # Impute missing values and encode categorical features
    input_data["gender"] = label_encoder.transform(input_data["gender"])
    
    # Make prediction
    prediction = best_rf_model.predict_proba(input_data)
    
    # Return prediction result
    return {"prediction": "Likely" if prediction[0][1] > 0.5 else "Unlikely"}

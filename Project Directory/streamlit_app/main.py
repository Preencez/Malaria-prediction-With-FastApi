import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load saved components
scaler = joblib.load(r"D:\Projects\Malaria prediction With FastApi\Project Directory\Ml components\scaler.pkl")
label_encoder = joblib.load(r"D:\Projects\Malaria prediction With FastApi\Project Directory\Ml components\label_encoder.pkl")
encoded_targets = joblib.load(r"D:\Projects\Malaria prediction With FastApi\Project Directory\Ml components\encoded_targets.pkl")
model = joblib.load(r"D:\Projects\Malaria prediction With FastApi\Project Directory\Ml components\random_forest_model.pkl")

class SymptomInput(BaseModel):
    symptoms: List[str]

@app.post("/predict")
async def predict_symptoms(symptom_input: SymptomInput):
    symptoms = symptom_input.symptoms

    # Preprocess input and make prediction
    scaled_data = scaler.transform([symptoms])
    encoded_data = label_encoder.transform(["Positive"])  # Dummy label for encoding
    prediction = model.predict(scaled_data)

    # Inverse transform the prediction
    prediction_label = label_encoder.inverse_transform(prediction)[0]

    return {"prediction": prediction_label}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

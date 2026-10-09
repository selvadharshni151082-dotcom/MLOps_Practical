import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_breast_cancer

app = FastAPI(title="Cancer Prediction API")

model = joblib.load("models/model.pkl")
dataset = load_breast_cancer(as_frame=True)
feature_names = list(dataset.data.columns)


class PredictionInput(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "ML Model API is running"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    if len(input_data.features) != len(feature_names):
        return {
            "error": f"Expected {len(feature_names)} features"
        }

    data = pd.DataFrame(
        [input_data.features],
        columns=feature_names
    )

    prediction = int(model.predict(data)[0])

    return {
        "prediction": prediction,
        "result": "Malignant" if prediction == 0 else "Benign"
    }
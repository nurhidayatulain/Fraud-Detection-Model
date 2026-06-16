from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/fraud_model.pkl")

@app.post("/predict")
def predict(transaction: dict):

    df = pd.DataFrame([transaction])

    prediction = int(model.predict(df)[0])

    return {
        "fraud_prediction": prediction
    }

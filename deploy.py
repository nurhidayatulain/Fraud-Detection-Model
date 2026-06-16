from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("models/fraud_model.pkl")

@app.post("/predict")
def predict(transaction: dict):

    df = pd.DataFrame([transaction])

    fraud_probability = float(
        model.predict_proba(df)[0][1]
    )

    prediction = int(
        model.predict(df)[0]
    )

    return {
        "fraud_prediction": prediction,
        "fraud_probability": round(fraud_probability, 3)
    }

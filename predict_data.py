import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")

sample = pd.DataFrame([{
    "amount": 6000,
    "txn_count_24h": 18,
    "new_device": 1,
    "country_risk": 1
}])

prediction = model.predict(sample)

print(prediction)

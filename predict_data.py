
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# New transaction
transaction = pd.DataFrame([{
    "amount": 8500,
    "txn_count_24h": 22,
    "new_device": 1,
    "country_risk": 1
}])

# Predict
prediction = model.predict(transaction)[0]

# Fraud probability
probability = model.predict_proba(transaction)[0][1]

print("Prediction:", prediction)
print("Fraud Probability:", round(probability, 3))

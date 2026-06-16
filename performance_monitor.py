import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

df = pd.read_csv("data/transactions.csv")

X = df.drop("is_fraud", axis=1)
y_true = df["is_fraud"]

model = joblib.load("models/fraud_model.pkl")

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:,1]

print("Precision:", round(precision_score(y_true,y_pred),3))
print("Recall:", round(recall_score(y_true,y_pred),3))
print("F1:", round(f1_score(y_true,y_pred),3))
print("ROC-AUC:", round(roc_auc_score(y_true,y_prob),3))

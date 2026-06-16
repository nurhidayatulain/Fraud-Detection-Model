# monitoring/drift_monitor.py

import pandas as pd

df = pd.read_csv("data/transactions.csv")

training_average_amount = 500

current_average_amount = df["amount"].mean()

change_pct = (
    (current_average_amount - training_average_amount)
    / training_average_amount
) * 100

print("="*50)
print("DATA DRIFT REPORT")
print("="*50)

print(f"Training Avg Amount : {training_average_amount:.2f}")
print(f"Current Avg Amount  : {current_average_amount:.2f}")
print(f"Change Percentage   : {change_pct:.2f}%")

if abs(change_pct) > 20:
    print("ALERT: Potential Data Drift Detected")
else:
    print("No Significant Drift")

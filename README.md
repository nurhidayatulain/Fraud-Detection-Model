# Fraud-Detection-Model
Fraud Detection Model end-to-end process flow

## Objective
Detect potentially fraudulent transactions in real time

## Business Goals:
• Reduce fraud losses
• Prioritize investigations
• Improve operational efficiency

## Dataset
Transactions Dataset

## Challenges
Highly imbalanced data.

## Techniques

- Feature Engineering
- Random Forest
- Logistic regression --> used as a benchmark model because it is simple, interpretable, and easy to explain to business users and model validators.
- XGBoost --> XGBoost delivered the strongest predictive performance, effectively captured complex fraud patterns, and is widely used in financial services for structured tabular data.

## Training data
- Random forest Handles nonlinear patterns and tabular data.

## Predict data
Score a new transaction

Input:
RM8,500 amount
22 transactions/day
New device
High-risk country

Output --> score and fraud probability

## Metrics
Ensures model remains effective after deployment

- Precision
- Recall
- F1 Score
- ROC-AUC

## Deployment

Real-time scoring through REST API

## Drift Monitoring
Data Drift --> input data changes compared to what the model was trained on
Model/Concept drift --> occurs when the underlying relationship between features and the target variable changes over time

- Average transaction amount changes significantly
- Potential trigger for retraining

## Explanaibility with SHAP
When?
After model training.

Why?
To explain model predictions.

Tools?
Python SHAP library.

Outputs?
• Feature importance
• Individual prediction explanation
• Global model explanation

Business Benefit?
Transparency, governance, and trust.

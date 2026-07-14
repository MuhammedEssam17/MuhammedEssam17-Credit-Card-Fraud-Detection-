# Credit Card Fraud Detection using Isolation Forest

## Overview

This project detects fraudulent credit card transactions using the **Isolation Forest** algorithm, an unsupervised anomaly detection technique.

Unlike supervised learning, the model is trained only on **normal transactions** and learns the pattern of normal behavior. Any transaction that significantly differs from these patterns is flagged as a potential fraud.

---

## Dataset

The project uses the Credit Card Fraud Detection dataset, which contains:

- 284,807 transactions
- 30 input features
- 492 fraudulent transactions
- Highly imbalanced data

Target labels:

- 0 = Normal transaction
- 1 = Fraudulent transaction

---

## Project Workflow

1. Load the dataset.
2. Separate normal and fraudulent transactions.
3. Train the model using only normal transactions.
4. Split the normal data:
   - 80% for training
   - 20% for testing
5. Combine the remaining normal transactions with all fraud transactions.
6. Remove unnecessary columns (`Class` to let the model learn from the features and `Time` because it's not a relevant feature for anomaly detection).
7. Scale the features using `StandardScaler`.
8. Train an `IsolationForest` model.
9. Predict anomalies on the test set.
10. Evaluate the model using:
    - Confusion Matrix
    - Precision
    - Recall
    - F1-score

---

## Why Isolation Forest?

Fraud detection is an anomaly detection problem because fraudulent transactions are rare compared to normal transactions.

Isolation Forest isolates unusual observations instead of learning from labeled fraud examples, making it well suited for highly imbalanced datasets.

---

## Libraries Used

- Pandas
- NumPy
- Scikit-learn

---

## Results

Model Performance:

- Accuracy: 99%
- Precision (Fraud): 0.60
- Recall (Fraud): 0.29
- F1-score (Fraud): 0.39

Confusion Matrix:

|               | Predicted Normal | Predicted Fraud |
|---------------|-----------------:|----------------:|
| Actual Normal | 56,767           | 96              |
| Actual Fraud  | 348              | 144             |

## Dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
import numpy as np

df = pd.read_csv("creditcard.csv")
print(df.head())

print("\n-----------------------------------------")

# Display the number of rows and columns in the dataset
print("Number of row and columns in the dataset:", df.shape)

# This tell how many normal(0) and fraud(1) transactions are there in the dataset
class_count = df["Class"].value_counts()
print(f"\nTransaction counts by class: {class_count}")

print("\n0 = Normal transactions, 1 = Fraud transactions")

print("\n-----------------------------------------")
# Seperate the dataset into normal and fraud transactions
normal_transactions = df[df["Class"] == 0]
fraud_transactions = df[df["Class"] == 1]

print("Now create a new seperate tables")
print(f"Normal transactions: {normal_transactions.shape}")
print(f"Fraud transactions: {fraud_transactions.shape}")
print("\n-----------------------------------------")

# Train the model on 80% of normal transactions only
normal_shuffeled = normal_transactions.sample(frac=1, random_state=42)
split_index = int(len(normal_shuffeled) * 0.8)
X_train = normal_shuffeled.iloc[:split_index]

# Grab the remaining 20% of normal transactions
X_remaining = normal_shuffeled.iloc[split_index:]

# Combine the remaining normal transactions with all fraud transactions
X_test = pd.concat([X_remaining, fraud_transactions])

# Save the combined validation labels for testing the model
y_test = X_test["Class"]

# Drop the "Class" and "Time" columns from the training and combined datasets
# We only want the model to learn from the mathematical features (like V1, V2, Amount)
X_train = X_train.drop(columns=["Class", "Time"])
X_test = X_test.drop(columns=["Class", "Time"])
print(f"\nTraining dataset shape: {X_train.shape}")
print(f"Testing dataset shape: {X_test.shape}")

scaler = StandardScaler()
# Scale the training and combined datasets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

contamination_rate = len(fraud_transactions) / len(df)
isolation_forest = IsolationForest(contamination=contamination_rate, random_state=42)
# Training the model on the normal transactions
isolation_forest.fit(X_train_scaled)
# 1 = normal and -1 = anomaly (fraud)
predictions = isolation_forest.predict(X_test_scaled)
# Convert the predictions to 0 = normal and 1 = anomaly (fraud)
predictions = np.where(predictions == -1, 1, 0)

# compare the predictions with the actual labels
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
# Show the classification report for precision, recall scores
print("\nClassification Report:")
print(classification_report(y_test, predictions))
print(f"\nPredicted anomalies: {np.sum(predictions == 1)}")
print(f"Actual frauds: {np.sum(y_test == 1)}")
print(f"Contamination rate: {contamination_rate:.4f}")

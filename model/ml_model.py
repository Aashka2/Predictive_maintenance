import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder


file_path = "/Users/aashkagupta/Desktop/Mini_project_semVI/model/predictive_maintenance.csv"
df = pd.read_csv(file_path)

print("Dataset Info:\n", df.info())
print("\nFirst 5 rows:\n", df.head())


df.dropna(inplace=True)
df.columns = df.columns.str.strip()


target_column = "Target"
if target_column not in df.columns:
    raise ValueError(
        f"Column '{target_column}' not found in dataset. Update script with the correct target column.")

X = df.drop(columns=["Failure Type", target_column])
y = df[target_column]


categorical_cols = ["Product ID", "Type"]

# One-Hot Encoding for categorical features
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Ensure `model` is properly trained before saving
joblib.dump(
    model, "/Users/aashkagupta/Desktop/Mini_project_semVI/model/trained_model.pkl")

print("Model saved successfully!")

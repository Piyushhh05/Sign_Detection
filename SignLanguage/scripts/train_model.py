
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Path to CSV (adjust if needed)
csv_path = "data/hand_landmarks.csv"

# Load data
columns = [f"{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']] + ["label"]
df = pd.read_csv(csv_path, names=columns)

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"✅ Model trained! Accuracy: {accuracy * 100:.2f}%")

# Save the model
os.makedirs("scripts", exist_ok=True)
model_path = "scripts/sign_model.pkl"
joblib.dump(clf, model_path)
print(f"✅ Model saved to {model_path}")